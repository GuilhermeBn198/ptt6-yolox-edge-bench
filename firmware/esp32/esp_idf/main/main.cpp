#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_system.h"
#include "esp_log.h"
#include "driver/gpio.h"

#include "model_data.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

static const char *TAG = "YOLOX_BENCHMARK";

// Sincronização via GPIO conforme metodologia PTT6 (Protocolo v1.3)
#define BENCHMARK_GPIO GPIO_NUM_4

// Configurações do Protocolo Experimental v1.3 (3 blocos x 10 execuções = 30 válidas)
static constexpr int kNumBlocks = 3;
static constexpr int kRunsPerBlock = 10;
static constexpr int kWarmupPerBlock = 2;
static constexpr int kInterRunDelayMs = 100;
static constexpr int kBlockCooldownMs = 2000;

static constexpr float kConfThr = 0.25f;
static constexpr float kNmsThr = 0.45f;
static constexpr int kMaxDetections = 100;
static constexpr int kNumClasses = 11;

static const char *kClassNames[kNumClasses] = {
    "helmet", "gloves", "vest", "boots", "goggles",
    "none", "Person", "no_helmet", "no_goggle", "no_gloves", "no_boots"
};

// Arena de ativações na PSRAM (4 MB alocados na Octal PSRAM externa)
static constexpr size_t kTensorArenaSize = 4 * 1024 * 1024;
static uint8_t *g_tensor_arena = nullptr;

// Imagem de teste binária (416x416x3 uint8 RGB letterbox) embutida via linker
extern const uint8_t _binary_test_image_bin_start[] asm("_binary_test_image_bin_start");
extern const uint8_t _binary_test_image_bin_end[]   asm("_binary_test_image_bin_end");

struct Detection {
    float x1, y1, x2, y2;
    float score;
    int cls_id;
};

static void init_gpio(void) {
    gpio_config_t io_conf = {};
    io_conf.intr_type = GPIO_INTR_DISABLE;
    io_conf.mode = GPIO_MODE_OUTPUT;
    io_conf.pin_bit_mask = (1ULL << BENCHMARK_GPIO);
    io_conf.pull_down_en = GPIO_PULLDOWN_ENABLE;
    io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
    gpio_config(&io_conf);
    gpio_set_level(BENCHMARK_GPIO, 0);
}

static float box_iou(const Detection &a, const Detection &b) {
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);
    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;
    float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-9f);
}

// Fase 1 (t_pre): Cópia e quantização do tensor 416x416x3 uint8 letterbox -> int8
static void preprocess_tensor(const uint8_t *src_rgb, size_t n_bytes, int8_t *dst_tensor, float in_scale, int32_t in_zp) {
    if (std::fabs(in_scale - 1.0f) < 1e-5f && in_zp == -128) {
        for (size_t i = 0; i < n_bytes; i++) {
            dst_tensor[i] = static_cast<int8_t>(static_cast<int>(src_rgb[i]) - 128);
        }
    } else {
        float inv_scale = 1.0f / in_scale;
        for (size_t i = 0; i < n_bytes; i++) {
            int32_t q = static_cast<int32_t>(std::round(static_cast<float>(src_rgb[i]) * inv_scale)) + in_zp;
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            dst_tensor[i] = static_cast<int8_t>(q);
        }
    }
}

// Fase 3 (t_post): Dequantização + Decodificação das 3549 âncoras YOLOX + NMS por classe
static std::vector<Detection> postprocess_yolox(
    const int8_t *raw_out,
    int target_h,
    int target_w,
    float out_scale,
    int32_t out_zp,
    float conf_thr,
    float nms_thr
) {
    static const int strides[3] = {8, 16, 32};
    std::vector<Detection> candidates_by_class[kNumClasses];

    int anchor_idx = 0;
    for (int s_idx = 0; s_idx < 3; s_idx++) {
        int stride = strides[s_idx];
        int gh = target_h / stride;
        int gw = target_w / stride;
        for (int gy = 0; gy < gh; gy++) {
            for (int gx = 0; gx < gw; gx++) {
                const int8_t *row = raw_out + anchor_idx * (5 + kNumClasses);
                anchor_idx++;

                float obj_conf = (static_cast<float>(row[4]) - static_cast<float>(out_zp)) * out_scale;
                if (obj_conf <= 0.0f) {
                    continue;
                }

                // Verificar se alguma classe supera conf_thr antes de calcular std::exp
                bool any_valid = false;
                float cls_scores[kNumClasses];
                for (int c = 0; c < kNumClasses; c++) {
                    float c_prob = (static_cast<float>(row[5 + c]) - static_cast<float>(out_zp)) * out_scale;
                    float sc = obj_conf * c_prob;
                    cls_scores[c] = sc;
                    if (sc >= conf_thr) {
                        any_valid = true;
                    }
                }
                if (!any_valid) {
                    continue;
                }

                float dx = (static_cast<float>(row[0]) - static_cast<float>(out_zp)) * out_scale;
                float dy = (static_cast<float>(row[1]) - static_cast<float>(out_zp)) * out_scale;
                float dw = (static_cast<float>(row[2]) - static_cast<float>(out_zp)) * out_scale;
                float dh = (static_cast<float>(row[3]) - static_cast<float>(out_zp)) * out_scale;

                if (dw < -10.0f) dw = -10.0f;
                if (dw > 10.0f) dw = 10.0f;
                if (dh < -10.0f) dh = -10.0f;
                if (dh > 10.0f) dh = 10.0f;

                float cx = (dx + static_cast<float>(gx)) * static_cast<float>(stride);
                float cy = (dy + static_cast<float>(gy)) * static_cast<float>(stride);
                float w  = std::exp(dw) * static_cast<float>(stride);
                float h  = std::exp(dh) * static_cast<float>(stride);

                float x1 = cx - w * 0.5f;
                float y1 = cy - h * 0.5f;
                float x2 = cx + w * 0.5f;
                float y2 = cy + h * 0.5f;

                for (int c = 0; c < kNumClasses; c++) {
                    if (cls_scores[c] >= conf_thr) {
                        candidates_by_class[c].push_back({x1, y1, x2, y2, cls_scores[c], c});
                    }
                }
            }
        }
    }

    // NMS class-aware idêntico ao benchmark_yolox.py
    std::vector<Detection> final_dets;
    for (int c = 0; c < kNumClasses; c++) {
        auto &cands = candidates_by_class[c];
        if (cands.empty()) continue;

        std::sort(cands.begin(), cands.end(), [](const Detection &a, const Detection &b) {
            return a.score > b.score;
        });

        std::vector<bool> suppressed(cands.size(), false);
        for (size_t i = 0; i < cands.size(); i++) {
            if (suppressed[i]) continue;
            final_dets.push_back(cands[i]);
            if (static_cast<int>(final_dets.size()) >= kMaxDetections) break;
            for (size_t j = i + 1; j < cands.size(); j++) {
                if (!suppressed[j] && box_iou(cands[i], cands[j]) > nms_thr) {
                    suppressed[j] = true;
                }
            }
        }
    }
    return final_dets;
}

extern "C" void app_main(void) {
    vTaskDelay(pdMS_TO_TICKS(1500));
    printf("\n=======================================================================\n");
    printf("  Benchmark YOLOX-Nano INT8 — ESP32-S3 (Protocolo Experimental v1.3)   \n");
    printf("=======================================================================\n");

    // 1. Diagnóstico de Memória antes da alocação
    size_t free_internal_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t free_psram_before = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("SRAM Interna Livre (boot): %u bytes (%.2f KiB)\n",
           (unsigned)free_internal_before, free_internal_before / 1024.0);
    printf("PSRAM Octal Livre  (boot): %u bytes (%.2f KiB)\n",
           (unsigned)free_psram_before, free_psram_before / 1024.0);

    if (free_psram_before < kTensorArenaSize) {
        ESP_LOGE(TAG, "ERRO: PSRAM insuficiente (%u bytes) para arena de %u bytes!",
                 (unsigned)free_psram_before, (unsigned)kTensorArenaSize);
        return;
    }

    // 2. Alocação da Tensor Arena em PSRAM
    g_tensor_arena = static_cast<uint8_t *>(
        heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
    );
    if (g_tensor_arena == nullptr) {
        ESP_LOGE(TAG, "ERRO: Falha ao alocar tensor_arena de %u bytes na PSRAM!", (unsigned)kTensorArenaSize);
        return;
    }

    init_gpio();

    // 3. Carregar o Modelo TFLite INT8
    const tflite::Model *model = tflite::GetModel(g_yolox_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Versao do schema incompativel!");
        return;
    }

    // 4. Configurar o Interpretador TFLM com operadores do YOLOX-Nano
    static tflite::MicroMutableOpResolver<20> resolver;
    resolver.AddAdd();
    resolver.AddConcatenation();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddLogistic();
    resolver.AddMaxPool2D();
    resolver.AddMul();
    resolver.AddPad();
    resolver.AddQuantize();
    resolver.AddReshape();
    resolver.AddResizeNearestNeighbor();
    resolver.AddStridedSlice();
    resolver.AddTranspose();

    static tflite::MicroInterpreter interpreter(model, resolver, g_tensor_arena, kTensorArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "ERRO: Falha em AllocateTensors()!");
        return;
    }

    TfLiteTensor *input_tensor = interpreter.input(0);
    TfLiteTensor *output_tensor = interpreter.output(0);
    size_t arena_used = interpreter.arena_used_bytes();
    size_t free_internal_after = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t free_psram_after = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

    int target_h = input_tensor->dims->data[1];
    int target_w = input_tensor->dims->data[2];
    float in_scale = input_tensor->params.scale != 0.0f ? input_tensor->params.scale : 1.0f;
    int32_t in_zp = input_tensor->params.zero_point;
    float out_scale = output_tensor->params.scale != 0.0f ? output_tensor->params.scale : 1.0f;
    int32_t out_zp = output_tensor->params.zero_point;

    size_t img_bytes = _binary_test_image_bin_end - _binary_test_image_bin_start;

    printf("Runtime       : esp-tflite-micro (ESP-NN SIMD Xtensa LX7)\n");
    printf("Modelo        : yolox_nano_ppe_int8.tflite (%u bytes / %.3f MiB)\n",
           (unsigned)g_yolox_model_data_len, g_yolox_model_data_len / (1024.0 * 1024.0));
    printf("Resolucao     : %dx%d | in_scale=%.6f, in_zp=%ld | out_scale=%.6f, out_zp=%ld\n",
           target_w, target_h, in_scale, (long)in_zp, out_scale, (long)out_zp);
    printf("Arena TFLM    : %u bytes (%.2f KiB / %.3f MiB)\n",
           (unsigned)arena_used, arena_used / 1024.0, arena_used / (1024.0 * 1024.0));
    printf("MEM_FOOTPRINT,%u,%u,%u,%u\n",
           (unsigned)g_yolox_model_data_len, (unsigned)arena_used,
           (unsigned)free_internal_after, (unsigned)free_psram_after);
    fflush(stdout);

    // 5. Execução de Sanidade / Previsão (E1) + Impressão das Detecções
    printf("\n=== PREDICTION_CHECK_START ===\n");
    int64_t tp0 = esp_timer_get_time();
    preprocess_tensor(_binary_test_image_bin_start, std::min(img_bytes, input_tensor->bytes),
                      input_tensor->data.int8, in_scale, in_zp);
    int64_t tp1 = esp_timer_get_time();
    interpreter.Invoke();
    int64_t tp2 = esp_timer_get_time();
    std::vector<Detection> init_dets = postprocess_yolox(
        output_tensor->data.int8, target_h, target_w, out_scale, out_zp, kConfThr, kNmsThr
    );
    int64_t tp3 = esp_timer_get_time();

    printf("FIRST_RUN_TIMING_US,t_pre=%lld,t_inf=%lld,t_post=%lld,t_total=%lld,num_dets=%u\n",
           (long long)(tp1 - tp0), (long long)(tp2 - tp1),
           (long long)(tp3 - tp2), (long long)(tp3 - tp0),
           (unsigned)init_dets.size());
    for (size_t i = 0; i < init_dets.size(); i++) {
        const auto &d = init_dets[i];
        const char *cname = (d.cls_id >= 0 && d.cls_id < kNumClasses) ? kClassNames[d.cls_id] : "unknown";
        printf("DET,%u,%d,%s,%.4f,%.2f,%.2f,%.2f,%.2f\n",
               (unsigned)i, d.cls_id, cname, d.score, d.x1, d.y1, d.x2, d.y2);
    }
    printf("=== PREDICTION_CHECK_END ===\n\n");
    fflush(stdout);

    // 6. Bateria Oficial em 3 Blocos Independentes (Protocolo v1.3 — E3)
    printf("=== BENCHMARK_START ===\n");
    printf("DEVICE=ESP32-S3\n");
    printf("CPU_FREQ_MHZ=240\n");
    printf("MODEL=YOLOX-Nano-INT8\n");
    printf("INPUT_RESOLUTION=%dx%dx3\n", target_w, target_h);
    printf("BLOCKS=%d\n", kNumBlocks);
    printf("RUNS_PER_BLOCK=%d\n", kRunsPerBlock);
    printf("WARMUP_PER_BLOCK=%d\n", kWarmupPerBlock);
    printf("CSV_HEADER:iteration,block,run_in_block,t_pre_us,t_inf_us,t_post_us,t_total_us,t_pre_ms,t_inf_ms,t_post_ms,t_total_ms,latency_us,latency_ms,num_detections,arena_kb,free_sram_kb,free_psram_kb\n");
    fflush(stdout);

    std::vector<double> all_t_pre_ms, all_t_inf_ms, all_t_post_ms, all_t_total_ms;
    int global_iter = 0;

    for (int b = 1; b <= kNumBlocks; b++) {
        printf("# --- Bloco %d/%d: Executando %d inferencias de warmup (descartadas) ---\n",
               b, kNumBlocks, kWarmupPerBlock);
        fflush(stdout);
        for (int w = 1; w <= kWarmupPerBlock; w++) {
            preprocess_tensor(_binary_test_image_bin_start, std::min(img_bytes, input_tensor->bytes),
                              input_tensor->data.int8, in_scale, in_zp);
            interpreter.Invoke();
            (void)postprocess_yolox(output_tensor->data.int8, target_h, target_w, out_scale, out_zp, kConfThr, kNmsThr);
            vTaskDelay(pdMS_TO_TICKS(50));
        }

        printf("# --- Bloco %d/%d: Coletando %d execucoes validas ---\n", b, kNumBlocks, kRunsPerBlock);
        fflush(stdout);

        for (int r = 1; r <= kRunsPerBlock; r++) {
            global_iter++;
            gpio_set_level(BENCHMARK_GPIO, 1);

            int64_t t0 = esp_timer_get_time();
            preprocess_tensor(_binary_test_image_bin_start, std::min(img_bytes, input_tensor->bytes),
                              input_tensor->data.int8, in_scale, in_zp);
            int64_t t1 = esp_timer_get_time();

            TfLiteStatus status = interpreter.Invoke();
            int64_t t2 = esp_timer_get_time();

            std::vector<Detection> dets = postprocess_yolox(
                output_tensor->data.int8, target_h, target_w, out_scale, out_zp, kConfThr, kNmsThr
            );
            int64_t t3 = esp_timer_get_time();

            gpio_set_level(BENCHMARK_GPIO, 0);

            if (status != kTfLiteOk) {
                ESP_LOGE(TAG, "Falha na inferencia %d!", global_iter);
                break;
            }

            int64_t t_pre_us = t1 - t0;
            int64_t t_inf_us = t2 - t1;
            int64_t t_post_us = t3 - t2;
            int64_t t_total_us = t3 - t0;

            double t_pre_ms = t_pre_us / 1000.0;
            double t_inf_ms = t_inf_us / 1000.0;
            double t_post_ms = t_post_us / 1000.0;
            double t_total_ms = t_total_us / 1000.0;

            all_t_pre_ms.push_back(t_pre_ms);
            all_t_inf_ms.push_back(t_inf_ms);
            all_t_post_ms.push_back(t_post_ms);
            all_t_total_ms.push_back(t_total_ms);

            size_t sram_kb = heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024;
            size_t psram_kb = heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024;
            size_t arena_kb = arena_used / 1024;

            printf("CSV_ROW,%d,%d,%d,%lld,%lld,%lld,%lld,%.3f,%.3f,%.3f,%.3f,%lld,%.3f,%u,%u,%u,%u\n",
                   global_iter, b, r,
                   (long long)t_pre_us, (long long)t_inf_us, (long long)t_post_us, (long long)t_total_us,
                   t_pre_ms, t_inf_ms, t_post_ms, t_total_ms,
                   (long long)t_inf_us, t_inf_ms,
                   (unsigned)dets.size(), (unsigned)arena_kb, (unsigned)sram_kb, (unsigned)psram_kb);
            // Também imprime linha RUN compatível com coletores legados
            printf("RUN,%d,%lld\n", global_iter, (long long)t_inf_us);
            fflush(stdout);

            vTaskDelay(pdMS_TO_TICKS(kInterRunDelayMs));
        }

        if (b < kNumBlocks) {
            printf("# [Pausa termica entre blocos] Aguardando %d ms...\n", kBlockCooldownMs);
            fflush(stdout);
            vTaskDelay(pdMS_TO_TICKS(kBlockCooldownMs));
        }
    }

    printf("=== BENCHMARK_END ===\n\n");
    fflush(stdout);

    if (!all_t_inf_ms.empty()) {
        auto calc_mean = [](const std::vector<double> &v) {
            double s = 0.0;
            for (double x : v) s += x;
            return s / v.size();
        };
        auto calc_std = [&](const std::vector<double> &v, double m) {
            double s = 0.0;
            for (double x : v) s += (x - m) * (x - m);
            return std::sqrt(s / (v.size() > 1 ? v.size() - 1 : 1));
        };
        auto calc_median = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            size_t n = v.size();
            return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
        };

        double m_pre = calc_mean(all_t_pre_ms);
        double m_inf = calc_mean(all_t_inf_ms);
        double m_post = calc_mean(all_t_post_ms);
        double m_tot = calc_mean(all_t_total_ms);

        printf("================ RESUMO ESTATISTICO (PROTOCOLO v1.3) ================\n");
        printf("Dispositivo        : ESP32-S3 @ 240 MHz (esp-tflite-micro INT8)\n");
        printf("Amostras Validas   : %u (%d blocos x %d execucoes)\n",
               (unsigned)all_t_inf_ms.size(), kNumBlocks, kRunsPerBlock);
        printf("t_pre   Mediana/Med: %.2f ms / %.2f +- %.2f ms\n",
               calc_median(all_t_pre_ms), m_pre, calc_std(all_t_pre_ms, m_pre));
        printf("t_inf   Mediana/Med: %.2f ms / %.2f +- %.2f ms\n",
               calc_median(all_t_inf_ms), m_inf, calc_std(all_t_inf_ms, m_inf));
        printf("t_post  Mediana/Med: %.2f ms / %.2f +- %.2f ms\n",
               calc_median(all_t_post_ms), m_post, calc_std(all_t_post_ms, m_post));
        printf("t_total Mediana/Med: %.2f ms / %.2f +- %.2f ms\n",
               calc_median(all_t_total_ms), m_tot, calc_std(all_t_total_ms, m_tot));
        printf("Throughput (t_inf) : %.3f FPS | (t_total): %.3f FPS\n",
               1000.0 / m_inf, 1000.0 / m_tot);
        printf("Arena TFLM Usada   : %.2f KiB | Previsoes na imagem: %u deteccao(oes)\n",
               arena_used / 1024.0, (unsigned)init_dets.size());
        printf("=====================================================================\n\n");
        fflush(stdout);
    }

    while (true) {
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}
