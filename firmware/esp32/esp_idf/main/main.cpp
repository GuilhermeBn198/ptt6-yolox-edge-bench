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
#include "driver/uart.h"
#include "driver/usb_serial_jtag.h"

#include "model_data.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

static const char *TAG = "YOLOX_BENCHMARK";

#define BENCHMARK_GPIO GPIO_NUM_4
#define BENCH_UART_PORT UART_NUM_0

static constexpr int kNumBlocks = 3;
static constexpr int kRunsPerBlock = 10;
static constexpr int kWarmupPerBlock = 5;
static constexpr float kConfThr = 0.25f;
static constexpr float kNmsThr = 0.45f;
static constexpr int kMaxDetections = 100;
static constexpr int kNumClasses = 11;
static constexpr size_t kInputBytes = 416 * 416 * 3;
static constexpr size_t kTensorArenaSize = 4 * 1024 * 1024;

static const char *kClassNames[kNumClasses] = {
    "helmet", "gloves", "vest", "boots", "goggles",
    "none", "Person", "no_helmet", "no_goggle", "no_gloves", "no_boots"
};

static uint8_t *g_tensor_arena = nullptr;
static uint8_t *g_img_buf = nullptr;

// Imagem BGR 416x416x3 padrão embutida na Flash (01_image1.bin)
extern const uint8_t _binary_test_image_bin_start[] asm("_binary_test_image_bin_start");
extern const uint8_t _binary_test_image_bin_end[]   asm("_binary_test_image_bin_end");

struct Detection {
    float x1, y1, x2, y2;
    float score;
    int cls_id;
};

struct TimedRun {
    int64_t t_pre_us;
    int64_t t_inf_us;
    int64_t t_post_us;
    int64_t t_total_us;
    int n_det;
};

// CRC32 IEEE 802.3 idêntico a zlib.crc32 em Python
static uint32_t calc_crc32(const uint8_t *data, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int k = 0; k < 8; k++) {
            crc = (crc >> 1) ^ (0xEDB88320u & (-(int32_t)(crc & 1)));
        }
    }
    return ~crc;
}

// SHA-256 compacto (FIPS 180-4) para auditar hash do .tflite no boot sem dependência externa
static void calc_sha256_hex(const uint8_t *data, size_t len, char out_hex[65]) {
    static const uint32_t K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };
    auto rotr = [](uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); };
    uint32_t H[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    uint64_t bit_len = static_cast<uint64_t>(len) * 8;
    size_t rem = len % 64;
    size_t pad_len = (rem < 56) ? (56 - rem) : (120 - rem);
    size_t total_blocks = (len + pad_len + 8) / 64;

    for (size_t b = 0; b < total_blocks; b++) {
        uint8_t blk[64];
        for (size_t i = 0; i < 64; i++) {
            size_t idx = b * 64 + i;
            if (idx < len) blk[i] = data[idx];
            else if (idx == len) blk[i] = 0x80;
            else if (idx >= total_blocks * 64 - 8) {
                int shift = (int)((total_blocks * 64 - 1 - idx) * 8);
                blk[i] = static_cast<uint8_t>((bit_len >> shift) & 0xFF);
            } else blk[i] = 0;
        }
        uint32_t W[64];
        for (int i = 0; i < 16; i++) {
            W[i] = (uint32_t(blk[4*i]) << 24) | (uint32_t(blk[4*i+1]) << 16) |
                   (uint32_t(blk[4*i+2]) << 8) | uint32_t(blk[4*i+3]);
        }
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = rotr(W[i-15], 7) ^ rotr(W[i-15], 18) ^ (W[i-15] >> 3);
            uint32_t s1 = rotr(W[i-2], 17) ^ rotr(W[i-2], 19) ^ (W[i-2] >> 10);
            W[i] = W[i-16] + s0 + W[i-7] + s1;
        }
        uint32_t a = H[0], b_v = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t t1 = h + S1 + ch + K[i] + W[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b_v) ^ (a & c) ^ (b_v & c);
            uint32_t t2 = S0 + maj;
            h = g; g = f; f = e; e = d + t1; d = c; c = b_v; b_v = a; a = t1 + t2;
        }
        H[0] += a; H[1] += b_v; H[2] += c; H[3] += d; H[4] += e; H[5] += f; H[6] += g; H[7] += h;
    }
    for (int i = 0; i < 8; i++) {
        sprintf(out_hex + i * 8, "%08lx", (unsigned long)H[i]);
    }
    out_hex[64] = '\0';
}

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

static void init_uart(void) {
    if (!uart_is_driver_installed(BENCH_UART_PORT)) {
        uart_driver_install(BENCH_UART_PORT, 32768, 0, 0, NULL, 0);
    }
    if (!usb_serial_jtag_is_driver_installed()) {
        usb_serial_jtag_driver_config_t usb_cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
        usb_cfg.rx_buffer_size = 32768;
        usb_cfg.tx_buffer_size = 4096;
        usb_serial_jtag_driver_install(&usb_cfg);
    }
}

static int g_active_serial_backend = 0; // 0 = auto, 1 = usb_serial_jtag, 2 = uart0

static bool uart_read_exact(uint8_t *dst, size_t n, uint32_t timeout_ms) {
    size_t got = 0;
    int64_t deadline = esp_timer_get_time() + static_cast<int64_t>(timeout_ms) * 1000LL;
    while (got < n) {
        int64_t rem_us = deadline - esp_timer_get_time();
        if (rem_us <= 0) return false;

        if (g_active_serial_backend == 1) {
            TickType_t ticks = pdMS_TO_TICKS(std::min<int64_t>(20, (rem_us / 1000) + 1));
            int r = usb_serial_jtag_read_bytes(dst + got, n - got, ticks);
            if (r > 0) got += static_cast<size_t>(r);
        } else if (g_active_serial_backend == 2) {
            TickType_t ticks = pdMS_TO_TICKS(std::min<int64_t>(20, (rem_us / 1000) + 1));
            int r = uart_read_bytes(BENCH_UART_PORT, dst + got, n - got, ticks);
            if (r > 0) got += static_cast<size_t>(r);
        } else {
            int r_usb = usb_serial_jtag_read_bytes(dst + got, n - got, pdMS_TO_TICKS(5));
            if (r_usb > 0) {
                g_active_serial_backend = 1;
                got += static_cast<size_t>(r_usb);
                continue;
            }
            int r_uart = uart_read_bytes(BENCH_UART_PORT, dst + got, n - got, pdMS_TO_TICKS(5));
            if (r_uart > 0) {
                g_active_serial_backend = 2;
                got += static_cast<size_t>(r_uart);
                continue;
            }
        }
    }
    return true;
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

// Fase 1 (t_pre): Cópia e quantização do tensor 416x416x3 uint8 BGR -> int8
static void preprocess_tensor(const uint8_t *src_bgr, size_t n_bytes, int8_t *dst_tensor, float in_scale, int32_t in_zp) {
    if (std::fabs(in_scale - 1.0f) < 1e-5f && in_zp == -128) {
        for (size_t i = 0; i < n_bytes; i++) {
            dst_tensor[i] = static_cast<int8_t>(static_cast<int>(src_bgr[i]) - 128);
        }
    } else {
        float inv_scale = 1.0f / in_scale;
        for (size_t i = 0; i < n_bytes; i++) {
            int32_t q = static_cast<int32_t>(std::round(static_cast<float>(src_bgr[i]) * inv_scale)) + in_zp;
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            dst_tensor[i] = static_cast<int8_t>(q);
        }
    }
}

// Fase 3 (t_post): Decodificação YOLOX unificada (argmax por âncora, score = obj * cls + NMS por classe)
// Idêntica a benchmark_yolox.py, bench_collect.py, eval_map_tflite.py e tflm_host_eval.cpp
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

                int best_cls = 0;
                float best_prob = (static_cast<float>(row[5]) - static_cast<float>(out_zp)) * out_scale;
                for (int c = 1; c < kNumClasses; c++) {
                    float c_prob = (static_cast<float>(row[5 + c]) - static_cast<float>(out_zp)) * out_scale;
                    if (c_prob > best_prob) {
                        best_prob = c_prob;
                        best_cls = c;
                    }
                }
                float score = obj_conf * best_prob;
                if (score < conf_thr) {
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

                candidates_by_class[best_cls].push_back({
                    cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f, score, best_cls
                });
            }
        }
    }

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
            for (size_t j = i + 1; j < cands.size(); j++) {
                if (!suppressed[j] && box_iou(cands[i], cands[j]) > nms_thr) {
                    suppressed[j] = true;
                }
            }
        }
    }
    std::sort(final_dets.begin(), final_dets.end(), [](const Detection &a, const Detection &b) {
        return a.score > b.score;
    });
    if (static_cast<int>(final_dets.size()) > kMaxDetections) {
        final_dets.resize(kMaxDetections);
    }
    return final_dets;
}

extern "C" void app_main(void) {
    vTaskDelay(pdMS_TO_TICKS(1000));
    init_uart();
    init_gpio();

    printf("\n=======================================================================\n");
    printf("  Benchmark YOLOX-Nano INT8 — ESP32-S3 (Protocolo Experimental v1.5)   \n");
    printf("=======================================================================\n");

    size_t free_internal_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t free_psram_before = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("SRAM Interna Livre (boot): %u bytes (%.2f KiB)\n",
           (unsigned)free_internal_before, free_internal_before / 1024.0);
    printf("PSRAM Octal Livre  (boot): %u bytes (%.2f KiB)\n",
           (unsigned)free_psram_before, free_psram_before / 1024.0);

    g_tensor_arena = static_cast<uint8_t *>(
        heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
    );
    g_img_buf = static_cast<uint8_t *>(
        heap_caps_malloc(kInputBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
    );
    if (!g_tensor_arena || !g_img_buf) {
        ESP_LOGE(TAG, "ERRO: Falha ao alocar tensor_arena ou g_img_buf na PSRAM!");
        return;
    }

    size_t flash_img_bytes = _binary_test_image_bin_end - _binary_test_image_bin_start;
    std::memcpy(g_img_buf, _binary_test_image_bin_start, std::min(flash_img_bytes, kInputBytes));

    const tflite::Model *model = tflite::GetModel(g_yolox_model_data);
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

    char model_sha[65];
    calc_sha256_hex(g_yolox_model_data, g_yolox_model_data_len, model_sha);

    printf("Runtime       : esp-tflite-micro (ESP-NN SIMD Xtensa LX7)\n");
    printf("Modelo        : yolox_nano_ppe_int8.tflite (%u bytes / %.3f MiB)\n",
           (unsigned)g_yolox_model_data_len, g_yolox_model_data_len / (1024.0 * 1024.0));
    printf("SHA-256       : %s\n", model_sha);
    printf("Resolucao     : %dx%d (BGR) | in_scale=%.6f, in_zp=%ld | out_scale=%.6f, out_zp=%ld\n",
           target_w, target_h, in_scale, (long)in_zp, out_scale, (long)out_zp);
    printf("Arena TFLM    : %u bytes (%.2f KiB / %.3f MiB)\n",
           (unsigned)arena_used, arena_used / 1024.0, arena_used / (1024.0 * 1024.0));
    printf("MEM_FOOTPRINT,%u,%u,%u,%u\n",
           (unsigned)g_yolox_model_data_len, (unsigned)arena_used,
           (unsigned)free_internal_after, (unsigned)free_psram_after);
    printf("BOOT,sha256=%s,r=%d,classes=%d,arena=%u,sram_free=%u,psram_free=%u,out_scale=%.6f,out_zp=%ld,core=ESP-IDF-v6.1\n",
           model_sha, target_w, kNumClasses, (unsigned)arena_used,
           (unsigned)free_internal_after, (unsigned)free_psram_after, out_scale, (long)out_zp);
    fflush(stdout);

    auto run_timed_window = [&](std::vector<Detection> &out_dets) -> TimedRun {
        gpio_set_level(BENCHMARK_GPIO, 1);
        int64_t t0 = esp_timer_get_time();
        preprocess_tensor(g_img_buf, kInputBytes, input_tensor->data.int8, in_scale, in_zp);
        int64_t t1 = esp_timer_get_time();
        interpreter.Invoke();
        int64_t t2 = esp_timer_get_time();
        out_dets = postprocess_yolox(output_tensor->data.int8, target_h, target_w, out_scale, out_zp, kConfThr, kNmsThr);
        int64_t t3 = esp_timer_get_time();
        gpio_set_level(BENCHMARK_GPIO, 0);
        return {t1 - t0, t2 - t1, t3 - t2, t3 - t0, static_cast<int>(out_dets.size())};
    };

    // Verifica se o coletor do PC enviou comando serial nos primeiros 4 segundos após BOOT
    uint8_t cmd_magic[3] = {0, 0, 0};
    bool serial_mode = uart_read_exact(cmd_magic, 3, 4000);

    if (serial_mode) {
        // Modo de Ingestão Serial Direta (Seção 4.8 de main.tex / D33)
        while (true) {
            if (std::memcmp(cmd_magic, "IMG", 3) == 0) {
                uint8_t hdr[6];
                if (!uart_read_exact(hdr, 6, 5000)) break;
                uint16_t id = static_cast<uint16_t>(hdr[0]) | (static_cast<uint16_t>(hdr[1]) << 8);
                uint32_t len = static_cast<uint32_t>(hdr[2]) |
                               (static_cast<uint32_t>(hdr[3]) << 8) |
                               (static_cast<uint32_t>(hdr[4]) << 16) |
                               (static_cast<uint32_t>(hdr[5]) << 24);
                if (len != kInputBytes) {
                    printf("NAK,%u,len\n", (unsigned)id);
                    fflush(stdout);
                } else if (!uart_read_exact(g_img_buf, len, 20000)) {
                    printf("NAK,%u,timeout\n", (unsigned)id);
                    fflush(stdout);
                } else {
                    uint8_t crc_buf[4];
                    if (!uart_read_exact(crc_buf, 4, 5000)) {
                        printf("NAK,%u,crc_timeout\n", (unsigned)id);
                        fflush(stdout);
                    } else {
                        uint32_t expected_crc = static_cast<uint32_t>(crc_buf[0]) |
                                                (static_cast<uint32_t>(crc_buf[1]) << 8) |
                                                (static_cast<uint32_t>(crc_buf[2]) << 16) |
                                                (static_cast<uint32_t>(crc_buf[3]) << 24);
                        uint32_t actual_crc = calc_crc32(g_img_buf, len);
                        if (actual_crc != expected_crc) {
                            printf("NAK,%u,crc\n", (unsigned)id);
                            fflush(stdout);
                        } else {
                            printf("ACK,%u\n", (unsigned)id);
                            fflush(stdout);
                            uart_wait_tx_done(BENCH_UART_PORT, pdMS_TO_TICKS(100));

                            std::vector<Detection> dets;
                            TimedRun tr = run_timed_window(dets);

                            printf("RES,%u,%lld,%lld,%lld,%lld,%d\n",
                                   (unsigned)id,
                                   (long long)tr.t_pre_us, (long long)tr.t_inf_us,
                                   (long long)tr.t_post_us, (long long)tr.t_total_us, tr.n_det);
                            for (const auto &d : dets) {
                                printf("DET,%u,%.2f,%.2f,%.2f,%.2f,%.4f,%d\n",
                                       (unsigned)id, d.x1, d.y1, d.x2, d.y2, d.score, d.cls_id);
                            }
                            printf("END,%u\n", (unsigned)id);
                            fflush(stdout);
                        }
                    }
                }
            } else if (std::memcmp(cmd_magic, "RUN", 3) == 0) {
                // Reexecuta janela cronometrada sobre a imagem atual em g_img_buf (sem reenviar 519 KB)
                uint8_t hdr[2];
                if (!uart_read_exact(hdr, 2, 5000)) break;
                uint16_t id = static_cast<uint16_t>(hdr[0]) | (static_cast<uint16_t>(hdr[1]) << 8);
                printf("ACK,%u\n", (unsigned)id);
                fflush(stdout);
                uart_wait_tx_done(BENCH_UART_PORT, pdMS_TO_TICKS(100));

                std::vector<Detection> dets;
                TimedRun tr = run_timed_window(dets);

                printf("RES,%u,%lld,%lld,%lld,%lld,%d\n",
                       (unsigned)id,
                       (long long)tr.t_pre_us, (long long)tr.t_inf_us,
                       (long long)tr.t_post_us, (long long)tr.t_total_us, tr.n_det);
                for (const auto &d : dets) {
                    printf("DET,%u,%.2f,%.2f,%.2f,%.2f,%.4f,%d\n",
                           (unsigned)id, d.x1, d.y1, d.x2, d.y2, d.score, d.cls_id);
                }
                printf("END,%u\n", (unsigned)id);
                fflush(stdout);
            } else if (std::memcmp(cmd_magic, "E4S", 3) == 0) {
                // Experimento E4: Laço fechado contínuo por duration_s segundos sem I/O serial durante a janela
                uint8_t hdr[2];
                if (!uart_read_exact(hdr, 2, 5000)) break;
                uint16_t dur_s = static_cast<uint16_t>(hdr[0]) | (static_cast<uint16_t>(hdr[1]) << 8);
                printf("ACK,E4,%u\n", (unsigned)dur_s);
                fflush(stdout);
                uart_wait_tx_done(BENCH_UART_PORT, pdMS_TO_TICKS(100));

                std::vector<TimedRun> e4_runs;
                std::vector<int64_t> e4_elapsed_us;
                e4_runs.reserve(32);
                e4_elapsed_us.reserve(32);

                int64_t t_loop_start = esp_timer_get_time();
                int64_t target_us = static_cast<int64_t>(dur_s) * 1000000LL;
                while ((esp_timer_get_time() - t_loop_start) < target_us) {
                    std::vector<Detection> dets;
                    TimedRun tr = run_timed_window(dets);
                    e4_runs.push_back(tr);
                    e4_elapsed_us.push_back(esp_timer_get_time() - t_loop_start);
                }
                int64_t total_elapsed_us = esp_timer_get_time() - t_loop_start;

                for (size_t i = 0; i < e4_runs.size(); i++) {
                    const auto &tr = e4_runs[i];
                    printf("E4_ROW,%u,%.3f,%lld,%lld,%lld,%lld,%d\n",
                           (unsigned)(i + 1), e4_elapsed_us[i] / 1000000.0,
                           (long long)tr.t_pre_us, (long long)tr.t_inf_us,
                           (long long)tr.t_post_us, (long long)tr.t_total_us, tr.n_det);
                }
                printf("E4_DONE,%u,%lld\n", (unsigned)e4_runs.size(), (long long)total_elapsed_us);
                fflush(stdout);
            } else {
                uart_flush_input(BENCH_UART_PORT);
            }

            if (!uart_read_exact(cmd_magic, 3, 120000)) {
                continue;
            }
        }
    }

    // Fallback autônomo caso nenhum coletor envie comandos nos primeiros 4 segundos
    printf("\n=== PREDICTION_CHECK_START ===\n");
    std::vector<Detection> init_dets;
    TimedRun tr0 = run_timed_window(init_dets);
    printf("FIRST_RUN_TIMING_US,t_pre=%lld,t_inf=%lld,t_post=%lld,t_total=%lld,num_dets=%u\n",
           (long long)tr0.t_pre_us, (long long)tr0.t_inf_us,
           (long long)tr0.t_post_us, (long long)tr0.t_total_us,
           (unsigned)init_dets.size());
    for (size_t i = 0; i < init_dets.size(); i++) {
        const auto &d = init_dets[i];
        const char *cname = (d.cls_id >= 0 && d.cls_id < kNumClasses) ? kClassNames[d.cls_id] : "unknown";
        printf("DET,%u,%d,%s,%.4f,%.2f,%.2f,%.2f,%.2f\n",
               (unsigned)i, d.cls_id, cname, d.score, d.x1, d.y1, d.x2, d.y2);
    }
    printf("=== PREDICTION_CHECK_END ===\n\n");
    fflush(stdout);

    printf("=== BENCHMARK_START ===\n");
    printf("CSV_HEADER:iteration,block,run_in_block,t_pre_us,t_inf_us,t_post_us,t_total_us,t_pre_ms,t_inf_ms,t_post_ms,t_total_ms,latency_us,latency_ms,num_detections,arena_kb,free_sram_kb,free_psram_kb\n");
    fflush(stdout);

    int global_iter = 0;
    for (int b = 1; b <= kNumBlocks; b++) {
        for (int w = 1; w <= kWarmupPerBlock; w++) {
            std::vector<Detection> wdets;
            (void)run_timed_window(wdets);
        }
        for (int r = 1; r <= kRunsPerBlock; r++) {
            global_iter++;
            std::vector<Detection> dets;
            TimedRun tr = run_timed_window(dets);
            size_t sram_kb = heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024;
            size_t psram_kb = heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024;
            size_t arena_kb = arena_used / 1024;
            printf("CSV_ROW,%d,%d,%d,%lld,%lld,%lld,%lld,%.3f,%.3f,%.3f,%.3f,%lld,%.3f,%u,%u,%u,%u\n",
                   global_iter, b, r,
                   (long long)tr.t_pre_us, (long long)tr.t_inf_us, (long long)tr.t_post_us, (long long)tr.t_total_us,
                   tr.t_pre_us / 1000.0, tr.t_inf_us / 1000.0, tr.t_post_us / 1000.0, tr.t_total_us / 1000.0,
                   (long long)tr.t_inf_us, tr.t_inf_us / 1000.0,
                   (unsigned)dets.size(), (unsigned)arena_kb, (unsigned)sram_kb, (unsigned)psram_kb);
            fflush(stdout);
        }
    }
    printf("=== BENCHMARK_END ===\n\n");
    fflush(stdout);

    while (true) {
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}
