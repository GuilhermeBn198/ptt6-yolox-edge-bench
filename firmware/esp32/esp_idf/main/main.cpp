#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

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
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "YOLOX_BENCHMARK";

// Sincronização via GPIO conforme metodologia PTT6 (Docs / Roteiro)
#define BENCHMARK_GPIO GPIO_NUM_4

// Configurações do Protocolo Experimental
static constexpr int kNumWarmup = 5;
static constexpr int kNumRuns = 30;
static constexpr int kCoolingDelayMs = 1000;

// Arena de ativações na PSRAM (4 MB alocados na Octal PSRAM externa)
static constexpr size_t kTensorArenaSize = 4 * 1024 * 1024;
static uint8_t *g_tensor_arena = nullptr;

// Imagem de teste binária embutida diretamente na partição rodata via linker
extern const uint8_t _binary_test_image_bin_start[] asm("_binary_test_image_bin_start");
extern const uint8_t _binary_test_image_bin_end[]   asm("_binary_test_image_bin_end");

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

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "=================================================");
    ESP_LOGI(TAG, "  Iniciando Benchmark YOLOX-Nano INT8 no ESP32-S3");
    ESP_LOGI(TAG, "=================================================");

    // 1. Diagnóstico de Memória
    size_t free_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    ESP_LOGI(TAG, "SRAM Interna Livre: %u bytes (%u KB)", free_internal, free_internal / 1024);
    ESP_LOGI(TAG, "PSRAM Octal Livre : %u bytes (%u KB)", free_psram, free_psram / 1024);

    if (free_psram < kTensorArenaSize) {
        ESP_LOGE(TAG, "ERRO: PSRAM insuficiente para arena de %u bytes!", kTensorArenaSize);
        return;
    }

    // 2. Alocação da Tensor Arena em PSRAM
    g_tensor_arena = static_cast<uint8_t *>(heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (g_tensor_arena == nullptr) {
        ESP_LOGE(TAG, "ERRO: Falha ao alocar tensor_arena de %u bytes na PSRAM!", kTensorArenaSize);
        return;
    }
    ESP_LOGI(TAG, "Tensor Arena de %u KB alocada com sucesso em PSRAM.", kTensorArenaSize / 1024);

    // 3. Inicializar GPIO de telemetria
    init_gpio();
    ESP_LOGI(TAG, "GPIO %d configurado para cronometragem por toggle de hardware.", BENCHMARK_GPIO);

    // 4. Carregar o Modelo TFLite INT8
    const tflite::Model *model = tflite::GetModel(g_yolox_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Versao do schema (%lu) incompativel com TFLITE_SCHEMA_VERSION (%d)",
                 model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    ESP_LOGI(TAG, "Modelo carregado: %u bytes.", g_yolox_model_data_len);

    // 5. Configurar o Interpretador TFLM com operadores específicos do YOLOX
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

    TfLiteStatus allocate_status = interpreter.AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "ERRO: Falha em AllocateTensors()!");
        return;
    }
    ESP_LOGI(TAG, "Tensores alocados com sucesso.");
    ESP_LOGI(TAG, "Memória da arena realmente utilizada pelo TFLM: %u bytes (%u KB)",
             interpreter.arena_used_bytes(), interpreter.arena_used_bytes() / 1024);

    // 6. Validar Tensores de Entrada e Saída
    TfLiteTensor *input_tensor = interpreter.input(0);
    TfLiteTensor *output_tensor = interpreter.output(0);

    if (input_tensor == nullptr || output_tensor == nullptr) {
        ESP_LOGE(TAG, "ERRO ao obter tensores de entrada/saida!");
        return;
    }

    ESP_LOGI(TAG, "Tensor Entrada: Tipo=%d, Bytes=%u, Dimensoes=[%ld, %ld, %ld, %ld]",
             input_tensor->type, input_tensor->bytes,
             input_tensor->dims->data[0], input_tensor->dims->data[1],
             input_tensor->dims->data[2], input_tensor->dims->data[3]);
    ESP_LOGI(TAG, "Tensor Saida  : Tipo=%d, Bytes=%u, Dimensoes=[%ld, %ld, %ld]",
             output_tensor->type, output_tensor->bytes,
             output_tensor->dims->data[0], output_tensor->dims->data[1],
             output_tensor->dims->data[2]);

    // 7. Carregar dados da imagem de teste (416x416x3 int8)
    size_t img_bytes = _binary_test_image_bin_end - _binary_test_image_bin_start;
    ESP_LOGI(TAG, "Tamanho da imagem de teste embutida: %u bytes", img_bytes);
    if (img_bytes != input_tensor->bytes) {
        ESP_LOGW(TAG, "Aviso: Tamanho da imagem (%u) difere do tensor de entrada (%u)",
                 img_bytes, input_tensor->bytes);
    }
    memcpy(input_tensor->data.int8, _binary_test_image_bin_start,
           std::min(img_bytes, (size_t)input_tensor->bytes));

    // 8. Fase de Warmup (5 inferências descartadas para estabilização de caches)
    ESP_LOGI(TAG, "Iniciando %d inferencias de warmup (descartadas)...", kNumWarmup);
    for (int i = 0; i < kNumWarmup; i++) {
        gpio_set_level(BENCHMARK_GPIO, 1);
        TfLiteStatus status = interpreter.Invoke();
        gpio_set_level(BENCHMARK_GPIO, 0);

        if (status != kTfLiteOk) {
            ESP_LOGE(TAG, "Erro no warmup %d!", i + 1);
            return;
        }
        ESP_LOGI(TAG, "Warmup %d/%d concluido.", i + 1, kNumWarmup);
        vTaskDelay(pdMS_TO_TICKS(kCoolingDelayMs));
    }

    ESP_LOGI(TAG, "Warmup finalizado. Aguardando 3 segundos para resfriamento...");
    vTaskDelay(pdMS_TO_TICKS(3000));

    // 9. Bateria de 30 Inferências Oficiais
    printf("\n=== BENCHMARK_START ===\n");
    printf("DEVICE=ESP32-S3\n");
    printf("CPU_FREQ_MHZ=240\n");
    printf("MODEL=YOLOX-Nano-INT8\n");
    printf("INPUT_RESOLUTION=416x416x3\n");
    printf("RUNS=%d\n", kNumRuns);
    printf("CSV_HEADER:RUN,LATENCY_US\n");

    std::vector<int64_t> latencies_us;
    latencies_us.reserve(kNumRuns);

    for (int run = 1; run <= kNumRuns; run++) {
        // Toggle GPIO e Medição de Tempo de Alta Resolução
        gpio_set_level(BENCHMARK_GPIO, 1);
        int64_t t_start = esp_timer_get_time();

        TfLiteStatus invoke_status = interpreter.Invoke();

        int64_t t_end = esp_timer_get_time();
        gpio_set_level(BENCHMARK_GPIO, 0);

        if (invoke_status != kTfLiteOk) {
            ESP_LOGE(TAG, "Falha na inferencia %d!", run);
            break;
        }

        int64_t latency = t_end - t_start;
        latencies_us.push_back(latency);

        // Saída serial em formato CSV padronizado
        printf("RUN,%d,%lld\n", run, (long long)latency);

        // Pausa térmica entre repetições
        vTaskDelay(pdMS_TO_TICKS(kCoolingDelayMs));
    }

    printf("=== BENCHMARK_END ===\n\n");

    // 10. Cálculo de Estatísticas Descritivas
    if (!latencies_us.empty()) {
        double sum = 0.0;
        int64_t min_lat = latencies_us[0];
        int64_t max_lat = latencies_us[0];

        for (int64_t lat : latencies_us) {
            sum += lat;
            if (lat < min_lat) min_lat = lat;
            if (lat > max_lat) max_lat = lat;
        }

        double mean_us = sum / latencies_us.size();
        double variance_us = 0.0;
        for (int64_t lat : latencies_us) {
            variance_us += (lat - mean_us) * (lat - mean_us);
        }
        variance_us /= latencies_us.size();
        double std_dev_us = std::sqrt(variance_us);

        printf("================ RESUMO ESTATISTICO ================\n");
        printf("Amostras coletadas : %zu\n", latencies_us.size());
        printf("Latencia Media     : %.2f ms (%.0f us)\n", mean_us / 1000.0, mean_us);
        printf("Desvio Padrao (s)  : %.2f ms (%.0f us)\n", std_dev_us / 1000.0, std_dev_us);
        printf("Latencia Minima    : %.2f ms (%.0f us)\n", min_lat / 1000.0, (double)min_lat);
        printf("Latencia Maxima    : %.2f ms (%.0f us)\n", max_lat / 1000.0, (double)max_lat);
        printf("Throughput Estimado: %.2f FPS\n", 1000000.0 / mean_us);
        printf("=====================================================\n\n");
    }

    ESP_LOGI(TAG, "Benchmark concluido com sucesso. Sistema em idle.");

    // Loop de repouso
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}

