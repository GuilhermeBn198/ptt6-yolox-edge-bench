// PTT6 — benchmark YOLOX-Nano int8 no ESP32 (LX6) com ingestão de imagens pela serial (protocolo v1.4, D33)
//
// ESQUELETO — ainda não compilado/testado no hardware. Ajustar:
//   1. model_data.h : `xxd -i yolox_nano_int8.tflite > model_data.h` (renomear para g_model / g_model_len)
//   2. kInputRes / kNumClasses / kArenaSize conforme o artefato e a resolução comum (D31)
//   3. lista de ops do MicroMutableOpResolver conforme o modelo (ver ops com o Netron / tflite analyzer)
//   4. RAW_HEAD = 1 se o export NÃO decodifica a saída do YOLOX (mesma opção --raw-head do SBC)
//
// Bibliotecas (Arduino IDE / PlatformIO): core "esp32" (Espressif) + TensorFlow Lite Micro para ESP32
// (ex.: "TensorFlowLite_ESP32" ou o componente esp-tflite-micro). Registrar as versões no meta.json.
// Protocolo serial: ver scripts/esp_serial_feeder.py e docs/experimentacao.md §5.4.

#include <Arduino.h>
#include <WiFi.h>
#include <esp_timer.h>
#include <mbedtls/sha256.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"  // g_model[], g_model_len

#define RAW_HEAD 0

constexpr int kInputRes = 160;            // r comum (D31) — ajustar
constexpr int kNumClasses = 1;            // bloco A = 1 (vest); bloco B = 11
constexpr size_t kArenaSize = 3 * 1024 * 1024;  // em PSRAM — ajustar após arena_used_bytes()
constexpr float kConfThr = 0.25f, kNmsIou = 0.45f;
constexpr int kMaxDet = 100;
constexpr size_t kImgBytes = (size_t)kInputRes * kInputRes * 3;

static uint8_t* g_img = nullptr;      // imagem recebida (r*r*3, BGR uint8, letterbox)
static uint8_t* g_arena = nullptr;
static tflite::MicroInterpreter* g_interp = nullptr;

struct Det { float x1, y1, x2, y2, score; int cls; };
static Det g_cand[2048];
static Det g_dets[kMaxDet];

// ------------------------------------------------------------------ utilidades
static uint32_t crc32_update(uint32_t crc, const uint8_t* p, size_t n) {  // igual a zlib.crc32
  crc = ~crc;
  while (n--) {
    crc ^= *p++;
    for (int k = 0; k < 8; k++) crc = (crc >> 1) ^ (0xEDB88320u & (-(int32_t)(crc & 1)));
  }
  return ~crc;
}

static bool read_exact(uint8_t* dst, size_t n, uint32_t timeout_ms = 10000) {
  size_t got = 0;
  uint32_t t0 = millis();
  while (got < n) {
    int a = Serial.available();
    if (a > 0) got += Serial.readBytes(dst + got, min((size_t)a, n - got));
    else if (millis() - t0 > timeout_ms) return false;
  }
  return true;
}

static float iou(const Det& a, const Det& b) {
  float xx1 = max(a.x1, b.x1), yy1 = max(a.y1, b.y1), xx2 = min(a.x2, b.x2), yy2 = min(a.y2, b.y2);
  float inter = max(0.f, xx2 - xx1) * max(0.f, yy2 - yy1);
  float ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return inter / (ua + 1e-9f);
}

static inline float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

// ------------------------------------------------------------------ pipeline (mesma lógica do SBC)
static void preprocess(TfLiteTensor* in) {
  const float s = in->params.scale;
  const int zp = in->params.zero_point;
  int8_t* q = in->data.int8;
  for (size_t i = 0; i < kImgBytes; i++) {  // YOLOX: sem normalização (0..255)
    int v = (int)lrintf(g_img[i] / s) + zp;
    q[i] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
  }
}

static int postprocess(const TfLiteTensor* out) {
  const int A = out->dims->data[1], C = out->dims->data[2];  // [1, A, 5 + classes]
  const float s = out->params.scale;
  const int zp = out->params.zero_point;
  auto val = [&](int a, int c) { return (out->data.int8[a * C + c] - zp) * s; };
  int nc = 0;
#if RAW_HEAD
  static const int strides[3] = {8, 16, 32};
  int a = 0;
  for (int si = 0; si < 3; si++) {
    const int st = strides[si], g = kInputRes / st;
    for (int gy = 0; gy < g; gy++)
      for (int gx = 0; gx < g; gx++, a++) {
#else
  for (int a = 0; a < A; a++) {
#endif
        int best = 0;
        float bs = -1e9f;
        for (int c = 0; c < kNumClasses; c++) {
          float v = val(a, 5 + c);
          if (v > bs) { bs = v; best = c; }
        }
#if RAW_HEAD
        float score = sigmoid(val(a, 4)) * sigmoid(bs);
        float cx = (val(a, 0) + gx) * st, cy = (val(a, 1) + gy) * st;
        float w = expf(val(a, 2)) * st, h = expf(val(a, 3)) * st;
#else
        float score = val(a, 4) * bs;
        float cx = val(a, 0), cy = val(a, 1), w = val(a, 2), h = val(a, 3);
#endif
        if (score >= kConfThr && nc < (int)(sizeof(g_cand) / sizeof(g_cand[0])))
          g_cand[nc++] = {cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score, best};
#if RAW_HEAD
      }
  }
#else
  }
#endif
  // NMS por classe (ordenação por score, inserção simples)
  for (int i = 1; i < nc; i++) {
    Det d = g_cand[i];
    int j = i - 1;
    while (j >= 0 && g_cand[j].score < d.score) { g_cand[j + 1] = g_cand[j]; j--; }
    g_cand[j + 1] = d;
  }
  int nd = 0;
  for (int i = 0; i < nc && nd < kMaxDet; i++) {
    bool keep = true;
    for (int k = 0; k < nd; k++)
      if (g_dets[k].cls == g_cand[i].cls && iou(g_dets[k], g_cand[i]) > kNmsIou) { keep = false; break; }
    if (keep) g_dets[nd++] = g_cand[i];
  }
  return nd;
}

// ------------------------------------------------------------------ setup / loop
void setup() {
  setCpuFrequencyMhz(240);  // D21
  WiFi.mode(WIFI_OFF);
  btStop();
  Serial.setRxBufferSize(16384);
  Serial.begin(921600);
  delay(200);

  g_img = (uint8_t*)ps_malloc(kImgBytes);
  g_arena = (uint8_t*)ps_malloc(kArenaSize);
  if (!g_img || !g_arena) { Serial.println("ERR,psram_alloc"); while (true) delay(1000); }

  static tflite::MicroMutableOpResolver<16> resolver;  // ajustar à lista real de ops do modelo
  resolver.AddConv2D(); resolver.AddDepthwiseConv2D(); resolver.AddAdd(); resolver.AddMul();
  resolver.AddConcatenation(); resolver.AddMaxPool2D(); resolver.AddResizeNearestNeighbor();
  resolver.AddReshape(); resolver.AddTranspose(); resolver.AddLogistic(); resolver.AddPad();
  resolver.AddStridedSlice(); resolver.AddQuantize(); resolver.AddDequantize(); resolver.AddSub();
  resolver.AddRelu();

  const tflite::Model* model = tflite::GetModel(g_model);
  static tflite::MicroInterpreter interp(model, resolver, g_arena, kArenaSize);
  g_interp = &interp;
  if (g_interp->AllocateTensors() != kTfLiteOk) { Serial.println("ERR,allocate_tensors"); while (true) delay(1000); }

  uint8_t h[32];
  mbedtls_sha256(g_model, g_model_len, h, 0);
  char hex[65];
  for (int i = 0; i < 32; i++) sprintf(hex + 2 * i, "%02x", h[i]);
  Serial.printf("BOOT,sha256=%s,r=%d,classes=%d,arena=%u,core=%s\n", hex, kInputRes, kNumClasses,
                (unsigned)g_interp->arena_used_bytes(), ESP.getSdkVersion());
}

void loop() {
  // 1) recebe o quadro (fora da janela)
  uint8_t magic[3];
  if (!read_exact(magic, 3, 60000)) return;
  if (memcmp(magic, "IMG", 3) != 0) { while (Serial.available()) Serial.read(); return; }
  uint8_t hdr[6];
  if (!read_exact(hdr, 6)) return;
  uint16_t id = hdr[0] | (hdr[1] << 8);
  uint32_t len = hdr[2] | (hdr[3] << 8) | (hdr[4] << 16) | ((uint32_t)hdr[5] << 24);
  if (len != kImgBytes) { Serial.printf("NAK,%u,len\n", id); return; }
  if (!read_exact(g_img, len)) { Serial.printf("NAK,%u,timeout\n", id); return; }
  uint8_t c[4];
  if (!read_exact(c, 4)) return;
  uint32_t crc = c[0] | (c[1] << 8) | (c[2] << 16) | ((uint32_t)c[3] << 24);
  if (crc32_update(0, g_img, len) != crc) { Serial.printf("NAK,%u,crc\n", id); return; }
  Serial.printf("ACK,%u\n", id);
  Serial.flush();  // garante que nada é transmitido dentro da janela

  // 2) janela cronometrada (D5/D28): nenhum I/O aqui
  int64_t t0 = esp_timer_get_time();
  preprocess(g_interp->input(0));
  int64_t t1 = esp_timer_get_time();
  TfLiteStatus st = g_interp->Invoke();
  int64_t t2 = esp_timer_get_time();
  int nd = (st == kTfLiteOk) ? postprocess(g_interp->output(0)) : -1;
  int64_t t3 = esp_timer_get_time();

  // 3) resultados (fora da janela)
  Serial.printf("RES,%u,%lld,%lld,%lld,%lld,%d\n", id, t1 - t0, t2 - t1, t3 - t2, t3 - t0, nd);
  for (int i = 0; i < nd; i++)
    Serial.printf("DET,%u,%.2f,%.2f,%.2f,%.2f,%.4f,%d\n", id, g_dets[i].x1, g_dets[i].y1, g_dets[i].x2,
                  g_dets[i].y2, g_dets[i].score, g_dets[i].cls);
  Serial.printf("END,%u\n", id);
}
