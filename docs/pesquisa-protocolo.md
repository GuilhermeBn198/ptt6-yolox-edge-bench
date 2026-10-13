# Pesquisa de literatura e fontes primárias — Protocolo experimental de benchmark Edge AI

**Objeto:** inferência de **YOLOX-Nano quantizado em int8** para detecção de EPIs (Equipamentos de Proteção Individual) em **ESP32-S3** (MCU, com PSRAM) vs **Raspberry Pi Zero 2 W** (SBC, Cortex-A53 quad-core @1 GHz).

**Métricas alvo:** latência (ms), throughput (inferências/s), energia por inferência (mJ) e eficiência (inferências/J).

**Protocolo preliminar a validar:** 30 execuções por plataforma, descartando as 5 primeiras (aquecimento); INA219 inline amostrando a 10 Hz; cronometragem por toggle de GPIO; dispersão via desvio-padrão, IQR e coeficiente de variação (CV).

**Regra de evidência:** toda afirmação abaixo termina com a fonte no formato `— autor/fonte, URL`. Onde não há fonte primária encontrada para o caso exato (p.ex. YOLOX-Nano int8 rodando em ESP32-S3), está escrito explicitamente **“não encontrado”**.

> Nota de escopo: este documento é levantamento de fontes, não é um protocolo fechado. As decisões de engenharia pendentes estão consolidadas em “Lacunas / decisões pendentes”.

---

## 1. Metodologia de benchmark de inferência em edge

### 1.1 O que a MLPerf Tiny (padrão de fato para TinyML) prescreve

- A MLPerf Tiny mede **latência, energia e acurácia** de inferência e foi desenhada por mais de 50 organizações; o alvo é “sistemas tiny machine learning de ultra-baixo consumo”. — MLPerf Tiny paper, https://arxiv.org/abs/2106.07597
- A MLPerf Tiny mede **apenas a inferência do modelo**: pre- e post-processing **não entram na janela de medição** na v1.2. — MLPerf Tiny rules, https://github.com/mlcommons/tiny/blob/master/benchmark/MLPerfTiny_Rules.adoc
- Procedimento de **latência** da MLPerf Tiny: executar **5 vezes** o seguinte — baixar um estímulo, carregar o tensor de entrada, **rodar inferência por no mínimo 10 segundos e 10 iterações**, medir as inferências por segundo (IPS) e reportar como score a **mediana dos cinco IPS**. — MLPerf Tiny paper, https://arxiv.org/abs/2106.07597
- Procedimento de **energia**: idêntico ao de latência, mas medindo a energia total usada na janela de tempo e calculando **micro-joules por inferência**; usa-se a mesma mediana de cinco medições. — MLPerf Tiny paper, https://arxiv.org/abs/2106.07597
- A MLPerf Tiny **não exige descarte explícito de warm-up**; ela dilui o transiente medindo por ≥10 s e ≥10 iterações e tomando a mediana de 5 execuções. — MLPerf Tiny paper, https://arxiv.org/abs/2106.07597
- Todos os benchmarks MLPerf Tiny são **single stream**, ou seja, medem a latência de uma inferência individual (além da qualidade do modelo e, opcionalmente, energia). — MLCommons Benchmarks, https://mlcommons.org/benchmarks/inference-tiny/

### 1.2 Limitações reconhecidas da MLPerf Tiny e alternativas metodológicas

- A própria MLPerf Tiny reconhece que “há muitos desafios em medir consumo de energia de modo justo”: dispositivos consomem faixas muito diferentes e **definir o escopo da medição de potência é difícil** quando os caminhos de dados e pré-processamento variam. — MLPerf Tiny paper, https://arxiv.org/abs/2106.07597
- Trabalho de 2025 propõe uma metodologia alternativa que **integra energia e latência** e **distingue três fases** — pré-inferência, inferência e pós-inferência — usando **disparo duplo por GPIO** para separar as fases; crítica explícita à MLPerf Tiny: necessidade de setups separados para latência, acurácia e energia, e monitores que *alimentam* o DUT (reduzindo flexibilidade). — Noronha et al., “Benchmarking Energy and Latency in TinyML”, https://arxiv.org/abs/2505.15622
- Esse trabalho executou **cada modelo MLPerf 1000 vezes** para “resultados estatisticamente relevantes”, e propõe o **Energy-Delay Product (EDP)** como métrica de eficiência. — Noronha et al., https://arxiv.org/abs/2505.15622
- Trabalho anterior sobre TFLite Micro em Cortex-M usa **sinalização por GPIO** para delimitar cada camada e a inferência completa, e um **RocketLogger** que rastreia GPIO “a até 64 kHz”. — Warden et al. / “Measuring what Really Matters”, https://arxiv.org/pdf/2104.10645

### 1.3 Como reportar resultados (estatística)

- **Erro padrão da média** `s_x̄ = s/√N`; pelo Teorema Central do Limite, `N = 10` já costuma ser “grande o suficiente” — mas a fórmula só é válida se as amostras forem **independentes**; autocorrelação (p.ex. deriva térmica/turbo) infla a variância real. — Piotr Kołaczkowski, “Estimating Benchmark Results Uncertainty”, https://pkolaczk.github.io/estimating-benchmark-errors/
- **Travar a frequência da CPU** num valor que se sustenta durante toda a medição, para evitar *frequency scaling* e *throttling* que introduzem variância. — Piotr Kołaczkowski, https://pkolaczk.github.io/estimating-benchmark-errors/
- Métodos “populares” de reportar (melhor de k, pior de k, média, mediana) podem ser **enganosos ou incorretos**; recomenda-se **intervalos de confiança (IC)**, p.ex. `x̄ ± t_{α/2,n−1} · s/√n`, e ANOVA + Tukey HSD para comparar >2 alternativas. Em estudo com 50 papers, métodos prevalentes levaram a conclusão incorreta em até 3% das comparações. — Bulej et al., “Adding Rigorous Statistics to the Java Benchmarker’s Toolbox”, https://dl.acm.org/doi/pdf/10.1145/1297846.1297891
- O “estado estacionário” (steady-state) **pode não existir**: Barrett et al. mostram que mesmo cargas determinísticas pequenas podem não atingir desempenho estável após warm-up, e o método clássico de detecção de steady-state de Georges et al. frequentemente identifica steady-state onde não há. — Antoch et al., “Experimental Evaluation Methodology for the Era of No Steady Performance”, https://dl.acm.org/doi/pdf/10.1145/3839448
- Desenho recomendado para variabilidade: **múltiplas execuções**, cada uma com inicialização/compilação/alocação frescas, e análise estatística correspondente (média intra-run → IC entre runs). — Antoch et al., https://dl.acm.org/doi/pdf/10.1145/3839448
- **NIST TN 1830**: erros aleatórios afetam a precisão/repetibilidade e podem ser reduzidos aumentando N; **erros sistemáticos (viés) não são reduzidos aumentando N** e devem ser detectados, controlados e corrigidos. Recomenda ICs e a GUM (JCGM 100/101). — NIST, “The ghost in the machine”, https://nvlpubs.nist.gov/nistpubs/TechnicalNotes/NIST.TN.1830.pdf
- Boas práticas de benchmark computacional (10 regras) incluem: definir propósito/escopo; incluir todos os métodos relevantes; dados representativos; versões de software/parâmetros; métricas quantitativas; medidas secundárias (runtime, requisitos); publicar de forma acessível; possibilitar extensões futuras; e **seguir boas práticas de reprodutibilidade** (código e dados públicos). — Weber et al., “Essential guidelines for computational method benchmarking”, https://pmc.ncbi.nlm.nih.gov/articles/PMC6584985/

---

## 2. Medição de energia / corrente

### 2.1 INA219 (medidor proposto)

- INA219: monitor de corrente/potência **bidirecional**, 0–26 V de barramento, I²C/SMBus, **ADC de 12 bits**; precisão de corrente ±0,2% (±0,3% INA219A), **±0,5% máx ao longo da temperatura (INA219B)**; consome até **1 mA**. — TI, INA219 datasheet (SBOS448G), https://www.ti.com/lit/ds/symlink/ina219.pdf
- Tempos de conversão do INA219 (12 bits): **84 µs** (9 bits, 1 amostra) · **532 µs** (12 bits, 1 amostra) · até **68,10 ms** com 128 amostras de média. — TI, INA219 datasheet, https://www.ti.com/lit/ds/symlink/ina219.pdf (ver também driver ESP-IDF, https://esp-idf-lib.github.io/ina219/)
- O ADC interno é **delta-sigma com ~500 kHz típicos (±30%)** de *front-end*; ruído/transientes perto dos harmônicos dessa taxa podem causar problemas. — TI, INA219 datasheet, https://www.ti.com/lit/ds/symlink/ina219.pdf
- **Alerta crucial para energia por inferência:** corrente (shunt) e tensão de barramento são convertidas **em instantes diferentes**; com 12 bits + 128 amostras de média pode haver **até ~68 ms entre as duas leituras**, ainda que os cálculos não somem ao tempo de conversão. O *Power register* não é, portanto, uma amostra simultânea de V·I nesse modo. — TI, INA219 datasheet, https://www.ti.com/lit/ds/symlink/ina219.pdf

### 2.2 Alternativas de maior largura de banda

- **Joulescope JS110**: mede corrente e tensão **sincronamente a 2.000.000 amostras/s**, 14 bits, **banda de 250 kHz**, resolução até **1,5 nA**; faixas até 10 A com precisão ±0,25%–±0,3%; **comutação de shunt em 1 µs**. — Jetperch, JS110 User's Guide v1.3, https://download.joulescope.com/docs/JoulescopeUsersGuide/JoulescopeUsersGuide_v1_3.pdf
- **Joulescope JS220**: 2 MSPS (16 bits), **banda de 300 kHz**, ±15 V/±3 A contínuos, resolução até 0,5 nA, enwavify (comutação de faixa em 1 µs); faz *downsample* digital para 1 MSPS e permite downsampling configurável (ex.: **1 kHz** para capturas longas). — Jetperch, JS220 User's Guide, https://download.joulescope.com/products/JS220/JS220-K000/users_guide/Joulescope%20JS220%20User's%20Guide%20v1_10.pdf
- **Nordic PPK2**: **100 ksps**, faixa 200 nA–1 A, resolução de 100 nA a 1 mA conforme a faixa, modos amperímetro e fonte. — Nordic Semiconductor, PPK2, https://www.nordicsemi.com/Products/Development-hardware/Power-Profiler-Kit-2
- **Monsoon HVPM (AAA10F)**: taxa de amostragem **fixa de 5000 Hz**; suporta 0,8–13 V a 6 A. — Monsoon Solutions, https://www.msoon.com/faqs-troubleshooting
- **Monsoon LVPM (Mobile Device Power Monitor)**: faixa de 2,1–4,5 V, até 3 A contínuos, **frequência de corte do integrador de 300 kHz**, resolução de 2,86 µA na escala fina. — Monsoon, LVPM Manual, https://msoon.github.io/powermonitor/PowerTool/doc/LVPM%20Manual.pdf
- **INA226**: 16 bits, comum 0–36 V, **140 µs a 8,244 ms** de conversão, 330 µA. — TI, INA226 datasheet, https://www.ti.com/lit/ds/symlink/ina226.pdf
- **INA260**: 16 bits, **shunt integrado**, **140 µs a 8,244 ms**, 310 µA, erro de ganho 0,1%. — TI, INA260 datasheet, https://www.ti.com/lit/ds/symlink/ina260.pdf

**Síntese comparativa (faixa/taxa):** INA219/226/260 ficam na casa de **~1,9 kHz no melhor caso (12/16 bits, 1 amostra)** e degradam fortemente com averaging; PPK2 = **100 kHz**; Joulescope = **2 MHz**; Monsoon HVPM = **5 kHz**; Monsoon LVPM = integrador de 300 kHz. — URLs acima.

### 2.3 Que taxa é necessária (raciocínio ancorado em fontes)

- A MLPerf Tiny afirma que a **frequência de amostragem configurável é a taxa de dados retornados do EMON, muito menor que a taxa real do ADC**, e que isso **não impacta o score** porque as taxas internas ficam bem dentro da constante de tempo da capacitância de desacoplamento da entrega de energia do DUT. Isso vale quando a energia é integrada em **janelas longas (≥10 s)**. — MLPerf Tiny paper, https://arxiv.org/abs/2106.07597
- Para **energia por inferência** (janela curta), a metodologia de 2025 usa **disparo duplo GPIO** para delimitar a fase de inferência e mede com shunt de 50 mΩ ±1% na **alimentação do core**, com 1000 repetições. — Noronha et al., https://arxiv.org/abs/2505.15622
- O trabalho com RocketLogger rastreia os GPIOs de status “a até **64 kHz**” para delimitar inferência e camadas e correlacionar com energia. — “Measuring what Really Matters”, https://arxiv.org/pdf/2104.10645
- Há evidência publicada de derivação de **energia por inferência em ESP32-S3** a partir de sensoriamento de corrente em nível de placa com **shunt de 10 mΩ amostrado a 10 kHz** (INT8 reduz latência de 47 ms→15 ms no ESP32-S3 no MobileNetV2-0.35). — Noronha, “Quantization-driven latency and energy characterisation of TinyML inference on microcontroller-class edge devices”, https://exa.ai/library/publication/3r0lt7p95r9

> **Inferência (não é número de fonte):** 10 Hz de amostragem = uma amostra a cada 100 ms. Para uma inferência de, p.ex., ~15 ms (ESP32-S3 INT8 do MobileNetV2-0.35) ou ~81–750 ms (YOLO no Zero 2 W), a 10 Hz **não resolve** a dinâmica intra-inferência e, no caso de 15 ms, captura **menos de uma amostra por inferência**. As fontes acima usam 10 kHz–100 kHz–2 MHz justamente para isso.

### 2.4 Como calcular energia por inferência corretamente

- Fórmula física: **E = ∫ P(t) dt** na janela delimitada pela inferência; P(t) = V(t)·I(t). É exatamente o que EMONs com V e I sincronizados (JS110/JS220) fazem por construção; o INA219 entrega V e I em instantes distintos (§2.1). — Jetperch JS110, https://download.joulescope.com/docs/JoulescopeUsersGuide/JoulescopeUsersGuide_v1_3.pdf; TI INA219, https://www.ti.com/lit/ds/symlink/ina219.pdf
- **Incluir ou não o consumo estático/idle:** a MLPerf Tiny mede a **energia total na janela de tempo** (com o DUT alimentado apenas pela fonte de medição, sem baterias/supercapacitores) — ou seja, o idle dentro da janela entra naturalmente. — MLPerf Tiny paper, https://arxiv.org/abs/2106.07597
- A metodologia de 2025 **separa as fases** (pré-inferência, inferência, pós-inferência) e mede cada uma, permitindo isolar o custo do modelo do custo de setup/teardown. — Noronha et al., https://arxiv.org/abs/2505.15622
- Efeito do consumo estático no resultado: no ESP32-S3 o consumo de energia por inferência é **sublinear** em relação à latência porque um núcleo fica ocioso e mantém um **piso de leakage** durante inferência single-threaded (diferente de Cortex-M, onde energia acompanha a latência quase linearmente). — Noronha, https://exa.ai/library/publication/3r0lt7p95r9
- Consumo em **Light-sleep** do ESP32-S3 = 240 µA; **Deep-sleep** = 7–8 µA; em **Modem-sleep a 240 MHz** o chip pode ir de ~32,9 mA (WAITI) a ~81 mA (dual-core, 128-bit). Isso mostra que o modo de energia altera drasticamente o piso de potência. — Espressif, ESP32-S3 datasheet §4.6, https://documentation.espressif.com/esp32_s3_datasheet_en.pdf
- Padrões internos de medição de energia da MLPerf Tiny: **ST LPM01A**, **Jetperch Joulescope JS110** e **Keysight N6705**; só a energia entregue ao core é medida (a dos level shifters do framework é excluída). — MLPerf Tiny paper, https://arxiv.org/abs/2106.07597

---

## 3. Runtimes / toolchains

### 3.1 ESP32-S3 (Xtensa LX7 dual-core, até 240 MHz)

- ESP32-S3: Xtensa dual-core 32-bit LX7 até 240 MHz, **bus de dados de 128 bits e instruções SIMD**, FPU de precisão simples, **512 KB SRAM**, 384 KB ROM, PSRAM in-package de 2/8/16 MB. — Espressif, ESP32-S3 datasheet, https://documentation.espressif.com/esp32_s3_datasheet_en.pdf
- **ESP-NN** fornece kernels otimizados (usados pelo TFLite Micro). Ganho medido no exemplo Person Detection: ESP32-S3 **2300 ms → 54 ms** (240 MHz) com ESP-NN. — espressif/esp-tflite-micro, https://github.com/espressif/esp-tflite-micro
- **ESP-DL** (componente `espressif/coco_detect`) tem modelos YOLO11n quantizados (S8) por chip. Exemplos de latência no **ESP32-S3**: `yolo11n_320_s8_v3_s3` → pré 15,1 ms + modelo **6.184,2 ms** + pós 17,6 ms (input 320×320). — espressif/esp-dl, https://github.com/espressif/esp-dl/blob/master/models/coco_detect/README.md e https://components.espressif.com/components/espressif/coco_detect
- **ESP-Detection** (baseado em YOLOv11 + ESP-DL) publica latência por chip: `espdet_pico` (0,36 M params, 224×224) → **126,2 ms no ESP32-S3** (>7 FPS contando pré/pós) e 51,4 ms no ESP32-P4. — espressif/esp-detection, https://github.com/espressif/esp-detection
- **esp32-camera** (driver de câmera) suporta ESP32-S3 e sensores DVP (OV2640, OV5640 etc.); **exige PSRAM exceto em CIF ou menor com JPEG**; usar YUV/RGB “coloca muita carga no chip porque escrever na PSRAM não é particularmente rápido”, podendo perder dados de imagem; recomenda-se capturar JPEG e converter. Há opção `CONFIG_CAMERA_PSRAM_DMA` (ESP32-S2/S3). — espressif/esp32-camera, https://github.com/espressif/esp32-camera
- Limite de PCLK da interface de câmera: ESP32 8 MHz, ESP32-S2 32 MHz, **ESP32-S3 40 MHz** (interface LCD-CAM dedicada). — Espressif ESP-FAQ, https://docs.espressif.com/projects/esp-faq/en/latest/application-solution/camera-application.html
- **Suporte a YOLOX-nano especificamente no ESP32-S3 (ESP-DL/esp-tflite-micro): não encontrado** nas fontes primárias consultadas. Os modelos publicados da Espressif são YOLO11n (ESP-DL) e ESPDet (baseado em YOLOv11); há forks de terceiros do YOLOX-ti-lite para TFLite int8 voltados a MCUs/Nuvoton (ver §3.3), mas sem número publicado de latência no ESP32-S3.

### 3.2 Raspberry Pi Zero 2 W (Cortex-A53 quad-core @1 GHz, 512 MB)

- Hardware: Broadcom BCM2710A1 / SiP RP3A0, **quad-core 64-bit Arm Cortex-A53 @1 GHz**, **512 MB LPDDR2**, sem NPU; **entrada 5 V DC 2,5 A**. — Raspberry Pi, Zero 2 W product brief, https://datasheets.raspberrypi.com/rpizero2/raspberry-pi-zero-2-w-product-brief.pdf
- **TensorFlow Lite runtime**: estudo com YOLOv5n em PTQ FP32→INT8 no Zero 2 W relata **2,06× de speedup** e pico de **7,4 FPS** com INT8, redução de 74,6% no tamanho (7,5 MB→1,9 MB) e perda de 0,4% em mAP@0.50. — Jeya Agastin K. et al., “Real-Time Object Detection Using Raspberry Pi Zero 2W”, https://goldncloudpublications.com/index.php/irjaem/article/download/1264/1223/2604
- **NCNN**: benchmarks oficiais (issue/PR do repositório) no Zero 2 W (1 GHz, 4 threads): `yolo-fastestv2` **81,29 ms**, `yolo-fastest-1.1` 99,17 ms, `nanodet_m` 166,27 ms, `mobilenet_yolo` 747,53 ms, `yolov4-tiny` 646,64 ms. — Tencent/ncnn issue #3485, https://github.com/Tencent/ncnn/issues/3485
- **NCNN é otimizado para ARM NEON** e “supports 8-bit quantization”; em Pi 5, NCNN (11,74 FPS) superou ONNX (6,74) e PyTorch (5,64) no mesmo modelo. — Tencent/ncnn, https://github.com/Tencent/ncnn e https://kadirmertabatay.github.io/rpi-yolo-benchmark/comparisons/ncnn/
- **ONNX Runtime** e **TFLite** também rodam no Zero 2 W; em micro-benchmark de CNN pequena no RPi 4B, TFLite foi mais rápido que ORT e ncnn naquele caso específico — mostrando que o ranking depende do modelo/tamanho, não é universal. — Avafly/optimize-cnn, https://github.com/Avafly/optimize-cnn
- O **YOLOX-Nano int8 rodando concretamente no Zero 2 W: não encontrado** em fonte primária. O proxy publicado mais próximo é YOLOv5n int8 (7,4 FPS) e ncnn `yolo-fastestv2` (~12 FPS), ambos Cortex-A53 CPU-only.

### 3.3 Artefatos do YOLOX-Nano e quantização

- **YOLOX oficial (Megvii)**: YOLOX-Nano = **0,91 M parâmetros, 1,08 GFLOPs, 416×416, 25,3% AP em COCO**, superando NanoDet em 1,8% AP. — Ge et al., “YOLOX: Exceeding YOLO Series in 2021”, https://arxiv.org/abs/2107.08430
- O YOLOX oficial publica **ONNX** (tabela de modelos com pesos) e exporta via `tools/export_onnx.py`; suporta também NCNN/TensorRT/OpenVINO. — Megvii-BaseDetection/YOLOX, https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime/README.md
- **YOLOX-ti-lite → TFLite (inclui int8)**: repositório `motokimura/yolox-ti-lite_tflite` exporta ONNX→TFLite com `onnx2tf`, e a versão **int8 static quantized** usa calibração com `-oiqt -qcind images calib_data_416x416_n200.npy` (200 imagens de calibração), gerando `yolox_nano_ti_lite_integer_quant.tflite`. — motokimura, https://github.com/motokimura/yolox-ti-lite_tflite
- **YOLOX-nano int8 TFLite para MCU**: `OpenNuvoton/ML_YOLO` disponibiliza **YoloX-nano** com “Int8 Full Quantization TFLite” (`yolox_ti_lite_tflite_int8`), incluindo suporte ao **compilador Vela** para NPUs ARM. — OpenNuvoton/ML_YOLO, https://github.com/OpenNuvoton/ML_YOLO ; ver também MaxCYCHEN/yolox-ti-lite_tflite_int8, https://github.com/MaxCYCHEN/yolox-ti-lite_tflite_int8
- **YOLOX-Nano int8 ONNX**: variante comunitária `Thefalley/yolox-nano-relu-mit` publica `yolox_nano_relu_int8_qop.onnx`, quantizado com `onnxruntime.quantize_static` (per-tensor, W simétrico / A assimétrico, **1000 imagens aleatórias de calibração**, seed=42), treinado a partir do TI edgeai-yolox. — Hugging Face, https://huggingface.co/Thefalley/yolox-nano-relu-mit
- **Procedimento oficial de quantização pós-treino INT8 (TFLite)**: converter com `optimizations=[Optimize.DEFAULT]`, `representative_dataset`, `target_spec.supported_ops=[TFLITE_BUILTINS_INT8]`, `inference_input_type/output_type=tf.int8`; o dataset representativo pode ser um subconjunto pequeno (**~100–500 amostras**) do treino/validação. — TensorFlow, Post-training quantization, https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/post_training_quantization.md

---

## 4. Números de desempenho publicados (YOLO/YOLOX int8; ESP32-S3 e RPi Zero 2 W / proxies)

### 4.1 ESP32-S3

- YOLO11n (S8) 320×320, ESP-DL no ESP32-S3: **modelo 6.184,2 ms** + pré 15,1 ms + pós 17,6 ms. — Espressif ESP-DL, https://components.espressif.com/components/espressif/coco_detect
- YOLO11n (S8) 640×640 no ESP32-S3: **modelo 26.162,6 ms** (≈26 s), superando 8 MB de PSRAM (precisa de 16 MB). — Espressif ESP-DL, https://github.com/espressif/esp-dl/blob/master/models/coco_detect/README.md
- ESPDet-Pico 224×224 (0,36 M params): **126,2 ms** no ESP32-S3; 416×416: 449,5 ms. — espressif/esp-detection, https://github.com/espressif/esp-detection
- Person Detection (MobileNet típico, int8, TFLite Micro + ESP-NN): **54 ms** de `invoke()` no ESP32-S3. — espressif/esp-tflite-micro, https://github.com/espressif/esp-tflite-micro
- ESP-DL User Guide (v1.1) traz latências de referência por SoC (p.ex. ESP32-S3 18.909 µs / 16.614 µs conforme modelo), sem ser YOLOX. — Espressif ESP-DL, https://docs.espressif.com/projects/esp-dl/en/release-v1.1/esp32s3/performance.html

> **YOLOX-Nano int8 especificamente no ESP32-S3: não encontrado** (latência/FPS/RAM). Todos os números acima são YOLO11n/ESPDet/MobileNet, servem como ordem de grandeza, não como baseline do modelo alvo.

### 4.2 Raspberry Pi Zero 2 W (e proxies)

- Zero 2 W (ncnn, CPU only): `yolo-fastestv2` **81,29 ms** (~12,3 FPS), `nanodet_m` **166,27 ms** (~6 FPS), `mobilenet_yolo` 747,53 ms. — Tencent/ncnn #3485, https://github.com/Tencent/ncnn/issues/3485
- Zero 2 W, YOLOv5n int8 via TFLite: **7,4 FPS**, mAP@0.50 −0,4%, modelo 1,9 MB. — Jeya Agastin K. et al., https://goldncloudpublications.com/index.php/irjaem/article/download/1264/1223/2604
- Proxy em RPi 4B (YOLOv5n int8, TFLite): 320px **7,2 FPS**, 416px 4,5 FPS, 640px 1,9 FPS; RPi 4 é mais rápido que o Zero 2 W, então o Zero 2 W deve ficar **abaixo** desses valores. — kiena-dev/YOLOv5-tensorflow-lite-Raspberry-Pi, https://github.com/kiena-dev/YOLOv5-tensorflow-lite-Raspberry-Pi
- Referência de RAM/limite: Zero 2 W tem **512 MB LPDDR2** compartilhados, sem NPU. — Raspberry Pi product brief (URL §3.2)

> **YOLOX-Nano int8 especificamente no Zero 2 W: não encontrado.** Os proxies apontam para ordem de grandeza de **~5–12 FPS** em variantes “nano/tiny” int8, a depender do runtime e resolução.

---

## 5. Datasets de EPIs

| Dataset | Classes | Imagens | Instâncias | Licença / disponibilidade | Fonte |
|---|---|---|---|---|---|
| **CHV** (Color Helmet and Vest) | 6 (person, vest, 4 cores de capacete) | 1.330 | — | “open for public use” (Google Drive / Baidu) | Wang et al. 2021, https://github.com/zijianwang-zw/ppe_detection ; paper https://www.sciencedirect.com/science/article/abs/pii/S0957417420308715 |
| **SH17** | 17 (person, head, face, glasses, mask, face-guard, ear, earmuffs, hands, gloves, foot, shoes, safety-vest, tools, helmet, medical-suit, safety-suit) | 8.099 | 75.994 | “research and commercial purposes” (GitHub/Zenodo/Kaggle) | Ahmad & Rahimi 2024, https://arxiv.org/abs/2407.04590 ; https://github.com/ahmadmughees/sh17dataset ; https://zenodo.org/records/12659325 |
| **MOCS** (Moving Objects in Construction Sites) | 13 (worker, cranes, excavator, truck, etc.) | 41.668 | 222.861 | disponível (coleta em 174 canteiros) | Anlab340 / MOCS, http://www.anlab340.com/Archives/IndexArctype/index/t_id/17.html ; via CSOD-24, https://irojournals.com/iroiip/article/download/775/689/1361 |
| **SHEL5K** | 6 (helmet, head, head with helmet, person with helmet, person without helmet, face) | 5.000 | 75.570 rótulos | público (Mendeley Data) | Sarker et al., Sensors 22(6):2315, https://mdpi-res.com/d_attachment/sensors/sensors-22-02315/... (via https://mdpi.com/1424-8220/22/6/2315) |
| **SHW / SHWD** | 1 (worker com/sem capacete) | 7.581 | 120.558 | público | citado em SH17, https://arxiv.org/html/2407.04590 |
| **GDUT-HWD** | 5 | 3.174 | 18.893 | público | citado em SH17, https://arxiv.org/html/2407.04590 |
| **Pictor-PPE** | 3 (worker, hat, vest) | 784 | — | público | citado em SH17, https://arxiv.org/html/2407.04590 |
| **Roboflow Universe (variantes CHV/PPE)** | variável (ex.: vest, person, blue/red/white/yellow helmet) | ~1,3k–vários | — | conforme a licença de cada workspace | Roboflow, https://universe.roboflow.com/conapi/chv_relabel ; https://universe.roboflow.com/rawabi-aldossary-0vynu/chv-pm6yc |

Notas:
- O CHV foi construído selecionando 1.330 imagens de alta qualidade entre ~10.000 da Internet e datasets abertos. — Wang et al., https://github.com/zijianwang-zw/ppe_detection
- Os dados do SH17 vêm do **Pexels** (“clear usage rights”) e a cobertura é global, minimizando viés regional; a resolução vai de 1920×1002 a 8192×5462, com muitos objetos pequenos (<1% da área da imagem). — Ahmad & Rahimi, https://zenodo.org/records/12659325
- O SH17 inclui **treino/validação já divididos** (`train_files.txt`/`val_files.txt`) e rótulos em YOLO e VOC. — Kaggle, https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection
- Baseline no SH17: YOLOv9-e atinge **70,9% mAP@50 / 48,7% mAP@50-95**; YOLOv9-t (menor) 58,5% mAP@50. — https://github.com/ahmadmughees/sh17dataset

---

## 6. Ameaças à validade (MCU vs SBC)

### 6.1 Raspberry Pi Zero 2 W — térmica, throttling e governor

- O SoC do Zero 2 W **começa a throttling em 80 °C** e **throttling máximo em 85 °C**, caindo até a frequência base (~600 MHz) e podendo desligar núcleos. — fórum oficial Raspberry Pi, https://forums.raspberrypi.com/viewtopic.php?t=350311 ; MakeUseOf, https://www.makeuseof.com/how-to-raspberry-pi-zero-2-w-overclock/
- Em teste com caixa oficial (plástico), o Zero 2 W atinge **81 °C a 100% de carga e sofre throttling**; nu (board only) 74 °C sem throttle; caixa metálica FLIRC 62 °C. Ambiente de teste ~22 °C. — PiCockpit, https://picockpit.com/raspberry-pi/does-the-raspberry-pi-zero-2-require-a-heatsink/
- Teste do Zero 2 W **sem throttling até 63 °C** com temperatura ambiente de 26–27 °C em benchmark multi-core — mas o próprio relatório detecta “silent throttling” em alguns cenários, mostrando que throttling nem sempre aparece na leitura de frequência. — CNX Software, https://www.cnx-software.com/2021/10/29/raspberry-pi-zero-2-w-mini-review-benchmarks-and-thermal-performance/
- Estudo acadêmico (Stressberry): no Zero 2 W **não-overclocked** a frequência ficou estável em **1000 MHz** durante 300 s de carga; com overclock a 1300 MHz, a frequência **caiu para 1000 MHz** para conter a temperatura. — Gamess & Hernandez, IJACSA 13(2), https://thesai.org/Downloads/Volume13No2/Paper_95-Performance_Evaluation_of_Different_Raspberry_Pi_Models.pdf
- Consequência metodológica: **travar o governor em `performance`** e controlar/registrar temperatura; caso contrário, `ondemand`/`powersave` e o throttle de 80 °C contaminam latência e energia. — Piotr Kołaczkowski, https://pkolaczk.github.io/estimating-benchmark-errors/ ; fórum RPi (URL acima)
- Enclausurar o Zero 2 W aumenta a probabilidade de throttling sob carga sustentada. — Raspberry Pi MagPi (via readmedium), https://readmedium.com/benchmarking-the-raspberry-pi-zero-2-w-b868accdbb50

### 6.2 ESP32-S3 — clock, DFS, modos de energia, PSRAM

- ESP32-S3 tem **DFS** e **Light-sleep automático** via `esp_pm_configure()`; a frequência de CPU pode mudar e o APB pode variar várias vezes por tick do RTOS — alterando timers e periféricos. Requer `CONFIG_PM_ENABLE`. — Espressif ESP-IDF, Power Management, https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/system/power_management.html
- Frequências selecionáveis: 80/160/240 MHz (CPU) e APB 80 MHz; lock `ESP_PM_CPU_FREQ_MAX` garante a máxima. Sem lock e com light-sleep habilitado, o sistema entra em Light-sleep, **aumentando a latência de interrupção**. — ESP-IDF Power Management, https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/system/power_management.html
- Para benchmark determinístico, **desabilitar PM/light-sleep** (ou travar no máximo) e registrar a configuração (`sdkconfig`) é essencial, pois o modo de energia muda o piso de corrente (Deep-sleep 7 µA vs Modem-sleep 33–81 mA a 240 MHz). — ESP32-S3 datasheet, https://documentation.espressif.com/esp32_s3_datasheet_en.pdf
- **PSRAM é lenta**: o driver de câmera alerta que escrever na PSRAM “não é particularmente rápido” e usar YUV/RGB pode causar perda de dados de imagem; recomenda JPEG + conversão. — espressif/esp32-camera, https://github.com/espressif/esp32-camera
- Variantes com PSRAM Octal (R8/R16V) têm **máxima temperatura ambiente reduzida a 65 °C** (85 °C se ECC ligado, perdendo 1/16 da PSRAM) — restrição térmica relevante para medições prolongadas. — ESP32-S3 datasheet, https://documentation.espressif.com/esp32_s3_datasheet_en.pdf
- O ESP32-S3 tem **L1 cache** para memória externa; desempenho de inferência depende de acesso alinhado e SIMD (como mostrado no caso Cortex-M), e a colocação de pesos/ativações em PSRAM vs SRAM altera latência. — ESP32-S3 datasheet (bus 128 bits/SIMD/L1 cache), https://documentation.espressif.com/esp32_s3_datasheet_en.pdf ; “Measuring what Really Matters” (acesso alinhado/SIMD), https://arxiv.org/pdf/2104.10645
- O `esp_timer`/timestamps de alta resolução e o toggle de GPIO são a base de cronometragem; a MLPerf Tiny exige resolução **≥1 ms** para latência e **borda de descida de GPIO** para sincronizar com o EMON. — MLPerf Tiny paper, https://arxiv.org/abs/2106.07597

### 6.3 Overhead de I/O, câmera e cronometragem por GPIO

- Pre/post-processing **não** entram na janela da MLPerf Tiny, mas variam muito entre implementações e são um dos principais vieses ao comparar latência “de pipeline”. — MLPerf Tiny paper, https://arxiv.org/abs/2106.07597
- No ESP32-S3, o custo de captura de câmera depende do formato e do número de frame buffers (`fb_count`): 1 buffer dá mais controle mas **maior tempo para obter o frame**; ≥2 buffers dobra o framerate com mais carga de CPU/memória. — espressif/esp32-camera, https://github.com/espressif/esp32-camera
- A nova metodologia TinyML mostra que **a fase de pré-inferência domina** o consumo/latência em alguns casos e que separar as fases evita atribuir custo de setup ao modelo. — Noronha et al., https://arxiv.org/abs/2505.15622
- A cronometragem por GPIO tem tempos de subida/descida “negligenciáveis comparados às fases”, mas ainda assim há atraso de propagação e jitter; MLPerf usa a borda para disparar o EMON externo. — Noronha et al., https://arxiv.org/abs/2505.15622 ; MLPerf Tiny paper, https://arxiv.org/abs/2106.07597
- **Erro sistemático da instrumentação** (ex.: queda de tensão do shunt, diferença de instante V/I do INA219) não se corrige com mais amostras, só calibração/correção — ponto crítico ao comparar energia entre MCU (poucos mW) e SBC (centenas de mW). — NIST TN 1830, https://nvlpubs.nist.gov/nistpubs/TechnicalNotes/NIST.TN.1830.pdf ; TI INA219 (V/I em instantes distintos), https://www.ti.com/lit/ds/symlink/ina219.pdf

---

## 7. Reprodutibilidade, report e defensabilidade de 30 execuções / 5 de warm-up

### 7.1 O que a literatura sustenta

- **MLPerf Tiny** não usa “30 execuções / 5 de warm-up”: usa **5 runs**, cada uma com ≥10 s e ≥10 iterações, e reporta a **mediana**. Portanto “30/5” **não deriva** da MLPerf; é uma escolha do protocolo, não um requisito. — MLPerf Tiny paper, https://arxiv.org/abs/2106.07597
- O estudo de rigor estatístico em benchmarks de VM usa justamente **30 invocações** por alternativa, com **ICs de 95%**, ANOVA e Tukey HSD; “best of 3/5/10/30” e “mean/median” foram mais enganosos que a abordagem com IC. Isso dá **suporte parcial** a “30 execuções”. — Bulej et al., https://dl.acm.org/doi/pdf/10.1145/1297846.1297891
- Para o Zero 2 W (modelo alvo), o **método dos 1000 disparos** da literatura TinyML mais recente é o mais conservador; a MLPerf fica em 5 runs longos. Existe um **continuum** de defensabilidade, não um número único. — Noronha et al., https://arxiv.org/abs/2505.15622
- **Warm-up:** a suposição de que existe “steady-state” após warm-up é **contradita empiricamente** (Barrett et al.); o número de iterações até estabilizar pode variar muito. Aquecimento ajuda com cache/DVFS/compilação JIT, mas “descartar 5” é um **palpite**, não um critério baseado em dado. — Antoch et al., https://dl.acm.org/doi/pdf/10.1145/3839448
- **Tamanho amostral vs variabilidade:** o OSDI'18 mediu CV de 0,3% a 29,2% em 70 combinações benchmarks×hardware; para atingir IC de 1%, CV=0,3% exigiu ~10 experimentos e CV=9% exigiu **até ~240**. Incluir **um outlier** aumentou em **2,1–5,9×** o número de repetições recomendado. — Maricq et al., “Taming Performance Variability”, https://www.usenix.org/system/files/osdi18-maricq.pdf
- Implicação direta: **30 execuções pode ser suficiente para CV baixo (<~4%)**, mas **insuficiente** se a distribuição for assimétrica/multimodal ou tiver outliers (típico de SBC com scheduler/throttle). — Maricq et al., https://www.usenix.org/system/files/osdi18-maricq.pdf
- Reportar **média±dp** é desejável, mas a **mediana + IQR** é mais robusta a outliers e a distribuições assimétricas; ICs (t-Student) e análise de dependência temporal (autocovariância) são recomendadas. — Piotr Kołaczkowski, https://pkolaczk.github.io/estimating-benchmark-errors/ ; NIST TN 1830, https://nvlpubs.nist.gov/nistpubs/TechnicalNotes/NIST.TN.1830.pdf

### 7.2 Boas práticas de reprodutibilidade/report (a incorporar)

1. **Versionar** firmware, SDK (ESP-IDF/TFLite/NCNN), compilador, modelo (hash), runtime e config (governor, PM, clock). — Weber et al., https://pmc.ncbi.nlm.nih.gov/articles/PMC6584985/
2. Publicar **código, dados brutos e scripts de análise**. — Weber et al. (regra 10), https://pmc.ncbi.nlm.nih.gov/articles/PMC6584985/ ; Antoch et al. (pacote de reprodutibilidade), https://dl.acm.org/doi/pdf/10.1145/3839448
3. Definir e **isolar as fases** (pré/inferência/pós) para não contaminar latência/energia. — Noronha et al., https://arxiv.org/abs/2505.15622
4. Reportar **IC**, não só um número; usar ANOVA/Tukey para comparar plataformas. — Bulej et al., https://dl.acm.org/doi/pdf/10.1145/1297846.1297891
5. **Controlar fatores sistemáticos** (térmica, governor, PM, tensão) e registrá-los. — NIST TN 1830, https://nvlpubs.nist.gov/nistpubs/TechnicalNotes/NIST.TN.1830.pdf
6. Tratar outliers de forma **transparente** (não silenciosamente), ciente de que um único outlier muda o N necessário. — Maricq et al., https://www.usenix.org/system/files/osdi18-maricq.pdf

---

## Lacunas / decisões pendentes

1. **Energia estática (idle/leakage) entra ou não na energia por inferência?**
   - Medir energia total da janela (como MLPerf Tiny) vs subtrair o piso idle (energia líquida da inferência). O ESP32-S3 tem piso de leakage quando um núcleo fica ocioso (§2.4), o que muda a conclusão sobre eficiência. **Decidir e documentar a convenção.**

2. **Taxa de amostragem do medidor.** O INA219 a 10 Hz é **insuficiente** para inferências curtas (§2.3). Decidir:
   - usar o INA219 na **taxa máxima** (12 bits, 1 amostra = 532 µs ≈ 1,9 kHz) e integrar por GPIO; **ou**
   - migrar para **Joulescope JS110/JS220 (2 MSPS)** ou **PPK2 (100 ksps)**.
   - Aceitar que o INA219 mede V e I em **instantes diferentes** (até 68 ms no modo 128 médias) e quantificar o erro.

3. **Aquecimento: descartar 5 é defensável?** A literatura não sustenta “5” como norma (MLPerf não descarta; steady-state pode não existir). Decidir:
   - descarte fixo (5) **vs** detecção de estabilização por dado (ex.: desvio relativo < x% entre janelas);
   - tempo total de aquecimento (especialmente no Zero 2 W, que aquece em segundos).

4. **Número de repetições (30 é suficiente?).** Depende do CV medido (§7.1). Plano aberto:
   - rodar um **piloto** para estimar CV/IQR e então calcular N necessário (referência: ~10 para CV 0,3%; ~240 para CV 9%);
   - decidir se N varia por plataforma.

5. **Tratamento de outliers.** Definir ex-ante (ex.: regra IQR de Tukey, remoção apenas com justificativa e report do N original) e ser transparente, dado o efeito desproporcional no N (§7.1).

6. **O que exatamente é cronometrado.** Só a chamada de inferência (`invoke()`), ou pré+inferência+pós? MLPerf tira pré/pós; a nova metodologia separa fases. **Decidir e reportar as três.**

7. **YOLOX-Nano int8: não há números publicados nas duas plataformas** (§4). Decisão de viés:
   - é o primeiro benchmark, logo não há baseline externo — definir como comparar (mesmas imagens de entrada, mesmo runtime por plataforma, medir FPS de pós-NMS inclusive?);

8. **DS do modelo de EPI.** Escolher entre CHV (1.330 imgs, 6 classes, leve), SH17 (8.099 imgs, 17 classes, mais rico, licença research+comercial), SHEL5K/CSOD-24. Decidir classes efetivas (foco EPI: capacete, colete, máscara, óculos) e se o modelo será re-treinado ou apenas quantizado.

9. **Runtime por plataforma.** ESP32-S3: **ESP-DL/esp-tflite-micro + ESP-NN** (YOLOX precisa port/quantização própria — não há artefato oficial). Zero 2 W: **NCNN** (NEON, +int8) tende a ser o mais rápido, mas TFLite é o mais compatível com o artefato YOLOX-ti-lite int8. **Decidir e declarar** (comparação entre runtimes diferentes é uma ameaça à validade).

10. **Dataset representativo de calibração int8.** Definir N (TFLite recomenda ~100–500; o pipeline YOLOX-ti-lite usou 200) e a origem (EPIs do próprio DS), para não “vazar” dados de validação.

11. **Controle do ambiente.** Temperatura, ventilação/heatsink, governor do Zero 2 W, PM/light-sleep do ESP32-S3, tensão de alimentação e capacitância do shunt — registrar tudo.

12. **Métrica de eficiência.** Além de inferências/J, considerar **EDP (Energy-Delay Product)**, proposto na literatura TinyML, que penaliza conjuntamente latência e energia. — Noronha et al., https://arxiv.org/abs/2505.15622
