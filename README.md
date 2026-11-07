# PTT6 — Benchmarking de Inferência do YOLO Quantizado em MCU e SBC

> Desempenho e eficiência energética de **YOLOX-Nano (int8)** para **detecção de EPIs** em cenários de **Edge AI**.

**Discentes**: Filipe Matias & Guilherme Lucas
**Orientador**: Herbert Oliveira Rocha
**Programa**: Mestrado Profissional em Computação Aplicada (PPCA/UFRR)
**Disciplina**: Arquitetura e Organização de Computadores
**Trilha**: PTT6 — Benchmarking e eficiência energética

---

## Sobre

Este repositório contém o experimento comparativo de inferência de um único modelo
(YOLOX-Nano, quantizado em **int8**) executado em duas plataformas representativas do
espectro Edge AI:

| Plataforma | Tipo | Papel no experimento |
|------------|------|----------------------|
| **ESP32-S3 (Xtensa LX7 @ 240 MHz, 8 MB Octal PSRAM)** | MCU | Extremo de baixo consumo (com instruções SIMD PIE via `esp-nn` + `esp-tflite-micro`) |
| **Raspberry Pi Zero 2 W (ARM Cortex-A53 @ 1,0 GHz)** | SBC | Extremo de maior throughput/recursos (`ai-edge-litert` + XNNPACK NEON) |

**Pergunta de pesquisa**: como se comparam latência, throughput e energia por inferência
entre a MCU e o SBC ao executar o mesmo pipeline?

**Hipótese H1**: o SBC apresenta throughput maior, mas energia por inferência igual ou
superior, dado seu consumo estático mais alto.

> **Nota de plataforma:** além da previsão de contingência para o ESP32 (LX6) na emenda **v1.1** (decisão D23), a execução física final foi realizada diretamente no **ESP32-S3 (QFN56 rev v0.2, 240 MHz, 8 MB Octal PSRAM @ 80 MHz)** na resolução cheia de $416\times 416$. Ver [`docs/protocolo.md`](docs/protocolo.md) §13 e [`docs/roteiro_implementacao.qmd`](docs/roteiro_implementacao.qmd).

## Métricas

- **Latência** de inferência (ms), de ponta a ponta (pré + inferência + pós)
- **Throughput** (inferências/s)
- **Energia por inferência** (mJ, **estimada por software**) e eficiência normalizada
  (**inferências/joule** e **EDP**)
- **Acurácia**: mAP@50 / mAP@50-95 no teste (condição de validade das métricas de energia/latência — v1.5)

## Protocolo experimental (resumo)

> Detalhes e justificativas em [`docs/protocolo.md`](docs/protocolo.md) (protocolo congelado + emendas v1.1–v1.5).

- **Dataset:** Ultralytics **Construction-PPE** — começa com 1 classe (colete) e depois 11 classes ([`docs/dataset.md`](docs/dataset.md))
- **Desenho experimental e coleta de métricas (v1.3):** [`docs/experimentacao.md`](docs/experimentacao.md)
- Tarefa de **detecção** (bounding boxes), mesmo artefato **YOLOX-Nano int8 → TFLite** (mesmo
  runtime e mesmo pipeline nas duas plataformas; hash do modelo registrado)
- **Entrada**: imagens do dataset **ingeridas diretamente no modelo** (serial no ESP32, disco no RPi);
  a **câmera ao vivo é extra** (decisão D33, emenda v1.4)
- Repetições definidas por **piloto** (CV → N para IC 95%; **piso 30**); **warm-up definido pelo dado**
- **Energia estimada por software**: modelo de potência por modo (datasheet) × tempo medido por fase
  (`E = Σ P_modo · t_fase`); **sem medidor físico inline** (decisão D24) — limitação declarada
- Estatística robusta: mediana + IQR, **Mann-Whitney U** + tamanho de efeito + IC por bootstrap
- Ameaça principal: **variação térmica** — mitigada com pausas de resfriamento e ambiente registrado
- **[Arquitetura e diagramas](docs/arquitetura/README.md)**: contexto/C4, fluxo de dados (DFD),
  casos de uso, sequência de execução, fases, implantação e modelo de dados

## Estrutura do repositório

```
models/               # Artefatos YOLOX-Nano TFLite (int8 e float32) e scripts de calibração/conversão
firmware/esp32/       # ESP32 (LX6 / S3): bench_serial.ino (Arduino) e esp_idf/ (ESP-IDF v6.1 + TFLite Micro)
raspberry/pi_zero_2w/ # Pipeline e orquestrador v1.3 no Raspberry Pi Zero 2 W (benchmark_yolox.py, run_all_v13_experiments.py, bench_collect.py)
telemetry/            # Fase 4 (E6): modelo de energia, estatística robusta (energy_and_stats_v13.py) e gráficos (plot_phase4_v13.py)
scripts/              # energy_model.py + power_model.toml + esp_serial_feeder.py
data/raw/             # Dados brutos das execuções no SBC e no ESP32-S3 (CSV, JSON e logs: E0, E1, E3, E3b, E3c, E4, E5)
data/processed/       # Relatórios consolidados da Fase 4 (phase4_e6_report.json, phase4_energy_stats_summary.csv)
report/figures/       # Figuras científicas geradas em PNG (300 DPI) e PDF vetorial (fig1 a fig6)
video/                # Link/roteiro do vídeo de demonstração (3–5 min)
docs/                 # Proposta, protocolo, experimentação, dataset, roteiro_implementacao.qmd, scripts.md, plano de ação
docs/projeto/         # Projeto em LaTeX (main.tex → main.pdf)
docs/arquitetura/     # Diagramas de arquitetura (C4, DFD, casos de uso, sequência)
.opencode/skills/     # Skill do agente com a base de conhecimento do projeto
```

## Resultados Experimentais Consolidados: SBC (Raspberry Pi Zero 2 W) vs. MCU (ESP32-S3) — Protocolo v1.3

Todos os experimentos do Protocolo v1.3 foram executados fisicamente em ambas as plataformas na resolução cheia de **$416\times 416$** (3 blocos independentes $\times$ 10 execuções válidas = 30 amostras por regime):
- **SBC**: Raspberry Pi Zero 2 W (ARM Cortex-A53 @ 1,0 GHz, governador `performance`, runtime `ai-edge-litert` v2.3.0, 10 *warmups*/bloco).
- **MCU**: ESP32-S3 (Xtensa LX7 dual-core @ 240 MHz, 8 MB Octal PSRAM `AP_3v3` @ 80 MHz, ESP-IDF v6.1 + `esp-tflite-micro` v1.4.2 + `esp-nn` v1.4.2 SIMD, 2 *warmups*/bloco).

| Plataforma / Configuração ($416\times 416$) | Mediana $\tilde{t}_{\text{pre}}$ | Mediana $\tilde{t}_{\text{inf}}$ (IQR) | Mediana $\tilde{t}_{\text{post}}$ | Mediana $\tilde{t}_{\text{total}}$ | Throughput ($\text{inf/s}$ / $\text{total/s}$) | Energia $E_{\text{inf}}$ [$P_{\min}\text{–}P_{\max}$] | Eficiência $\eta$ | $\text{EDP}$ ($\text{mJ}\cdot\text{ms}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SBC INT8 — 4 Threads (`E3`/`E3b`)** | 33,18 ms | **109,52 ms** (0,25 ms) | 6,05 ms | **148,76 ms** | **9,09** / 6,70 FPS | **253,13 mJ** $[203,61\text{–}319,17]$ | **3,95 inf/J** | **27.871** |
| **SBC INT8 — 2 Threads (`E3b`)** | 33,17 ms | **162,77 ms** (0,14 ms) | 6,05 ms | **202,00 ms** | **6,14** / 4,95 FPS | **268,64 mJ** $[211,65\text{–}341,90]$ | **3,72 inf/J** | **43.736** |
| **SBC INT8 — 1 Thread (`E3b`)** | 33,14 ms | **270,64 ms** (0,24 ms) | 6,02 ms | **309,89 ms** | **3,69** / 3,23 FPS | **270,68 mJ** $[203,01\text{–}351,88]$ | **3,69 inf/J** | **73.267** |
| **SBC Float32 — 4 Threads (`E3c`)** | 34,54 ms | **236,60 ms** (0,71 ms) | 8,52 ms | **279,65 ms** | **4,21** / 3,57 FPS | **545,94 mJ** $[439,13\text{–}688,36]$ | **1,83 inf/J** | **129.600** |
| **MCU ESP32-S3 INT8 (`E3`)** | 49,11 ms | **4953,29 ms** (0,003 ms) | **1,60 ms** | **5004,00 ms** | **0,202** / 0,200 FPS | **1109,54 mJ** $[490,38\text{–}2347,86]$ | **0,90 inf/J** | **5.495.873** |

- **Previsão On-Device e Equivalência Funcional (`E1`, [`data/raw/esp32_e1_predictions.json`](data/raw/esp32_e1_predictions.json))**: Em `test_image.jpg` ($416\times 416$), o ESP32-S3 detecta com precisão a classe primária `Person` (`cls_id=6`, `score=0.6111`, `bbox=[155.00, 71.20, 278.35, 268.18]`) e a classe secundária `none` (`cls_id=5`, `score=0.3222`, `bbox=[167.73, 103.97, 274.70, 265.65]`). O pós-processamento C++ `-O3` na MCU ($\tilde{t}_{\text{post}} = 1{,}60\,\text{ms}$) é **$3{,}78\times$ mais rápido** que a implementação Python/NumPy no SBC ($6{,}05\,\text{ms}$).
- **Pegada de Memória (`E5`, [`data/raw/esp32_e5_memory_footprint.json`](data/raw/esp32_e5_memory_footprint.json))**: Mesmo modelo `.tflite` INT8 em Flash ($1{,}158\,\text{MiB}$ vs. $3{,}490\,\text{MiB}$ em FP32); na MCU, a *Tensor Arena* do TFLite Micro ocupa apenas **$2{,}122\,\text{MiB}$ ($2172{,}43\,\text{KiB}$)** na Octal PSRAM (restando $353{,}59\,\text{KiB}$ de SRAM interna livre e $4093{,}60\,\text{KiB}$ de PSRAM livre), uma pegada de RAM **$28{,}6\times$ menor** que o pico de RSS do processo Linux no SBC ($60{,}60\,\text{MiB}$).
- **Lei de Amdahl no SBC (`E3b`)**: Speedup $S(2) = 1{,}6628\times$ ($\text{IC}_{95\%} [1{,}6621;\,1{,}6635]$) e $S(4) = 2{,}4712\times$ ($\text{IC}_{95\%} [2{,}4693;\,2{,}4727]$, $p = 3{,}02\times 10^{-11}$, $\delta = -1{,}0$), com fração paralelizável $p = 79{,}38\%$ e fração serial $1-p = 20{,}62\%$.
- **Estabilidade Contínua de 60 s no SBC (`E4`)**: 399 inferências ininterruptas (`delay=0`), mediana $\tilde{t}_{\text{inf}} = 109{,}57\,\text{ms}$, temperatura máxima $71{,}4\,^\circ\text{C}$ ($< 80\,^\circ\text{C}$) e `throttled=0x0`.
- **Teste Formal da Hipótese $H_1$ (`E6` — Teste de Interseção-União)**:
  1. **Parte 1 ($\text{Throughput}_{\text{SBC}} > \text{Throughput}_{\text{MCU}}$)**: **Confirmada** ($9{,}09 > 0{,}202\,\text{inf/s}$, razão de medianas $45{,}2283\times$, $\text{IC}_{95\%\,\text{BCa}} [45{,}1980;\,45{,}2504]$, Mann-Whitney $U = 900$, $p = 1{,}24\times 10^{-11}$, $\delta = +1{,}00$).
  2. **Parte 2 ($E_{\text{SBC}} \ge E_{\text{MCU}}$)**: **Refutada em $416\times 416$** ($E_{\text{SBC,nom}} = 253{,}13\,\text{mJ} < E_{\text{MCU,nom}} = 1109{,}54\,\text{mJ}$, $\delta = -1{,}00$). Como a razão de latências ($t_{\text{MCU}} / t_{\text{SBC}} = 45{,}23\times$, imposta pelo gargalo de barramento SPI Octal PSRAM a 80 MHz para uma arena de $2{,}12\,\text{MiB}$) supera a razão crítica de potências ($P_{\text{SBC}} / P_{\text{MCU}} = 10{,}27\times$ nominal ou $3{,}90\times$ com *overhead* de placa, cujo *break-even* exigiria $t_{\text{MCU}} \le 1124{,}5\,\text{ms}$), o SBC atinge menor energia por inferência pelo princípio arquitetural de **Race-to-Idle**.

## Reprodutibilidade

1. **Execução completa da bateria v1.3 no Raspberry Pi Zero 2 W**:
   ```bash
   bash raspberry/pi_zero_2w/setup_rpi.sh
   python3 raspberry/pi_zero_2w/run_all_v13_experiments.py
   ```
2. **Compilação, gravação e coleta serial no ESP32-S3 (ESP-IDF v6.1)**:
   ```bash
   bash firmware/esp32/esp_idf/flash_and_benchmark.sh /dev/ttyUSB0
   ```
3. **Análise energética, estatística robusta (`E6`) e geração das 6 figuras científicas**:
   ```bash
   python3 telemetry/energy_and_stats_v13.py
   python3 telemetry/plot_phase4_v13.py
   ```

## Entregáveis

- [x] Repositório versionado com licença
- [x] README de reprodutibilidade e roteiro operacional ([`docs/roteiro_implementacao.qmd`](docs/roteiro_implementacao.qmd))
- [x] Dados brutos no SBC (`data/raw/rpi_*.csv` e `*.json`), na MCU (`data/raw/esp32_*`) e relatórios processados (`data/processed/phase4_*`)
- [x] Figuras científicas em alta resolução (`report/figures/fig1..fig6` em `.png` e `.pdf`)
- [x] Execução física final na MCU (ESP32-S3) com validação de previsão (`E1`), benchmark em 3 blocos (`E3`) e memória (`E5`)
- [ ] Vídeo de demonstração (3–5 min) nas duas plataformas e submissão final do artigo PDF
- [ ] (pós-disciplina) Dataset com DOI no Zenodo + submissão a evento regional

## Plano de ação

O plano é acompanhado por **issues e milestones** no GitHub, espelhando as fases do
cronograma (15 semanas). Veja a aba [Issues](../../issues).

| Semanas | Fase |
|---------|------|
| 1–2 | Revisão e protocolo |
| 3–5 | Implementação do pipeline |
| 6–9 | Execução dos experimentos |
| 10–11 | Análise dos dados |
| 12–13 | Redação do relatório |
| 14–15 | Vídeo e entrega final |

## Base de conhecimento (skill do agente)

A proposta foi convertida em uma skill reutilizável em
[`.opencode/skills/ptt6-yolox-edge-bench/`](.opencode/skills/ptt6-yolox-edge-bench/SKILL.md),
cobrindo pergunta de pesquisa, hipótese, protocolo, riscos/Plano B, entregáveis e
um cheatsheet de decisão.

## Referências preliminares

- Patterson, D. A.; Hennessy, J. L. *Computer Organization and Design*.
- Datasheets: ESP32; Raspberry Pi Zero 2 W.
- YOLOX — https://github.com/Megvii-BaseDetection/YOLOX

## Licença

MIT — veja [LICENSE](LICENSE). Os dados publicados seguem CC-BY-4.0.
