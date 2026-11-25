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

## Resultados Experimentais Consolidados: SBC (Raspberry Pi Zero 2 W) vs. MCU (ESP32-S3) — Protocolo v1.5 (`docs/projeto/main.tex`)

Todos os experimentos previstos em [`docs/projeto/main.tex`](docs/projeto/main.tex) (`E0`, `E1`, `E2`, `E3`, `E3b`, `E4` e `E5`) foram executados fisicamente em ambas as plataformas na resolução comum de **$416\times 416$** sobre as **mesmas 50 entradas `.bin` do conjunto de teste em ordem de canais BGR** (pré-carregadas em RAM antes da janela cronometrada, com hash do modelo `.tflite` INT8 `SHA-256 = 7205d9dd41fb532c54eec2283a0f78e8aa491ed3f8c42b6b46c4e24df1778f54`):
- **SBC**: Raspberry Pi Zero 2 W (ARM Cortex-A53 @ 1,0 GHz, governador `performance`, runtime `ai-edge-litert` v2.3.0, 3 blocos $\times$ 10 execuções válidas + 10 *warmups*/bloco).
- **MCU**: ESP32-S3 (Xtensa LX7 dual-core @ 240 MHz, 8 MB Octal PSRAM `AP_3v3` @ 80 MHz, ESP-IDF v6.1 + `esp-tflite-micro` v1.4.2 + `esp-nn` v1.4.2 SIMD, ingestão serial `IMG` c/ CRC32, 3 blocos $\times$ 10 execuções válidas + 5 *warmups*/bloco).

| Plataforma / Configuração ($416\times 416$, BGR) | Mediana $\tilde{t}_{\text{pre}}$ | Mediana $\tilde{t}_{\text{inf}}$ (IQR) | Mediana $\tilde{t}_{\text{post}}$ | Mediana $\tilde{t}_{\text{total}}$ (IQR) | Throughput ($\text{total/s}$ / $\text{inf/s}$) | Energia 3 Fases $E_{3\text{F}}$ [$P_{\min}\text{–}P_{\max}$] | Energia $E_{\text{inf}}$ | Eficiência $\eta_{3\text{F}}$ / $\eta_{\text{inf}}$ | $\text{EDP}_{3\text{F}}$ ($\text{mJ}\cdot\text{ms}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SBC INT8 — 4 Threads (`E3`/`E3b`)** | 8,18 ms | **109,44 ms** (0,19 ms) | 5,26 ms | **122,91 ms** (1,30 ms) | **8,13** / 9,14 inf/s | **265,39 mJ** $[212,72\text{–}335,15]$ | 251,78 mJ | **3,77** / 3,97 inf/J | **32.665** |
| **SBC INT8 — 2 Threads (`E3b`)** | 8,40 ms | **162,71 ms** (0,07 ms) | 5,18 ms | **176,41 ms** (1,38 ms) | **5,67** / 6,15 inf/s | **282,28 mJ** $[221,88\text{–}359,65]$ | 268,48 mJ | **3,54** / 3,72 inf/J | **49.830** |
| **SBC INT8 — 1 Thread (`E3b`)** | 8,36 ms | **270,73 ms** (0,27 ms) | 5,19 ms | **284,18 ms** (1,36 ms) | **3,52** / 3,69 inf/s | **284,45 mJ** $[213,34\text{–}369,79]$ | 270,84 mJ | **3,52** / 3,69 inf/J | **80.915** |
| **SBC Float32 — 4 Threads (`E3` p/ `H2`)** | 6,34 ms | **236,47 ms** (1,24 ms) | 6,62 ms | **249,61 ms** (1,56 ms) | **4,00** / 4,22 inf/s | **557,98 mJ** $[448,10\text{–}704,05]$ | 544,95 mJ | **1,79** / 1,84 inf/J | **139.484** |
| **MCU ESP32-S3 INT8 (`E3` p/ `H1`)** | 30,64 ms | **4948,82 ms** (0,11 ms) | **1,72 ms** | **4981,17 ms** (0,28 ms) | **0,201** / 0,202 inf/s | **1115,79 mJ** $[493,14\text{–}2361,10]$ | 1108,53 mJ | **0,90** / 0,90 inf/J | **5.558.000** |

- **Acurácia Offline e Retenção $\rho$ com *Bootstrap* (`E0`, [`data/processed/e0_map_test.json`](data/processed/e0_map_test.json), 141 imagens de teste, 1.251 anotações, BGR, 1.000 reamostras)**:
  - **FP32 (`BGR`)**: $\mathrm{mAP@50} = 54{,}86\%$ ($\text{IC}_{95\%} [50{,}50\%;\,61{,}57\%]$) | $\mathrm{mAP@50:95} = 27{,}10\%$ ($\text{IC}_{95\%} [24{,}81\%;\,30{,}63\%]$) | $\mathrm{AP@50}(\text{vest}) = 88{,}82\%$ ($\text{IC}_{95\%} [81{,}75\%;\,95{,}23\%]$).
  - **INT8 (`BGR`)**: $\mathrm{mAP@50} = 15{,}97\%$ ($\text{IC}_{95\%} [13{,}91\%;\,19{,}30\%]$) | $\mathrm{mAP@50:95} = 6{,}14\%$ ($\text{IC}_{95\%} [5{,}28\%;\,7{,}57\%]$) | $\mathrm{AP@50}(\text{vest}) = 47{,}78\%$ ($\text{IC}_{95\%} [37{,}66\%;\,57{,}93\%]$).
  - **Retenção $\rho = \mathrm{mAP}_{\text{int8}} / \mathrm{mAP}_{\text{fp32}}$**: $\rho_{\mathrm{mAP@50}} = 0{,}2911$ ($\text{IC}_{95\%} [0{,}2557;\,0{,}3389]$) | $\rho_{\mathrm{mAP@50:95}} = 0{,}2266$ ($\text{IC}_{95\%} [0{,}1954;\,0{,}2693]$) | $\rho_{\mathrm{AP@50(vest)}} = 0{,}5379$ ($\text{IC}_{95\%} [0{,}4429;\,0{,}6324]$).
- **Equivalência Funcional nas 50 Imagens BGR (`E1`, [`data/processed/e1_equivalence_report.json`](data/processed/e1_equivalence_report.json))**:
  - **PC (`x86_64 LiteRT`) $\times$ SBC (`ARM64 LiteRT`)**: `148` vs. `148` detecções, $100\%$ de concordância no número de detecções (`50/50`) e nas classes (`50/50`), $\mathrm{IoU}$ médio de **$0{,}9962$** ($97{,}3\%$ com $\mathrm{IoU} \ge 0{,}95$) e $99{,}32\%$ com $|\Delta\text{score}| \le 1$ passo de quantização (`0,027536`).
  - **SBC/PC $\times$ MCU (`ESP32-S3 TFLM + ESP-NN SIMD`)**: `148` vs. `153` detecções nas 50 imagens (`117` pareadas da mesma classe com $\mathrm{IoU}$ médio de $0{,}7462$), documentando o efeito de arredondamento de requantização inteira entre os kernels `XNNPACK` (ARM NEON) e `ESP-NN` (Xtensa LX7). Em `01_image1.bin`, ambos detectam o capacete `helmet` (`cls_id=0`, `score=0,8492` na MCU vs. `0,7992` no SBC, `bbox=[268, 139, 316, 183]`).
- **Piloto de 100 Inferências (`E2`, [`data/raw/sbc_e2_pilot_summary.json`](data/raw/sbc_e2_pilot_summary.json) e [`data/raw/mcu_e2_pilot_summary.json`](data/raw/mcu_e2_pilot_summary.json))**:
  - **SBC**: $W_{\text{emp}} = 5$ (adotado $W = 10$), $\mathrm{CV}(t_{\text{total}}) = 1{,}140\% \implies N = 5 \implies N_{\text{final}} = 30$.
  - **MCU**: $W_{\text{emp}} = 5$ (adotado $W = 5$), $\mathrm{CV}(t_{\text{total}}) = 0{,}0064\% \implies N = 1 \implies N_{\text{final}} = 30$.
- **Throughput Sustentado em 60 s de Laço Fechado (`E4`, [`data/raw/rpi_e4_continuous_60s.csv`](data/raw/rpi_e4_continuous_60s.csv) e [`data/raw/esp32_e4_continuous_60s.csv`](data/raw/esp32_e4_continuous_60s.csv))**:
  - **SBC**: **481 inferências em $60{,}0\text{ s}$** (**$8{,}02\text{ inf/s}$** sustentado), $\tilde{t}_{\text{total}} = 123{,}38\text{ ms}$, temperatura máxima $65{,}0\,^\circ\text{C}$ e `throttled=0x0`.
  - **MCU**: **13 inferências em $64{,}75\text{ s}$** (**$0{,}201\text{ inf/s}$** sustentado), $\tilde{t}_{\text{total}} = 4981{,}00\text{ ms}$, deriva temporal nula.
- **Pegada de Memória (`E5`, [`data/raw/esp32_e5_memory_footprint.json`](data/raw/esp32_e5_memory_footprint.json))**: Mesmo modelo `.tflite` INT8 em Flash ($1{,}158\,\text{MiB}$ vs. $3{,}490\,\text{MiB}$ em FP32); na MCU, a *Tensor Arena* do TFLite Micro ocupa **$2{,}122\,\text{MiB}$ ($2172{,}43\,\text{KiB}$)** na Octal PSRAM ($28{,}6\times$ menor que o RSS do processo Linux no SBC).
- **Energia por Ciclo Periódico $E_{\text{ciclo}}(f)$ para $f \in \{0{,}2;\,0{,}5;\,1\}\text{ Hz}$ (`E5`, Seção 4.11 de `main.tex`)**:
  - Em **$f = 0{,}2\text{ Hz}$** (1 inspeção a cada $5\text{ s}$, onde $t_{\text{total,MCU}} = 4981{,}2\text{ ms} < 5000\text{ ms}$): devido ao consumo ocioso do SBC ($P_{\text{ocioso,SBC}} = 0{,}60\text{ W}$ durante $4{,}88\text{ s}$), o SBC consome **$E_{\text{ciclo,SBC}}(0{,}2\text{ Hz}) = 3191{,}54\text{ mJ/ciclo}$**, enquanto o ESP32-S3 consome apenas **$E_{\text{ciclo,MCU}}(0{,}2\text{ Hz}) = 1117{,}03\text{ mJ/ciclo}$** (**$2{,}86\times$ menor na MCU**)! Para $f \in \{0{,}5;\,1\}\text{ Hz}$, apenas o SBC atende à restrição de tempo real ($1391{,}54\text{ mJ}$ e $791{,}54\text{ mJ}$).
- **Teste Formal das Hipóteses $H_1$ e $H_2$ (`E5`/`E6`, [`data/processed/phase4_e6_report.json`](data/processed/phase4_e6_report.json))**:
  1. **Hipótese $H_1$ (Teste de Interseção-União — MCU vs. SBC em INT8)**:
     - **Parte 1 ($\text{Throughput}_{\text{SBC}} > \text{Throughput}_{\text{MCU}}$)**: **Confirmada** ($8{,}13 > 0{,}201\,\text{inf/s}$ na janela completa de 3 fases, speedup de **$40{,}5262\times$**, $\text{IC}_{95\%\,\text{BCa}} [40{,}2366;\,40{,}7773]$, Mann-Whitney $U = 900$, $p = 1{,}51\times 10^{-11}$, $\delta = +1{,}00$; em $t_{\text{inf}}$: speedup de **$45{,}2187\times$**).
     - **Parte 2 ($E_{\text{SBC}} \ge E_{\text{MCU}}$ por inferência isolada)**: **Refutada em $416\times 416$** ($E_{3\text{F,SBC}} = 265{,}39\,\text{mJ} < E_{3\text{F,MCU}} = 1115{,}79\,\text{mJ}$, robusta em todos os extremos da Tabela 5, incluindo $E_{\text{SBC,máx}} = 335{,}15\,\text{mJ} < E_{\text{MCU,mín}} = 493{,}14\,\text{mJ}$). Como $t_{\text{MCU}}/t_{\text{SBC}} = 40{,}53\times$ supera a razão crítica de potências ($9{,}64\times$ na janela de 3 fases ou $10{,}27\times$ em $t_{\text{inf}}$), o SBC consome menos energia por inferência ativa pelo princípio de **Race-to-Idle** (embora consuma $2{,}86\times$ mais energia por ciclo periódico em $f = 0{,}2\text{ Hz}$).
  2. **Hipótese $H_2$ (Trade-off INT8 vs. FP32 no SBC)**: **Confirmada** — a quantização INT8 reduz a latência total em **$2{,}0308\times$** ($\text{IC}_{95\%} [2{,}0213;\,2{,}0395]$, de $249{,}61$ para $122{,}91\text{ ms}$, $p = 1{,}51\times 10^{-11}$) e a energia de 3 fases em **$2{,}0990\times$** ($\text{IC}_{95\%} [2{,}0946;\,2{,}1042]$, de $557{,}98$ para $265{,}39\text{ mJ}$, $\delta = -1{,}00$), ao custo de uma retenção de acurácia quantificada em $\rho_{\mathrm{mAP@50}} = 0{,}2911$ ($\text{IC}_{95\%} [0{,}2557;\,0{,}3389]$) e $\rho_{\mathrm{AP@50(vest)}} = 0{,}5379$ ($\text{IC}_{95\%} [0{,}4429;\,0{,}6324]$).

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
