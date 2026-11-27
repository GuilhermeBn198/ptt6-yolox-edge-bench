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

## Resultados consolidados (artigo em `docs/projeto/main.pdf`)

Todos os números abaixo saem de [`scripts/analise_artigo.py`](scripts/analise_artigo.py) (semente fixa, a partir de `data/raw/`),
gravados em [`data/processed/artigo/resultados.json`](data/processed/artigo/resultados.json). Energia **estimada** na entrada de 5 V de
cada placa pelo modelo de [`scripts/power_model.toml`](scripts/power_model.toml) (SBC: medições publicadas; MCU: datasheet do
ESP32-S3 v2.2, Tab. 5-9); entre colchetes, os cenários mínimo e máximo.

| Configuração (416×416, E3, n = 30) | t_pre (ms) | t_inf (ms) | t_post (ms) | t_total (ms) | inf/s | E nominal [mín–máx] (mJ) |
|---|---:|---:|---:|---:|---:|---:|
| SBC int8, 4 threads | 8,18 | 109,44 | 5,26 | 122,91 | 8,14 | 265 [213–335] |
| SBC int8, 2 threads | 8,21 | 162,71 | 5,53 | 176,41 | 5,67 | 247 [192–316] |
| SBC int8, 1 thread | 8,14 | 270,72 | 5,47 | 284,18 | 3,52 | 284 [213–369] |
| SBC fp32, 4 threads | 6,14 | 236,47 | 7,12 | 249,61 | 4,01 | 557 [447–703] |
| MCU int8 (ESP32-S3) | 30,64 | 4.948,82 | 1,72 | 4.981,17 | 0,20 | 2.192 [1.190–3.935] |

- **E0 (teste, 141 imagens, BGR)**: fp32 mAP@50 55,0% (IC 95% 50,5–61,6), int8 16,7% (13,9–19,3); retenção ρ = 0,30 (0,26–0,34).
  O artefato int8 não atinge a referência de 0,99 (ameaça à validade declarada no artigo).
- **E1**: PC × SBC (LiteRT) equivalentes (97,3% das caixas com IoU ≥ 0,95); entre runtimes (LiteRT, TFLM de referência, ESP-NN)
  as saídas divergem, mas a qualidade de detecção nas 50 entradas é a mesma (F1 = 0,32).
- **H1**: throughput confirmado (SBC 40,5× mais rápido); parte energética não sustentada (MCU gasta 8,3× mais por inferência no
  nominal, 3,6–18,5× nos extremos).
- **Energia por ciclo**: o MCU gasta menos em todo período que consegue cumprir (≥ 4,98 s) no cenário nominal (SBC 1,45× mais a
  0,2 Hz; 2,4× a uma inspeção por minuto); no pior caso, só a partir de 113 s.
- **H2 (SBC)**: int8 × fp32: t_inf 2,16× menor e energia 2,10× menor.
- **E3b**: S(2) = 1,66×, S(4) = 2,47×, fração paralelizável efetiva p = 79,5%. **E4**: SBC 481 inferências em 60 s
  (8,02 inf/s, sem *throttling*); MCU 13 em 64,8 s.

## Reprodutibilidade

1. **Preparação das 50 entradas BGR ($416\times 416\times 3$ `uint8`) e avaliação de acurácia `E0` (COCO + bootstrap)**:
   ```bash
   python3 scripts/prepare_50_test_inputs.py
   python3 scripts/eval_map_tflite.py
   ```
2. **Execução completa da bateria no Raspberry Pi Zero 2 W (`E0` ocioso, `E1`, `E2`, `E3`, `E3b`, `E4`, `E5`)**:
   ```bash
   bash raspberry/pi_zero_2w/setup_rpi.sh
   python3 raspberry/pi_zero_2w/run_all_v13_experiments.py
   ```
3. **Compilação, gravação e coleta serial completa no ESP32-S3 (`E1`, `E2`, `E3`, `E4`, `E5` — ESP-IDF v6.1)**:
   ```bash
   bash firmware/esp32/esp_idf/flash_and_benchmark.sh /dev/ttyACM0
   python3 telemetry/record_esp32_serial.py --port /dev/ttyACM0
   ```
4. **Reprodução determinística das tabelas e figuras do artigo (`docs/projeto/main.pdf`) e da apresentação (`docs/apresentacao/slides.pdf`)**:
   ```bash
   python3 scripts/analise_artigo.py
   python3 scripts/figuras_artigo.py
   python3 telemetry/energy_and_stats_v13.py
   python3 telemetry/plot_phase4_v13.py
   ```

## Entregáveis

- [x] Repositório versionado com licença ([`LICENSE`](LICENSE)) e metadados de citação ([`CITATION.cff`](CITATION.cff))
- [x] README de reprodutibilidade e roteiro operacional ([`docs/roteiro_implementacao.qmd`](docs/roteiro_implementacao.qmd))
- [x] Dados brutos no SBC (`data/raw/SBC_*.csv`, `rpi_*.csv` e `*.json`), na MCU (`data/raw/MCU_E3_int8.csv`, `esp32_*`, `mcu_*`) e relatórios processados (`data/processed/artigo/*` e `data/processed/phase4_*`)
- [x] Figuras científicas em alta resolução (`docs/projeto/figs/fig_*.pdf` e `report/figures/fig1..fig6` em `.png` e `.pdf`)
- [x] Execução física final na MCU (ESP32-S3) e no SBC (Raspberry Pi Zero 2 W) com validação de previsão (`E1`), piloto (`E2`), benchmark em 3 blocos (`E3`/`E3b`), laço contínuo de 60 s (`E4`) e memória/energia (`E5`)
- [x] Vídeo de demonstração (3–5 min) nas duas plataformas ([`video/README.md`](video/README.md) — [YouTube](https://youtu.be/qOvJVo9876Y)), slides ([`docs/apresentacao/slides.pdf`](docs/apresentacao/slides.pdf)) e artigo final em PDF ([`docs/projeto/main.pdf`](docs/projeto/main.pdf))
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
