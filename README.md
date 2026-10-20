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
| **ESP32 (LX6)** | MCU | Extremo de baixo consumo (hardware disponível; ver emenda v1.1) |
| **Raspberry Pi Zero 2 W** | SBC | Extremo de maior throughput/recursos |

**Pergunta de pesquisa**: como se comparam latência, throughput e energia por inferência
entre a MCU e o SBC ao executar o mesmo pipeline?

**Hipótese H1**: o SBC apresenta throughput maior, mas energia por inferência igual ou
superior, dado seu consumo estático mais alto.

> **Nota de plataforma:** a proposta original citava o **ESP32-S3**; adotou-se o **ESP32 (LX6)**
> disponível, com aval do orientador (emenda **v1.1**, decisão D23). Ver
> [`docs/protocolo.md`](docs/protocolo.md) §13.

## Métricas

- **Latência** de inferência (ms), de ponta a ponta (pré + inferência + pós)
- **Throughput** (inferências/s)
- **Energia por inferência** (mJ, **estimada por software**) e eficiência normalizada
  (**inferências/joule** e **EDP**)
- **Acurácia**: mAP@50 / mAP@50-95

## Protocolo experimental (resumo)

> Detalhes e justificativas em [`docs/protocolo.md`](docs/protocolo.md) (protocolo congelado + emendas v1.1–v1.4).

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
data/raw/             # Dados brutos das execuções no SBC (CSV e JSON: E0, E1, E3b, E3c, E4, E5)
data/processed/       # Relatórios consolidados da Fase 4 (phase4_e6_report.json, phase4_energy_stats_summary.csv)
report/figures/       # Figuras científicas geradas em PNG (300 DPI) e PDF vetorial (fig1 a fig5)
video/                # Link/roteiro do vídeo de demonstração (3–5 min)
docs/                 # Proposta, protocolo, experimentação, dataset, roteiro_implementacao.qmd, scripts.md, plano de ação
docs/projeto/         # Projeto em LaTeX (main.tex → main.pdf)
docs/arquitetura/     # Diagramas de arquitetura (C4, DFD, casos de uso, sequência)
.opencode/skills/     # Skill do agente com a base de conhecimento do projeto
```

## Resultados Experimentais no SBC (Raspberry Pi Zero 2 W — Protocolo v1.3)

Todos os experimentos do Protocolo v1.3 no **Raspberry Pi Zero 2 W** (Cortex-A53 @ 1,0 GHz, governador `performance`, runtime `ai-edge-litert`, resolução $416\times 416$, 3 blocos independentes $\times$ 10 execuções válidas = 30 amostras + 10 *warmups* descartados/bloco) foram executados e auditados:

| Configuração ($416\times 416$) | Mediana $\tilde{t}_{\text{pre}}$ | Mediana $\tilde{t}_{\text{inf}}$ (IQR) | Mediana $\tilde{t}_{\text{post}}$ | Mediana $\tilde{t}_{\text{total}}$ | Throughput ($\text{inf/s}$ / $\text{total/s}$) | Energia $E_{\text{inf}}$ [$P_{\min}\text{–}P_{\max}$] | Eficiência $\eta$ | $\text{EDP}$ ($\text{mJ}\cdot\text{ms}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **INT8 — 4 Threads (`E3c`/`E3b`)** | 33,18 ms | **109,52 ms** (0,25 ms) | 6,05 ms | **148,76 ms** | **9,09** / 6,70 FPS | **253,13 mJ** $[203,61\text{–}319,17]$ | **3,95 inf/J** | **27.871** |
| **INT8 — 2 Threads (`E3b`)** | 33,17 ms | **162,77 ms** (0,14 ms) | 6,05 ms | **202,00 ms** | **6,14** / 4,95 FPS | **268,64 mJ** $[211,65\text{–}341,90]$ | **3,72 inf/J** | **43.736** |
| **INT8 — 1 Thread (`E3b`)** | 33,14 ms | **270,64 ms** (0,24 ms) | 6,02 ms | **309,89 ms** | **3,69** / 3,23 FPS | **270,68 mJ** $[203,01\text{–}351,88]$ | **3,69 inf/J** | **73.267** |
| **Float32 — 4 Threads (`E3c`)** | 34,54 ms | **236,60 ms** (0,71 ms) | 8,52 ms | **279,65 ms** | **4,21** / 3,57 FPS | **545,94 mJ** $[439,13\text{–}688,36]$ | **1,83 inf/J** | **129.600** |

- **Lei de Amdahl (`E3b`)**: Speedup $S(2) = 1{,}6628\times$ ($\text{IC}_{95\%} [1{,}6621;\,1{,}6635]$) e $S(4) = 2{,}4712\times$ ($\text{IC}_{95\%} [2{,}4693;\,2{,}4727]$, $p = 3{,}02\times 10^{-11}$, $\delta = -1{,}0$), com fração paralelizável $p = 79{,}38\%$ e fração serial $1-p = 20{,}62\%$. Demonstra o efeito *Race-to-Idle* (4 núcleos reduzem o EDP em $2{,}63\times$ vs. 1 núcleo).
- **Ganho da Quantização INT8 vs. FP32 (`E3c`)**: Redução de $3{,}01\times$ em Flash ($1{,}158\,\text{MiB}$ vs. $3{,}490\,\text{MiB}$), redução de RAM de pico ($60{,}60\,\text{MiB}$ vs. $74{,}23\,\text{MiB}$), speedup de $2{,}1604\times$ em $t_{\text{inf}}$ e redução de $2{,}16\times$ na energia por inferência.
- **Estabilidade Contínua de 60 s (`E4`)**: 399 inferências ininterruptas (`delay=0`), mediana $\tilde{t}_{\text{inf}} = 109{,}57\,\text{ms}$, temperatura máxima $71{,}4\,^\circ\text{C}$ ($< 80\,^\circ\text{C}$) e `throttled=0x0`.
- **Condição de *Break-Even* Energético SBC vs. MCU (`E6`)**: No cenário nominal ($P_{\text{SBC}} = 2{,}30\,\text{W}$, $P_{\text{MCU}} = 0{,}224\,\text{W} \Rightarrow 10{,}27\times$), $E_{\text{SBC}} \ge E_{\text{MCU}}$ se $t_{\text{MCU}} \le 1124{,}5\,\text{ms}$; no pior caso com *overhead* de placa da MCU ($1{,}85\,\text{W} / 0{,}474\,\text{W} = 3{,}90\times$), se $t_{\text{MCU}} \le 427{,}4\,\text{ms}$.

## Reprodutibilidade

1. **Execução completa da bateria v1.3 no Raspberry Pi Zero 2 W**:
   ```bash
   bash raspberry/pi_zero_2w/setup_rpi.sh
   python3 raspberry/pi_zero_2w/run_all_v13_experiments.py
   ```
2. **Análise energética, estatística robusta (`E6`) e geração das 5 figuras científicas**:
   ```bash
   python3 telemetry/energy_and_stats_v13.py
   python3 telemetry/plot_phase4_v13.py
   ```

## Entregáveis

- [x] Repositório versionado com licença
- [x] README de reprodutibilidade e roteiro operacional ([`docs/roteiro_implementacao.qmd`](docs/roteiro_implementacao.qmd))
- [x] Dados brutos no SBC (`data/raw/rpi_*.csv` e `*.json`) e relatórios processados (`data/processed/phase4_*`)
- [x] Figuras científicas em alta resolução (`report/figures/fig1..fig5` em `.png` e `.pdf`)
- [ ] Execução física final na MCU e vídeo de demonstração (3–5 min) nas duas plataformas
- [ ] Relatório técnico conclusivo final com tabelas e gráficos das duas plataformas
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
