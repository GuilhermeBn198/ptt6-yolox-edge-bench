# Protocolo Experimental — PTT6 (YOLOX-Nano int8: ESP32-S3 vs Raspberry Pi Zero 2 W)

> **Status: RASCUNHO v0.2 — ainda NÃO CONGELADO.** Decisões já tomadas pela dupla estão em
> **DECISÃO**; recomendações técnicas que aguardam OK estão em **RECOMENDADO**; discussões em
> aberto estão em **EM ABERTO**.
>
> Fundamentação e fontes: [`pesquisa-protocolo.md`](pesquisa-protocolo.md). Proposta original:
> [`proposta/PTT6_Filipe_Guilherme_assinado.pdf`](proposta/PTT6_Filipe_Guilherme_assinado.pdf).

- **Versão:** 0.2
- **Autores:** Filipe Matias, Guilherme Lucas — **Orientador:** Herbert Oliveira Rocha
- **Última atualização:** 2026-10-08

---

## 1. Objetivo, pergunta e hipótese

- **Tarefa:** **detecção** de EPIs (saída com bounding boxes), não classificação de imagem.
  O alvo é identificar os EPIs e o trabalhador; a obtenção de uma métrica de acurácia passa a
  incluir mAP, e não só acurácia de classe. **(DECISÃO)**
- **Objetivo geral:** comparar desempenho e eficiência energética da inferência de um modelo
  YOLOX-Nano int8 em uma MCU (ESP32-S3) e um SBC (Raspberry Pi Zero 2 W).
- **Pergunta:** como se comparam latência, throughput e energia por inferência entre as duas
  plataformas ao executar o mesmo YOLOX-Nano int8, detectando EPIs, em câmera ao vivo?
- **H1:** o SBC tem throughput maior, mas energia por inferência igual ou superior.
- **H0:** não há diferença nas métricas entre as plataformas. **(DECISÃO: adotar H0.)**
- **Métrica de eficiência:** inferências/joule **e** EDP (Energy-Delay Product). **(DECISÃO.)**
- **Prazo:** cronograma de **15 semanas** (a capa da proposta diz 11; prevalece o cronograma).

## 2. Plataformas e entrada

| | MCU | SBC |
|---|---|---|
| Modelo | ESP32-S3 (LX7 dual-core, até 240 MHz, PSRAM) | Raspberry Pi Zero 2 W (Cortex-A53 quad-core @1 GHz, 512 MB) |
| Entrada | **câmera ao vivo** (esp32-camera) | **câmera ao vivo** (CSI) |
| Medição | INA219 inline + fonte regulada | INA219 inline + fonte regulada |

- **Entrada (DECISÃO):** os testes usam **câmera ao vivo**; o dataset estático serve para treino,
  calibração e validação. Como as pipelines de câmera são diferentes nas duas plataformas
  (esp32-camera vs V4L2/libcamera), isso é registrado como **ameaça à validade**.
- **Relógio (RECOMENDADO):** ESP32-S3 travado em **240 MHz** com DFS/light-sleep desabilitados;
  RPi com governor **`performance`** a **1 GHz**. Registrar `sdkconfig` e `config.txt`.
- **Térmica:** registrar temperatura; heatsink e pausas de resfriamento.

## 3. Modelo e runtime

- **Modelo:** YOLOX-Nano quantizado int8 (0,91 M params, 1,08 GFLOPs, 416×416). **(DECISÃO: só
  quantização, sem re-treino.)**
- **Runtime (DECISÃO):** **o mesmo nas duas plataformas — TensorFlow Lite (int8)** — para
  comparação direta. (Alternativa descartada: NCNN no SBC, que trocaria hardware+runtime ao mesmo
  tempo e enfraqueceria H1.)
- **Artefato (RECOMENDADO):** um único `.tflite` int8 obtido por **YOLOX-ti-lite → TFLite**
  (`onnx2tf`), usado nas duas pontas; registrar **hash** do arquivo no repositório.
- **Calibração int8 (RECOMENDADO):** **~500 imagens** representativas do conjunto de treino
  (faixa da literatura: 100–500; alguns trabalhos usam 1.000), estratificadas por classe, sem
  vazar o conjunto de validação.

## 4. Dataset de EPIs

- **Base (DECISÃO):** começar por **um único EPI — o colete (`safety-vest`)** — e expandir depois.
  A expansão de classes será acompanhada por issues no GitHub (fora do escopo mínimo).
- **Datasets a considerar (EM ABERTO — verificar padrão de indústria):**
  - **CHV** — 1.330 imgs, 6 classes, uso público. **Boa opção para começar com colete.**
  - **SH17** — 8.099 imgs, 17 classes, licença research+comercial. **Atenção:** a classe
    `safety-vest` tem muito poucas amostras (~45 imagens) — ruim para um estudo só de colete.
  - **Ultralytics Construction-PPE** — 1.416 imgs, 11 classes, inclui classes de "EPI ausente"
    (`no_helmet` etc.); licença **AGPL-3.0**. Relevante por ser ferramenta de referência da área.
  - **SFCHD** — 12.373 imgs, 7 classes, imagens de câmeras de vigilância em ambiente real.
  - Complementos: SHEL5K, SHEL/SHWD, Pictor-PPE, GDUT-HWD.
- **Pré-processamento (DECISÃO):** idêntico nas duas plataformas (resize, normalização, ordem de
  canais), documentado em `docs/dataset.md`.

## 5. Métricas e definições

| Métrica | Definição | Janela |
|---|---|---|
| Latência | tempo de uma inferência de ponta a ponta | **pré + inferência + pós (DECISÃO: medir tudo)** |
| Throughput | inferências por segundo | idem |
| Energia por inferência | `E = ∫ P(t)·dt` na janela | **total da janela, incluindo idle (DECISÃO)** |
| Eficiência | inferências/joule | derivada |
| EDP | energia × latência | derivada (DECISÃO: incluir) |
| Acurácia | mAP@50 / mAP@50-95 (tarefa de detecção) | validação |

- **DECISÃO:** medir a latência **de ponta a ponta (tudo)** e a energia **total da janela**,
  incluindo o consumo parado. Mesmo assim, **reportar as 3 fases separadas** (pré-inferência,
  inferência, pós-inferência) para permitir reanálise — o número principal é o total.

## 6. Protocolo de medição

- **Warm-up (RECOMENDADO):** definido **pelo dado**, não fixo. Rodar um piloto e descartar até a
  variação relativa entre janelas consecutivas cair abaixo de um limiar (ex.: < 2%), com piso de
  segurança (ex.: 5 execuções). Justificar com o piloto. (O "descartar 5" fixo não tem lastro na
  literatura.)
- **Repetições (RECOMENDADO):** rodar um **piloto** em cada plataforma, estimar o **CV** e calcular
  o N necessário para um IC de 95% com margem alvo; **piso de 30**. N pode diferir por plataforma
  (o SBC tende a ter mais variabilidade).
- **Instrumentação de energia (DECISÃO: manter INA219):**
  - Operar o INA219 na **taxa máxima** (12 bits, 1 amostra ≈ 532 µs ≈ 1,9 kHz), não a 10 Hz.
  - Delimitar a janela por **GPIO** e integrar a potência no tempo da janela.
  - Calibrar contra uma carga conhecida.
  - **Limitação conhecida:** o INA219 mede tensão e corrente em **instantes diferentes** (até ~68 ms
    no modo 128 médias) e a 10 Hz não resolve inferências curtas. Se o erro for relevante, será
    **declarado como limitação** no relatório (Plano B já previsto). Se necessário no futuro,
    considerar PPK2/Joulescope (2 MSPS).
- **Cronometragem (RECOMENDADO):** **duas bordas de GPIO** (disparo duplo) para delimitar a janela
  e, se possível, separar as 3 fases.

## 7. Variáveis

- **Independente:** plataforma.
- **Dependentes:** latência (ms), throughput (inf/s), energia/inferência (mJ).
- **Controladas (registrar):** clock/governor, PM/light-sleep, versões (firmware, SDK, runtime,
  compilador), hash do modelo, tensão de alimentação, temperatura, formato/nº de frame buffers da
  câmera, imagens/ambiente de entrada.

## 8. Análise estatística

- **Reportar (DECISÃO):** mediana + IQR **e** média ± dp; IC 95%.
- **Teste (RECOMENDADO):** com exatamente **duas** plataformas, usar **Mann-Whitney U** (robusto a
  não normalidade/outliers) como principal, com **tamanho de efeito** (Cliff's delta / rank-biserial)
  e **IC por bootstrap** da diferença. ANOVA/Tukey só se houvesse >2 grupos (não é o caso).
  Welch t-test como verificação de robustez se a normalidade for plausível.
- **Teste de H1:** verificar simultaneamente throughput(SBC) > throughput(MCU) **e**
  energia(SBC) ≥ energia(MCU).
- **Outliers (EM ABERTO):** discutir política (ver seção 12). Recomendação inicial: reportar os
  dados brutos; análise principal com todas as execuções válidas (pós-warm-up); verificação de
  sensibilidade removendo >1,5×IQR (Tukey), com justificativa e reportando N original e removido.

## 9. Ameaças à validade e mitigação

| Ameaça | Mitigação |
|---|---|
| Throttling térmico no Zero 2 W (80/85 °C) | governor `performance`, heatsink, pausas, registrar temp. |
| DFS/light-sleep no ESP32-S3 | desabilitar PM ou travar `ESP_PM_CPU_FREQ_MAX`; registrar `sdkconfig` |
| **Pipelines de câmera diferentes** (esp32-camera vs V4L2) | declarar; fixar resolução/formato/nº de buffers |
| PSRAM lenta / cache | fixar layout do modelo; registrar uso de PSRAM vs SRAM |
| Erro sistemático do INA219 (V/I não simultâneos) | calibrar; quantificar; declarar como limitação |
| Deriva de frequência/autocorrelação | múltiplas execuções; travar frequência; análise temporal |
| Sem baseline publicado de YOLOX-Nano nas duas plataformas | reportar como medição primeira; reforçar reprodutibilidade |

## 10. Plano B (contingência)

- **Medidor indisponível/defeituoso:** estimar energia = potência do datasheet × tempo medido,
  declarando a limitação.
- **Modelo não performa em algum device:** usar outras versões quantizadas do YOLO, mantendo o
  protocolo.

## 11. Riscos

| Risco | Prob. | Impacto | Plano B |
|---|---|---|---|
| Medidor de potência físico indisponível | média | alto | estimativa por datasheet × tempo |
| Modelo não performa em um dispositivo | média | alto | outras versões quantizadas do YOLO |

## 12. Registro de decisões e pendências

### 12.1 Decisões tomadas

| # | Decisão | Valor | Data |
|---|---|---|---|
| D1 | Tarefa | Detecção de EPIs (bounding boxes) | 2026-10-08 |
| D2 | Prazo | 15 semanas | 2026-10-08 |
| D3 | Entrada | Câmera ao vivo (dataset estático só p/ treino/calibração) | 2026-10-08 |
| D4 | Câmera | Entra no escopo | 2026-10-08 |
| D5 | Janela de latência | Tudo (pré + inferência + pós) | 2026-10-08 |
| D6 | Energia | Total da janela (inclui idle) | 2026-10-08 |
| D7 | Instrumento | Manter INA219; erro → limitação | 2026-10-08 |
| D8 | Runtime | Mesmo nas duas plataformas (TFLite int8) | 2026-10-08 |
| D9 | Modelo | Somente quantização (sem re-treino) | 2026-10-08 |
| D10 | Classes | Começar por colete (1 classe); expandir via issues | 2026-10-08 |
| D11 | Pré-processamento | Idêntico nas duas plataformas | 2026-10-08 |
| D12 | Métrica de eficiência | inf/J + EDP | 2026-10-08 |
| D13 | H0 | Adotada | 2026-10-08 |
| D14 | Título | Manter provisório | 2026-10-08 |
| D15 | Ambiente | Registrar (temp, tensão, shunt, buffers) | 2026-10-08 |

### 12.2 Recomendações técnicas (aguardando OK)

| # | Item | Recomendação |
|---|---|---|
| R1 | Warm-up | Definido por dado + piso (não fixo em 5) |
| R2 | N repetições | Piloto → CV → N; piso 30 |
| R3 | Cronometragem | Duas bordas de GPIO (separar fases) |
| R4 | Artefato int8 | YOLOX-ti-lite → TFLite, mesmo arquivo nas duas; hash |
| R5 | Calibração | ~500 imagens estratificadas |
| R6 | Teste estatístico | Mann-Whitney U + tamanho de efeito + IC bootstrap |
| R7 | Clock/governor | ESP32-S3 240 MHz sem PM; RPi `performance` 1 GHz |

### 12.3 Em aberto

| # | Item | Situação |
|---|---|---|
| P1 | Melhor abordagem para medir/estimar energia (proxy) | precisa de mais definição |
| P2 | Política de outliers | discutir |
| P3 | Como reportar a ausência de baseline publicado | definir redação |
| P4 | Dataset de EPI padrão de indústria | verificar alternativas (ver §4) |

---

## Referências

Fontes primárias detalhadas em [`pesquisa-protocolo.md`](pesquisa-protocolo.md). Principais:
MLPerf Tiny (arXiv:2106.07597), Noronha et al. 2025 (arXiv:2505.15622), Maricq et al. OSDI'18,
NIST TN 1830, datasheets INA219/ESP32-S3/RPi Zero 2 W, repositórios Espressif/YOLOX/NCNN,
datasets CHV/SH17/SFCHD/Construction-PPE.
