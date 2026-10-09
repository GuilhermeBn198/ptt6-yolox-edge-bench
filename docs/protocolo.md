# Protocolo Experimental — PTT6 (YOLOX-Nano int8: ESP32-S3 vs Raspberry Pi Zero 2 W)

> **Status: RASCUNHO — NÃO CONGELADO.** Este documento é discutido pela dupla antes de virar a
> base da coleta. Itens marcados com **⚠️ DECIDIR** são decisões abertas; itens com **✅ PROPOSTO**
> já têm recomendação e só precisam de aval.
>
> Fundamentação e fontes: [`pesquisa-protocolo.md`](pesquisa-protocolo.md). Proposta original:
> [`proposta/PTT6_Filipe_Guilherme_assinado.pdf`](proposta/PTT6_Filipe_Guilherme_assinado.pdf).

- **Versão:** 0.1 (rascunho)
- **Autores:** Filipe Matias, Guilherme Lucas — **Orientador:** Herbert Oliveira Rocha
- **Data:** 2026-10-08

---

## 1. Objetivo, pergunta e hipótese

- **Objetivo geral:** comparar desempenho e eficiência energética da inferência de um modelo
  YOLOX-Nano int8 em uma MCU (ESP32-S3) e um SBC (Raspberry Pi Zero 2 W).
- **Pergunta:** como se comparam latência, throughput e energia por inferência entre as duas
  plataformas ao executar o mesmo YOLOX-Nano int8 para classificação de EPIs?
- **H1:** o SBC tem throughput maior, mas energia por inferência igual ou superior (consumo
  estático mais alto).
- **Hipótese nula (H0):** não há diferença nas métricas entre as plataformas.
- **Métricas de eficiência:** inferências/joule; **✅ PROPOSTO** reportar também **EDP
  (Energy-Delay Product)**, que penaliza latência e energia em conjunto (Noronha et al., 2025).

## 2. Plataformas

| | MCU | SBC |
|---|---|---|
| Modelo | **ESP32-S3** (Xtensa LX7 dual-core, até 240 MHz, PSRAM) | **Raspberry Pi Zero 2 W** (Cortex-A53 quad-core @1 GHz, 512 MB, sem NPU) |
| Alimentação | fonte regulada + INA219 inline | fonte regulada + INA219 inline |
| Relógio | **⚠️ DECIDIR** travar 240 MHz e desabilitar PM/light-sleep | **⚠️ DECIDIR** governor `performance` travado em 1 GHz |
| Térmica | limite 65 °C nas variantes Octal PSRAM | throttling a 80 °C / máximo 85 °C |

## 3. Modelo e runtime

- **Modelo:** YOLOX-Nano quantizado int8 (0,91 M params, 1,08 GFLOPs, 416×416, 25,3% AP COCO).
- **Fonte do artefato:** **⚠️ DECIDIR** entre YOLOX-ti-lite→TFLite int8
  (`motokimura/yolox-ti-lite_tflite`) ou ONNX int8. Registrar hash do modelo.
- **Runtime por plataforma:**
  - ESP32-S3: **✅ PROPOSTO** esp-tflite-micro + **ESP-NN** (kernels int8); alternativa ESP-DL.
  - Zero 2 W: **⚠️ DECIDIR** NCNN (NEON, tende a ser o mais rápido) **vs** TFLite (mais
    compatível com o artefato YOLOX-ti-lite int8).
  - **⚠️ DECIDIR** e declarar explicitamente: comparar runtimes diferentes é uma ameaça à
    validade; se possível, manter o **mesmo runtime (TFLite)** nas duas plataformas.
- **Calibração int8:** **✅ PROPOSTO** ~200 imagens representativas de EPIs (faixa 100–500),
  sem vazar o conjunto de validação.

## 4. Dataset de EPIs

- **⚠️ DECIDIR** entre:
  - **CHV** — 1.330 imgs, 6 classes, licença "open for public use" → leve, cabe na MCU.
  - **SH17** — 8.099 imgs, 17 classes, licença research+comercial, split pronto → mais rico.
  - **SHEL5K** (5.000 imgs, 6 classes) / **CSOD-24** como complementos.
- **✅ PROPOSTO** foco em subconjunto de EPI (capacete, colete, máscara, óculos).
- **⚠️ DECIDIR** se o modelo será apenas **quantizado** (escopo mínimo) ou **re-treinado**.
- **Pré-processamento:** idêntico nas duas plataformas (mesma normalização/resize/ordem de canais);
  documentar em `docs/dataset.md`.

## 5. Métricas e definições

| Métrica | Definição | Janela |
|---|---|---|
| Latência | tempo de **uma** inferência | **⚠️ DECIDIR** só `invoke()` vs pré+inferência+pós |
| Throughput | inferências por segundo | mesma janela da latência |
| Energia por inferência | `E = ∫ P(t)·dt` na janela | mesma janela |
| Eficiência | inferências/joule | derivada |
| EDP | energia × latência | derivada (opcional) |

- **✅ PROPOSTO:** isolar e reportar as **três fases** (pré-inferência, inferência, pós-inferência),
  com disparo duplo por GPIO, para não atribuir custo de setup ao modelo (Noronha et al., 2025).
- **Energia estática/idle entra na conta?** **⚠️ DECIDIR** convenção: energia total da janela
  (como MLPerf Tiny) **vs** energia líquida subtraindo o piso idle. Documentar a escolha.

## 6. Protocolo de medição

- **Warm-up:** **⚠️ DECIDIR** entre descarte fixo (5) **vs** detecção de estabilização por dado
  (ex.: desvio relativo < x% entre janelas). A literatura **não** sustenta "5" como norma (MLPerf
  não descarta; steady-state pode não existir).
- **Repetições (N):** **⚠️ DECIDIR** após **piloto**. 30 pode bastar para CV baixo (<~4%), mas
  CV alto/outliers (típico de SBC) pode exigir até ~240 (Maricq et al., OSDI'18).
- **Instrumentação de energia:** **⚠️ DECIDIR**
  - **(a)** INA219 na taxa máxima (~1,9 kHz), delimitado por GPIO; **ou**
  - **(b)** migrar para **Joulescope JS110/JS220 (2 MSPS)** ou **Nordic PPK2 (100 ksps)**.
  - **Nota:** INA219 a **10 Hz é insuficiente** (100 ms/amostra) e mede V e I em instantes
    diferentes (até ~68 ms no modo 128 médias) — se mantido, quantificar o erro sistemático.
- **Cronometragem:** toggle de GPIO (borda de descida) sincronizando latência e energia com o
  mesmo instante; registrar jitter.
- **Controle de temperatura:** pausas de resfriamento entre baterias, ambiente refrigerado;
  registrar temperatura (interna, quando disponível).

## 7. Variáveis

- **Independente:** plataforma.
- **Dependentes:** latência (ms), throughput (inf/s), energia/inferência (mJ).
- **Controladas (registrar):** clock/governor, PM/light-sleep, versões (firmware, SDK, runtime,
  compilador), hash do modelo, tensão de alimentação, temperatura ambiente, formato/nº de frame
  buffers da câmera, images de entrada (idênticas).

## 8. Análise estatística

- **✅ PROPOSTO:** reportar **mediana + IQR** (robusto a outliers) **e** média±dp; **IC 95%**
  (t-Student). Comparar plataformas com teste apropriado (ex.: Mann-Whitney U ou ANOVA/Tukey).
- **Tratamento de outliers:** **⚠️ DECIDIR** ex-ante (ex.: regra IQR de Tukey), remoção apenas
  transparente, reportando o N original e o removido.
- **Teste de H1:** verificar simultaneamente throughput(SBC) > throughput(MCU) **e**
  energia(SBC) ≥ energia(MCU).

## 9. Ameaças à validade e mitigação

| Ameaça | Mitigação |
|---|---|
| Throttling térmico no Zero 2 W (80/85 °C) | governor `performance`, heatsink, pausas, registrar temp. |
| DFS/light-sleep no ESP32-S3 | desabilitar PM ou travar `ESP_PM_CPU_FREQ_MAX`; registrar `sdkconfig` |
| PSRAM lenta / cache | fixar layout do modelo; registrar uso de PSRAM vs SRAM |
| Runtime diferente entre plataformas | declarar; se possível, mesmo runtime |
| Erro sistemático do instrumento (V/I não simultâneos) | calibrar; quantificar; considerar Joulescope/PPK2 |
| Deriva de frequência/autocorrelação | múltiplas execuções; travar frequência; análise temporal |

## 10. Plano B (contingência)

- **Medidor indisponível/defeituoso:** estimar energia = potência do datasheet × tempo medido,
  **declarando a limitação**.
- **Modelo não performa em algum device:** usar outras versões quantizadas do YOLO, mantendo o
  protocolo.

## 11. Riscos

| Risco | Prob. | Impacto | Plano B |
|---|---|---|---|
| Medidor de potência físico indisponível | média | alto | estimativa por datasheet × tempo |
| Modelo não performa em um dispositivo | média | alto | outras versões quantizadas do YOLO |

## 12. Registro de decisões (preencher ao congelar)

| # | Decisão | Valor escolhido | Data | Responsável |
|---|---|---|---|---|
| D1 | Janela de latência (só inferência vs 3 fases) | | | |
| D2 | Energia estática entra? | | | |
| D3 | Instrumento/taxa de amostragem | | | |
| D4 | Warm-up (descarte 5 vs detecção) | | | |
| D5 | N repetições | | | |
| D6 | Regra de outliers | | | |
| D7 | Runtime por plataforma | | | |
| D8 | Dataset de EPIs | | | |
| D9 | Re-treinar ou só quantizar | | | |
| D10 | Frequência/governor/PM travados | | | |

---

## Referências

Fontes primárias detalhadas em [`pesquisa-protocolo.md`](pesquisa-protocolo.md). Principais:
MLPerf Tiny (arXiv:2106.07597), Noronha et al. 2025 (arXiv:2505.15622), Maricq et al. OSDI'18,
NIST TN 1830, datasheets INA219/ESP32-S3/RPi Zero 2 W, repositórios Espressif/YOLOX/NCNN.
