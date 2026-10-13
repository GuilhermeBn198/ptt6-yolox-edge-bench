# Protocolo Experimental — PTT6 (YOLOX-Nano int8: ESP32-S3 vs Raspberry Pi Zero 2 W)

> **Status: CONGELADO v1.0** (sujeito a revisão do orientador/colega). Decisões finais em **DECISÃO**.
>
> Fundamentação e fontes: [`pesquisa-protocolo.md`](pesquisa-protocolo.md). Proposta original:
> [`proposta/PTT6_Filipe_Guilherme_assinado.pdf`](proposta/PTT6_Filipe_Guilherme_assinado.pdf).

- **Versão:** 1.0 (congelada)
- **Autores:** Filipe Matias, Guilherme Lucas — **Orientador:** Herbert Oliveira Rocha
- **Última atualização:** 2026-10-08

---

## Resumo do plano

1. Medir, de forma reprodutível, a inferência do YOLOX-Nano int8 em duas plataformas de borda — ESP32-S3 (MCU) e Raspberry Pi Zero 2 W (SBC) — para detecção de EPIs.
2. Tarefa é detecção (bounding boxes), não classificação; acurácia entra como mAP@50 e mAP@50-95.
3. Modelo apenas quantizado, sem re-treino; mesmo runtime TFLite int8 nas duas plataformas, para isolar o efeito do hardware.
4. Artefato único: YOLOX-ti-lite convertido para TFLite int8, calibrado com 750 imagens, com hash registrado.
5. Começamos com um único EPI, o colete; conjuntos candidatos CHV e Ultralytics Construction-PPE (o SH17 tem pouquíssimas amostras de colete).
6. Medimos latência de ponta a ponta (pré + inferência + pós), throughput e energia por inferência (total da janela, incluindo o consumo parado).
7. Eficiência comparada por inferências/joule e por EDP (energia × latência), com H1 e H0 formalizadas.
8. Energia medida com INA219 na taxa máxima (~1,9 kHz), janela delimitada por GPIO e integração de potência; o erro do INA219 fica declarado como limitação.
9. Um piloto define o número de repetições (piso 30, referência até ~240) e o warm-up pelo próprio dado; duas bordas de GPIO separam as três fases.
10. Estatística: mediana + IQR, teste de Mann-Whitney U com tamanho de efeito e IC por bootstrap; não removemos outliers (opção A), só checamos sensibilidade.
11. Configuração travada: ESP32-S3 a 240 MHz sem power management e Raspberry Pi com governor performance a 1 GHz, com ambiente e temperatura registrados.
12. Testes com câmera ao vivo; resolução, formato e número de buffers iguais nas duas pontas, e a diferença de pipeline de câmera declarada como ameaça à validade.
13. Entregáveis: repositório com licença, README de reprodutibilidade, CSV bruto das execuções, vídeo de 3–5 min e relatório técnico conclusivo.
14. Prazo de 15 semanas; depois, publicar o dataset no Zenodo (DOI) e submeter a um evento regional.
15. Título permanece provisório.

---

## 1. Objetivo, pergunta e hipótese

- **Tarefa:** **detecção** de EPIs (saída com bounding boxes), não classificação de imagem.
  A acurácia é medida por **mAP**. **(DECISÃO)**
- **Objetivo geral:** comparar desempenho e eficiência energética da inferência de um modelo
  YOLOX-Nano int8 em uma MCU (ESP32-S3) e um SBC (Raspberry Pi Zero 2 W).
- **Pergunta:** como se comparam latência, throughput e energia por inferência entre as duas
  plataformas ao executar o mesmo YOLOX-Nano int8, detectando EPIs, em câmera ao vivo?
- **H1:** o SBC tem throughput maior, mas energia por inferência igual ou superior.
- **H0:** não há diferença nas métricas entre as plataformas. **(DECISÃO.)**
- **Eficiência:** inferências/joule **e** EDP. **(DECISÃO.)**
- **Prazo:** cronograma de **15 semanas**.

## 2. Plataformas e entrada

| | MCU | SBC |
|---|---|---|
| Modelo | ESP32-S3 (LX7 dual-core, até 240 MHz, PSRAM) | Raspberry Pi Zero 2 W (Cortex-A53 quad-core @1 GHz, 512 MB) |
| Entrada | câmera ao vivo (esp32-camera) | câmera ao vivo (CSI) |
| Medição | INA219 inline + fonte regulada | INA219 inline + fonte regulada |

- **Entrada (DECISÃO):** testes com **câmera ao vivo**; dataset estático para treino/calibração/validação.
  As pipelines de câmera diferem (esp32-camera vs V4L2/libcamera) → **ameaça à validade**: fixar
  resolução, formato e número de buffers iguais.
- **Relógio (DECISÃO):** ESP32-S3 travado em **240 MHz** com DFS/light-sleep desabilitados;
  RPi com governor **`performance`** a **1 GHz**. Registrar `sdkconfig` e `config.txt`.

## 3. Modelo e runtime

- **Modelo:** YOLOX-Nano int8 (0,91 M params, 1,08 GFLOPs, 416×416). **(DECISÃO: só quantização.)**
- **Runtime (DECISÃO):** **mesmo nas duas plataformas — TFLite int8** — para comparação direta.
- **Artefato (DECISÃO):** um único `.tflite` int8 por **YOLOX-ti-lite → TFLite** (`onnx2tf`), usado
  nas duas pontas; registrar **hash**.
- **Calibração int8 (DECISÃO):** **750 imagens** representativas do conjunto de treino,
  estratificadas por classe, sem vazar validação.

## 4. Dataset de EPIs

- **Base (DECISÃO):** começar por **um EPI — o colete (`safety-vest`)**; expandir depois (issue #20).
- **Conjuntos registrados para uso:** **CHV** (1.330 imgs, 6 classes, uso público) e
  **Ultralytics Construction-PPE** (1.416 imgs, 11 classes, inclui classes de "EPI ausente";
  AGPL-3.0) como candidatos diretos para o colete. Complementos avaliados: **SH17** (8.099 imgs,
  17 classes — **atenção: só ~45 imgs de `safety-vest`**), **SFCHD** (12.373 imgs, cenário real),
  SHEL5K, SHEL/SHWD, Pictor-PPE, GDUT-HWD.
- **Pré-processamento (DECISÃO):** idêntico nas duas plataformas; documentado em `docs/dataset.md`.

## 5. Métricas e definições

| Métrica | Definição | Janela |
|---|---|---|
| Latência | tempo de uma inferência de ponta a ponta | **pré + inferência + pós (DECISÃO)** |
| Throughput | inferências por segundo | idem |
| Energia por inferência | `E = ∫ P(t)·dt` na janela | **total, incluindo idle (DECISÃO)** |
| Eficiência | inferências/joule | derivada |
| EDP | energia × latência | derivada |
| Acurácia | mAP@50 / mAP@50-95 | validação |

- Mesmo medindo o total, **reportar as 3 fases separadas** (pré/inferência/pós) para permitir reanálise.

## 6. Protocolo de medição

- **Energia (DECISÃO + RECOMENDADO):** manter o **INA219**. Operar na **taxa máxima** (12 bits,
  1 amostra ≈ 532 µs ≈ **1,9 kHz**, não 10 Hz); delimitar a janela por **GPIO** e **integrar a
  potência** no tempo (`E = ∫ P(t)·dt`); **calibrar** contra carga conhecida. **Limitação a
  documentar:** o INA219 mede tensão e corrente em instantes diferentes (até ~68 ms no modo 128
  médias) e não resolve inferências curtas — o erro será declarado no relatório. Se o erro
  comprometer a comparação, avaliar upgrade para **Nordic PPK2** (~100 US$, 100 ksps) ou Joulescope.
- **Cronometragem (DECISÃO):** **duas bordas de GPIO** (disparo duplo) para delimitar a janela e
  separar as 3 fases.
- **Repetições (DECISÃO):** rodar **piloto**, estimar o **CV** e calcular N para IC de 95%; **piso
  de 30** e referência de teto **~240** (CV alto/outliers). N pode diferir por plataforma.
- **Warm-up (DECISÃO):** definido **pelo dado** no piloto — descartar até a variação relativa
  entre janelas consecutivas cair abaixo de um limiar (ex.: < 2%), com piso de segurança (ex.: 5).
  Não usar "descartar 5" fixo (sem lastro na literatura).

## 7. Variáveis

- **Independente:** plataforma.
- **Dependentes:** latência (ms), throughput (inf/s), energia/inferência (mJ).
- **Controladas (registrar):** clock/governor, PM/light-sleep, versões (firmware, SDK, runtime,
  compilador), hash do modelo, tensão de alimentação, temperatura, formato/nº de frame buffers da
  câmera, imagens/ambiente.

## 8. Análise estatística

- **Reportar:** mediana + IQR **e** média ± dp; IC 95%.
- **Teste (DECISÃO):** **Mann-Whitney U** (robusto a não normalidade/outliers) + **tamanho de efeito**
  (Cliff's delta / rank-biserial) + **IC por bootstrap** da diferença. (ANOVA/Tukey não se aplica a
  dois grupos.) Welch t como verificação de robustez se a normalidade for plausível.
- **Teste de H1:** throughput(SBC) > throughput(MCU) **e** energia(SBC) ≥ energia(MCU).
- **Outliers (DECISÃO):** **opção A — não remover**. Análise principal com todas as execuções válidas
  usando estatística robusta (mediana, Mann-Whitney). A remoção por Tukey (>1,5×IQR) fica apenas como
  **verificação de sensibilidade**, reportando N original e removido.

## 9. Ameaças à validade e mitigação

| Ameaça | Mitigação |
|---|---|
| Throttling térmico no Zero 2 W (80/85 °C) | governor `performance`, heatsink, pausas, registrar temp. |
| DFS/light-sleep no ESP32-S3 | desabilitar PM ou travar `ESP_PM_CPU_FREQ_MAX`; registrar `sdkconfig` |
| Pipelines de câmera diferentes | declarar; fixar resolução/formato/nº de buffers |
| PSRAM lenta / cache | fixar layout do modelo; registrar uso de PSRAM vs SRAM |
| Erro sistemático do INA219 (V/I não simultâneos) | calibrar; quantificar; declarar como limitação |
| Deriva de frequência/autocorrelação | múltiplas execuções; travar frequência; análise temporal |
| Sem baseline publicado de YOLOX-Nano nas duas plataformas | ver §12 (D16) |

## 10. Plano B (contingência)

- **Medidor indisponível/defeituoso:** estimar energia = potência do datasheet × tempo medido,
  declarando a limitação.
- **Modelo não performa em algum device:** usar outras versões quantizadas do YOLO, mantendo o protocolo.

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
| D3 | Entrada | Câmera ao vivo | 2026-10-08 |
| D4 | Câmera | Entra no escopo | 2026-10-08 |
| D5 | Janela de latência | Tudo (pré + inferência + pós) | 2026-10-08 |
| D6 | Energia | Total da janela (inclui idle) | 2026-10-08 |
| D7 | Instrumento | Manter INA219 (taxa máx, GPIO, calibrar; erro → limitação) | 2026-10-08 |
| D8 | Runtime | Mesmo nas duas plataformas (TFLite int8) | 2026-10-08 |
| D9 | Modelo | Somente quantização | 2026-10-08 |
| D10 | Classes | Começar por colete (1 classe); expandir via issue #20 | 2026-10-08 |
| D11 | Pré-processamento | Idêntico nas duas plataformas | 2026-10-08 |
| D12 | Eficiência | inf/J + EDP | 2026-10-08 |
| D13 | H0 | Adotada | 2026-10-08 |
| D14 | Título | Manter provisório | 2026-10-08 |
| D15 | Ambiente | Registrar (temp, tensão, shunt, buffers) | 2026-10-08 |
| D16 | Ausência de baseline | Rigor de reprodutibilidade + citar modelos análogos como contexto + dataset publicado como contribuição principal | 2026-10-08 |
| D17 | Calibração int8 | 750 imagens estratificadas | 2026-10-08 |
| D18 | Outliers | Opção A (não remover; robusto) com Tukey como sensibilidade | 2026-10-08 |
| D19 | Repetições | Piloto → CV → N; piso 30, teto de referência ~240 | 2026-10-08 |
| D20 | Teste estatístico | Mann-Whitney U + tamanho de efeito + IC bootstrap | 2026-10-08 |
| D21 | Clock/governor | ESP32-S3 240 MHz sem PM; RPi `performance` 1 GHz | 2026-10-08 |
| D22 | Warm-up | Definido por dado no piloto (variação < 2%, piso 5) | 2026-10-08 |

### 12.2 Em aberto

Nenhuma pendência: o protocolo está **congelado (v1.0)**. Alterações futuras exigem nova versão documentada.

---

## Referências

Fontes primárias detalhadas em [`pesquisa-protocolo.md`](pesquisa-protocolo.md). Principais:
MLPerf Tiny (arXiv:2106.07597), Noronha et al. 2025 (arXiv:2505.15622), Maricq et al. OSDI'18,
NIST TN 1830, datasheets INA219/ESP32-S3/RPi Zero 2 W, repositórios Espressif/YOLOX/NCNN,
datasets CHV/SH17/SFCHD/Construction-PPE.
