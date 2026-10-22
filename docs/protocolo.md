# Protocolo Experimental — PTT6 (YOLOX-Nano int8: ESP32 (LX6) vs Raspberry Pi Zero 2 W)

> **Status: v1.5** — emendas sobre o protocolo **v1.0 congelado**: **v1.1** (plataforma MCU, §13),
> **v1.2** (energia estimada por software + execução em fases, §14–§15), **v1.3** (dataset, treino,
> cronometragem, modelo de potência e desenho experimental, §16 + [`experimentacao.md`](experimentacao.md)) e **v1.4**
> (câmera ao vivo vira **extra**; validação e medição por **ingestão direta de imagens** no modelo, §17) e **v1.5**
> (métricas com **acurácia como condição de validade**, E0 medido, protocolo de treino e de quantização, §18 +
> [`revisao/README.md`](revisao/README.md)). A estatística permanece congelada; as métricas foram ampliadas na v1.5. Decisões finais em **DECISÃO**.
>
> Fundamentação e fontes: [`pesquisa-protocolo.md`](pesquisa-protocolo.md). Proposta original:
> [`proposta/PTT6_Filipe_Guilherme_assinado.pdf`](proposta/PTT6_Filipe_Guilherme_assinado.pdf).

- **Versão:** 1.5 (v1.1/v1.2 aprovadas com o orientador; v1.3–v1.5 decididas pela dupla — validar com o orientador)
- **Autores:** Filipe Matias, Guilherme Lucas — **Orientador:** Herbert Oliveira Rocha
- **Última atualização:** 2026-10-22

### Histórico de versões

| Versão | Data | Mudança |
|---|---|---|
| 1.0 | 2026-10-08 | Protocolo congelado (22 decisões, D1–D22). |
| 1.1 | 2026-10-13 | **D23:** plataforma MCU passa de ESP32-**S3** para **ESP32 (LX6)**, o hardware efetivamente disponível; aprovado pelo orientador (não é impeditivo). Impactos em §2, §3, §9 e §13. |
| 1.2 | 2026-10-13 | **D24:** energia por inferência passa a ser **estimada por software** (modelo de potência × tempo medido), **descartando a medição física com INA219** — inviável no prazo, decidido com o orientador. **D25:** execução **em fases** — validar o pipeline com **imagens estáticas** antes de integrar a **câmera ao vivo**. Impactos em §2, §5, §6, §9–§11 e §14–§15. |

| 1.3 | 2026-10-18 | **D26:** dataset **Ultralytics Construction-PPE** (1 classe `vest` → 11 classes). **D27:** um *fine-tuning* por configuração de classes, fora do escopo medido (D9 = sem re-treino por plataforma). **D28:** cronometragem por timestamps monotônicos em software. **D29:** parâmetros do modelo de potência + sensibilidade/*break-even*. **D30:** entradas estáticas como tensores brutos idênticos. **D31:** resolução comum definida pela viabilidade no ESP32. **D32:** coleta em 3 blocos por plataforma. INA219 **definitivamente fora** (pedido ao orientador, mas não será usado). Desenho completo em [`experimentacao.md`](experimentacao.md). |
| 1.4 | 2026-10-19 | **D33:** a **câmera ao vivo passa a ser extra** (fora do caminho crítico e do teste de H1). A validação e a medição usam **ingestão direta de imagens** no modelo: no ESP32 (Arduino/ESP-IDF), as entradas `.bin` chegam pela **serial** (USB-UART) e ficam em RAM antes da janela; no RPi, são lidas do disco para a RAM. Revisa D3, D4 e D25. Detalhes em §17 e [`experimentacao.md`](experimentacao.md). |
| 1.5 | 2026-10-22 | Revisão de literatura com auditoria e contra-revisão ([`revisao/`](revisao/README.md)) + **E0 medido**. **D34:** avaliação conjunta acurácia + latência + energia, lado a lado, com acurácia como condição de validade (referência MLPerf: ≥ 99% do FP32); sem métrica composta escalar. **D35:** H1/H2 reformuladas. **D36:** protocolo do E0 (split de teste, BGR, conf 0,001/NMS 0,65, IC *bootstrap*). **D37:** protocolo de treino (épocas como hiperparâmetro, parada pela validação; **as 80 épocas não têm respaldo**). **D38:** protocolo de quantização (BGR em toda a cadeia, 750 imagens, diagnóstico SiLU × calibração). **D39:** experimentos a refazer. **D40:** firmware alinhado a D23 (ESP32 LX6). |

---

## Resumo do plano

1. Medir, de forma reprodutível, a inferência do YOLOX-Nano int8 em duas plataformas de borda — ESP32 (LX6) (MCU) e Raspberry Pi Zero 2 W (SBC) — para detecção de EPIs.
2. Tarefa é detecção (bounding boxes), não classificação; acurácia entra como mAP@50 e mAP@50-95.
3. Modelo apenas quantizado, sem re-treino; mesmo runtime TFLite int8 nas duas plataformas, para isolar o efeito do hardware.
4. Artefato único por configuração de classes: YOLOX-Nano (ti-lite) treinado uma vez no notebook (D27), convertido para TFLite int8, calibrado com 750 imagens, com hash registrado.
5. Dataset **Ultralytics Construction-PPE** (D26): começamos com uma classe (colete, `vest`) e depois passamos às 11 classes.
6. Medimos latência de ponta a ponta (pré + inferência + pós), throughput e energia por inferência **estimada por software** (total da janela, incluindo o consumo parado).
7. Eficiência comparada por inferências/joule e por EDP (energia × latência), com H1 e H0 formalizadas.
8. Energia **estimada por software** (modelo de potência por modo × tempo medido por fase); **sem medidor físico inline** — a limitação é declarada no relatório.
9. Desenvolvimento **em fases**: validar o pipeline com **imagens estáticas** (sem câmera) e medir com a **mesma ingestão direta de imagens**; a **câmera ao vivo é extra** (v1.4/D33). Um piloto define o número de repetições (piso 30, referência até ~240) e o warm-up pelo próprio dado; duas bordas de GPIO separam as três fases.
10. Estatística: mediana + IQR, teste de Mann-Whitney U com tamanho de efeito e IC por bootstrap; não removemos outliers (opção A), só checamos sensibilidade.
11. Configuração travada: ESP32 (LX6) a 240 MHz sem power management e Raspberry Pi com governor performance a 1 GHz, com ambiente e temperatura registrados.
12. Entrada: imagens do próprio dataset, pré-processadas no PC e **ingeridas diretamente no modelo** (serial no ESP32, disco no RPi), idênticas byte a byte; a câmera ao vivo fica como extra, se houver tempo.
13. Entregáveis: repositório com licença, README de reprodutibilidade, CSV bruto das execuções, vídeo de 3–5 min e relatório técnico conclusivo.
14. Prazo de 15 semanas; depois, publicar o dataset no Zenodo (DOI) e submeter a um evento regional.
15. Título permanece provisório.

---

## 1. Objetivo, pergunta e hipótese

- **Tarefa:** **detecção** de EPIs (saída com bounding boxes), não classificação de imagem.
  A acurácia é medida por **mAP**. **(DECISÃO)**
- **Objetivo geral:** comparar desempenho e eficiência energética da inferência de um modelo
  YOLOX-Nano int8 em uma MCU (ESP32 (LX6)) e um SBC (Raspberry Pi Zero 2 W).
- **Pergunta:** como se comparam latência, throughput e energia por inferência entre as duas
  plataformas ao executar o mesmo YOLOX-Nano int8, detectando EPIs (validação com imagens estáticas
  ingeridas diretamente no modelo)?
- **H1:** o SBC tem throughput maior, mas energia por inferência igual ou superior. *(v1.5/D35: testada com o
  **mesmo artefato int8** nas duas plataformas — acurácia idêntica, verificada pelo E1 — e interpretada junto do
  mAP desse artefato; ver §18.)*
- **H2 (v1.5):** no SBC, o int8 reduz latência e energia por inferência em relação ao fp32, ao custo de uma
  perda de mAP que deve ser quantificada (retenção int8/fp32 com IC); avaliada como trade-off, não como ganho isolado.
- **H0:** não há diferença nas métricas entre as plataformas. **(DECISÃO.)**
- **Eficiência:** inferências/joule **e** EDP. **(DECISÃO.)**
- **Prazo:** cronograma de **15 semanas**.

## 2. Plataformas e entrada

| | MCU | SBC |
|---|---|---|
| Modelo | **ESP32 (LX6)** — dual-core Xtensa LX6, até 240 MHz | Raspberry Pi Zero 2 W (Cortex-A53 quad-core @1 GHz, 512 MB) |
| Entrada | imagens `.bin` via serial → RAM (câmera: extra) | imagens `.bin` do disco → RAM (câmera: extra) |
| Medição | energia **estimada por software** + cronometragem por GPIO | energia **estimada por software** + cronometragem por GPIO |

> **Emenda v1.1 (D23):** a plataforma MCU é o **ESP32 (LX6)**, hardware efetivamente disponível à
> dupla, e não o **ESP32-S3** citado na proposta. O orientador avaliou que a substituição **não é
> impeditiva**. Consequências registradas como ameaça à validade em §9: o LX6 **não possui** as
> extensões vetoriais/AI (SIMD) do S3, o que tende a **aumentar a latência** e alterar o perfil de
> consumo da MCU. O S3 permanece apenas como **referência documental**, não como plataforma medida.

- **Entrada e fases (DECISÃO, v1.2 / D25):** o desenvolvimento é **em fases**. **Fase 1:** conversão
  e validação do modelo com **imagens estáticas** (o dataset), **sem câmera** — garante que a
  detecção funciona antes de qualquer tempo real. **Fase 2:** integração da **câmera ao vivo**. A
  medição final só ocorre com o pipeline validado na fase anterior.
- **Câmera (DECISÃO, v1.4 / D33):** passa a ser **extra opcional** — não bloqueia a medição nem entra no teste de H1. Se for feita: As pipelines diferem
  (esp32-camera vs V4L2/libcamera) → **ameaça à validade**: fixar resolução, formato e número de
  buffers iguais nas duas pontas.
- **Relógio (DECISÃO):** ESP32 (LX6) travado em **240 MHz** com DFS/light-sleep desabilitados;
  RPi com governor **`performance`** a **1 GHz**. Registrar `sdkconfig` e `config.txt`.

## 3. Modelo e runtime

- **Modelo:** YOLOX-Nano int8 (0,91 M params, 1,08 GFLOPs, 416×416). **(DECISÃO: só quantização.)**
- **Runtime (DECISÃO):** **mesmo nas duas plataformas — TFLite int8** — para comparação direta.
- **Artefato (DECISÃO):** um único `.tflite` int8 por **YOLOX-ti-lite → TFLite** (`onnx2tf`), usado
  nas duas pontas; registrar **hash**.
- **Calibração int8 (DECISÃO):** **750 imagens** representativas do conjunto de treino,
  estratificadas por classe, sem vazar validação.
- **Viabilidade na plataforma adotada (nota v1.1):** o YOLOX-Nano a 416×416 (~1,08 GFLOPs) é pesado
  para o **ESP32 (LX6)**, que não tem extensões vetoriais/AI. Antes dos experimentos, o **piloto**
  deve confirmar que a inferência roda e medir a latência esperada; se inviável, mantém-se o mesmo
  runtime/pipeline e ajusta-se a resolução de entrada de forma **idêntica nas duas plataformas**
  (registrando a mudança e seu efeito na comparabilidade).

## 4. Dataset de EPIs

- **Dataset (DECISÃO v1.3 / D26):** **Ultralytics Construction-PPE** (1.416 imgs, 11 classes, AGPL-3.0,
  splits oficiais). Bloco A: **uma classe — colete (`vest`)**; bloco B: **todas as 11 classes** (issue #20).
  Detalhes e contagens em [`dataset.md`](dataset.md).
- **Histórico:** antes da v1.3 o recorte inicial era o colete com CHV/Construction-PPE como candidatos.
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
| Energia por inferência | `E = Σ P_modo · t_fase` — potência do **modelo por modo** × tempo **medido** por fase | **total, incluindo idle (DECISÃO); estimada por software (D24)** |
| Eficiência | inferências/joule | derivada |
| EDP | energia × latência | derivada |
| Acurácia | mAP@50 / mAP@50-95 + AP@50 do colete | **teste** (v1.5/D36; IC *bootstrap*) |
| Retenção de acurácia (v1.5) | mAP(int8) / mAP(fp32), mesmo split | derivada (D34) |

- Mesmo estimando o total, **reportar as 3 fases separadas** (pré/inferência/pós) para permitir reanálise.
- A energia é **derivada de software** (não medida fisicamente): o tempo vem da instrumentação; a
  potência vem de um **modelo parametrizado** (§6). Registrar tensão e valores assumidos.

## 6. Protocolo de medição

- **Energia (DECISÃO v1.2 / D24 — por software):** a medição física inline com **INA219** foi
  **descartada** por ser **inviável no prazo** (decisão conjunta com o orientador). A energia por
  inferência é **estimada por software** a partir de um **modelo de potência por modo de operação**
  (`P_idle`, `P_ativo`) com valores **típicos de datasheet**, multiplicados pelos **tempos medidos por
  fase**: `E = Σ (P_modo × t_fase)` na janela `[t0, t3]`. Registrar tensão de alimentação e os valores
  assumidos. **Limitação a documentar:** não é medição física — o modelo **não captura transientes** e
  a exatidão fica limitada aos valores de datasheet; a comparação é **relativa** entre plataformas sob
  o **mesmo modelo**. Se um medidor físico (INA219/PPK2) ficar disponível, usá-lo apenas como
  **verificação cruzada** (não como método principal).
- **Cronometragem (DECISÃO):** manter **duas bordas de GPIO** (disparo duplo) para delimitar a janela e
  separar as 3 fases. Alternativa por **timestamps de alta resolução em software** é aceita caso o GPIO
  não esteja instrumentado, declarando a troca como limitação.
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
| DFS/light-sleep no ESP32 (LX6) | desabilitar PM ou travar `ESP_PM_CPU_FREQ_MAX`; registrar `sdkconfig` |
| **Plataforma MCU difere da proposta** (ESP32 LX6 em vez de ESP32-S3) | desvio **aprovado pelo orientador**; declarar no relatório; **não** extrapolar resultados para o S3; registrar impactos esperados (sem SIMD, latência/consumo) |
| **Energia estimada por software** (modelo de potência, não medição física) | usar o mesmo modelo nas duas plataformas; registrar tensão e valores assumidos; verificação cruzada se houver medidor; declarar a limitação |
| Modelo de potência não captura transientes/picos | documentar como limitação; reportar energia relativa, não absoluta; sensibilidade a `P_mode` |
| Pipelines de câmera diferentes (extra, v1.4) | fora de H1; se feito, fixar resolução/formato/nº de buffers |
| Ingestão por serial no ESP32 × disco no RPi | transferência **fora** da janela; janela começa com a imagem já em RAM nas duas plataformas (D33) |
| PSRAM lenta / cache | fixar layout do modelo; registrar uso de PSRAM vs SRAM |
| Deriva de frequência/autocorrelação | múltiplas execuções; travar frequência; análise temporal |
| Sem baseline publicado de YOLOX-Nano nas duas plataformas | ver §12 (D16) |

## 10. Plano B (contingência)

- **Energia:** método principal é **software** (modelo de potência × tempo medido, D24). Se surgir um
  medidor físico (INA219/PPK2), usá-lo como **verificação cruzada** — não como método principal.
- **Modelo não performa em algum device:** usar outras versões quantizadas do YOLO, mantendo o protocolo.

## 11. Riscos

| Risco | Prob. | Impacto | Resposta |
|---|---|---|---|
| Energia estimada por software não refletir o consumo real | média | alto | mesmo modelo nas duas plataformas; registrar premissas; verificação cruzada se possível; declarar limitação |
| Modelo não performa em um dispositivo | média | alto | outras versões quantizadas do YOLO |
| Câmera (extra) não integrar | — | baixo | sem impacto em H1 (v1.4); declarar como trabalho futuro |
| Flash/PSRAM do ESP32 insuficiente para guardar as imagens | média | médio | ingestão por serial, uma imagem por vez (D33) |

## 12. Registro de decisões e pendências

### 12.1 Decisões tomadas

| # | Decisão | Valor | Data |
|---|---|---|---|
| D1 | Tarefa | Detecção de EPIs (bounding boxes) | 2026-10-08 |
| D2 | Prazo | 15 semanas | 2026-10-08 |
| D3 | Entrada | Imagens estáticas na validação; câmera ao vivo na fase 2 (revisada por D25 e D33) | 2026-10-08 |
| D4 | Câmera | Entra no escopo, na **fase 2** (revisada por D25; **extra** desde D33) | 2026-10-08 |
| D5 | Janela de latência | Tudo (pré + inferência + pós) | 2026-10-08 |
| D6 | Energia | Total da janela (inclui idle) | 2026-10-08 |
| D7 | Instrumento | ~~Medir energia com INA219~~ — **revogada por D24** (energia estimada por software) | 2026-10-08 |
| D8 | Runtime | Mesmo nas duas plataformas (TFLite int8) | 2026-10-08 |
| D9 | Modelo | Somente quantização | 2026-10-08 |
| D10 | Classes | Começar por colete (1 classe); expandir via issue #20 | 2026-10-08 |
| D11 | Pré-processamento | Idêntico nas duas plataformas | 2026-10-08 |
| D12 | Eficiência | inf/J + EDP | 2026-10-08 |
| D13 | H0 | Adotada | 2026-10-08 |
| D14 | Título | Manter provisório | 2026-10-08 |
| D15 | Ambiente | Registrar (temperatura, tensão de alimentação, buffers) | 2026-10-08 |
| D16 | Ausência de baseline | Rigor de reprodutibilidade + citar modelos análogos como contexto + dataset publicado como contribuição principal | 2026-10-08 |
| D17 | Calibração int8 | 750 imagens estratificadas | 2026-10-08 |
| D18 | Outliers | Opção A (não remover; robusto) com Tukey como sensibilidade | 2026-10-08 |
| D19 | Repetições | Piloto → CV → N; piso 30, teto de referência ~240 | 2026-10-08 |
| D20 | Teste estatístico | Mann-Whitney U + tamanho de efeito + IC bootstrap | 2026-10-08 |
| D21 | Clock/governor | ESP32 (LX6) 240 MHz sem PM; RPi `performance` 1 GHz | 2026-10-08 |
| D22 | Warm-up | Definido por dado no piloto (variação < 2%, piso 5) | 2026-10-08 |
| D23 | **Plataforma MCU** | **ESP32 (LX6)** (hardware disponível) em vez do ESP32-S3 da proposta; aprovado pelo orientador; S3 fica como referência documental | 2026-10-13 |
| D24 | **Energia (instrumento)** | **Estimar por software** (modelo de potência por modo × tempo medido por fase); **INA219 descartado** por inviabilidade no prazo (com o orientador); limitação declarada; medidor físico só como verificação cruzada | 2026-10-13 |
| D25 | **Estratégia de execução** | **Em fases**: validar o pipeline com **imagens estáticas** (sem câmera) e só depois integrar a **câmera ao vivo** | 2026-10-13 |

| D26 | **Dataset** | Ultralytics Construction-PPE; bloco A = `vest` (1 classe), bloco B = 11 classes | 2026-10-18 |
| D27 | **Treino** | Um *fine-tuning* do YOLOX-Nano (ti-lite) por configuração de classes, no notebook, fora do escopo medido; sem QAT; PTQ int8 | 2026-10-18 |
| D28 | **Cronometragem** | Timestamps monotônicos (`perf_counter_ns` / `esp_timer_get_time`), sem I/O na janela; GPIO só como verificação | 2026-10-18 |
| D29 | **Modelo de potência** | Valores nominais + faixa (experimentacao.md §6); H1 avaliada no nominal e nos extremos; *break-even* reportado | 2026-10-18 |
| D30 | **Entradas estáticas** | Tensores `r×r×3` uint8 em letterbox, idênticos byte a byte; JPEG/resize só na fase de câmera | 2026-10-18 |
| D31 | **Resolução** | Comum às duas plataformas; maior de {416, 320, 256, 224, 160} viável no ESP32 (definida no piloto) | 2026-10-18 |
| D32 | **Estrutura de coleta** | 3 blocos independentes por plataforma (power-cycle), N = max(30, N_piloto) | 2026-10-18 |
| D33 | **Ingestão / câmera** | Câmera ao vivo = **extra**; validação e medição com ingestão direta de imagens (serial no ESP32, disco no RPi), transferência fora da janela | 2026-10-19 |
| D34 | **Estrutura de avaliação** | Acurácia (mAP@50, mAP@50:95, AP do colete) + latência + energia reportados lado a lado para cada {plataforma × precisão}; acurácia é condição de validade (referência MLPerf ≥ 99% do FP32), retenção int8/FP32 contínua com IC; sem métrica composta escalar | 2026-10-22 |
| D35 | **Hipóteses** | H1 com o mesmo artefato int8 nas duas plataformas; H2 = trade-off int8 × fp32 no SBC; fp32 no ESP32 só como tentativa condicional | 2026-10-22 |
| D36 | **E0 (acurácia)** | Split de teste (141 imgs, 1.251 anotações), mesmo pré/pós dos dispositivos, BGR, conf 0,001/NMS 0,65 (avaliação) separado de 0,25/0,45 (implantação), AP por classe, IC *bootstrap* por imagens; rótulo explícito de split | 2026-10-22 |
| D37 | **Treino** | Épocas como hiperparâmetro com teto declarado; mAP de validação por época; parada antecipada por paciência; checkpoint pelo melhor mAP de validação; ≥ 1 seed com curva (2ª se houver tempo); hiperparâmetros reportados | 2026-10-22 |
| D38 | **Quantização** | Apenas **correções de execução**: BGR em toda a cadeia e calibração com as 750 imagens já previstas (D17). Sem investigar a quantização (proposta §6.2 exclui *tuning*). Se o int8 corrigido não atingir acurácia aceitável → **Plano B da proposta**: outra versão quantizada do YOLO | 2026-10-22 (rev. 2026-10-23) |
| D39 | **Reaproveitamento** | Latência/energia medidas no RPi valem para o grafo SiLU, mas não como desempenho de detector; refazer E0/E1 e repetir E3/E4 com o artefato final | 2026-10-22 |
| D40 | **Firmware ESP32** | `sdkconfig` deve ter alvo `esp32` (LX6, D23), não `esp32s3`; PSRAM/resolução conforme o módulo real | 2026-10-22 |

### 12.2 Em aberto

Pendências de parametrização (não metodológicas): **aprovar os valores nominais do modelo de potência**
(rascunho em `scripts/power_model.toml`), definir a **resolução comum** `r` no piloto (D31) e confirmar o
módulo do ESP32 (com/sem PSRAM). O protocolo está na **v1.5** (v1.0 congelada + emendas D23–D40). Novas alterações exigem nova
versão documentada neste histórico.

---

## 13. Emenda v1.1 — Plataforma MCU

- **Contexto:** a dupla dispõe de um **ESP32 (LX6)** — o "ESP32 comum" — e **não** de um ESP32-S3.
  O orientador avaliou que a substituição **não é impeditiva** para o projeto.
- **Decisão (D23):** adotar o **ESP32 (LX6)** como plataforma MCU medida. O ESP32-S3 permanece
  apenas como **referência documental** (proposta original).
- **O que NÃO muda:** a pergunta de pesquisa, H1/H0, o par MCU×SBC, o artefato único (YOLOX-Nano
  int8 → TFLite), as métricas, o protocolo de medição (energia por software + cronometragem por GPIO),
  a estatística e os entregáveis.
- **O que muda / deve ser declarado:**
  1. **Ameaça à validade** — o LX6 não tem as extensões vetoriais/AI (SIMD) do S3; resultados **não**
     se extrapolam para o S3. Registrar como limitação (§9).
  2. **Viabilidade** — confirmar no piloto que o YOLOX-Nano roda no LX6 no tempo aceitável (§3).
  3. **Documentação** — atualizar tabelas de plataforma, README, skill e diagramas de arquitetura.
  4. **Rastreabilidade** — este documento mantém o vínculo com a proposta (que citava o S3).

---

## 14. Emenda v1.2 — Energia estimada por software

- **Contexto:** a avaliação com o orientador concluiu que a **medição física de energia (INA219) é
  inviável no prazo** do projeto (lead time do componente e custo de bancada).
- **Decisão (D24):** a energia por inferência passa a ser **estimada por software**, por um **modelo
  de potência**:
  ```
  P(t) ≈ P_modo        (constante por modo: idle / ativo)
  E    = Σ (P_modo × t_fase)   sobre as fases pré, inferência e pós
  ```
  - `t_fase` vem da **instrumentação** (GPIO, §6); `P_modo` vem de **valores típicos de datasheet**,
    registrados com a tensão de alimentação assumida.
- **O que NÃO muda:** as **métricas** (latência, throughput, mJ, inf/J, EDP), a janela (pré+inf+pós),
  a estatística e os entregáveis. A energia continua sendo comparada **entre plataformas**.
- **O que muda / deve ser declarado:**
  1. **Não é medição física** — o modelo **não captura transientes** nem picos dinâmicos.
  2. **Exatidão limitada** aos valores de datasheet; conclusões são **relativas**, não absolutas.
  3. **Premissas registradas** (tensão, `P_idle`, `P_ativo`) e **mesmo modelo** nas duas plataformas.
  4. **Verificação cruzada** opcional com medidor físico, se disponível (Plano B invertido).

## 15. Estratégia em fases (imagens estáticas → tempo real)

- **Motivação:** garantir que a **detecção funciona** antes de lidar com a complexidade do tempo real.
  A câmera **não** é pressuposta desde o início.
- **Decisão (D25):** executar em fases, cada uma com critério de saída:

| Fase | O quê | Critério de saída |
|---|---|---|
| **F1 — Modelo e validação estática** | Dataset preparado, artefato `.tflite` int8 com hash, inferência sobre **imagens estáticas** nas duas plataformas | mAP determinada; mesma predição nas duas plataformas (equivalência) |
| ~~F2 — Câmera e tempo real~~ → **extra (v1.4)** | Integração opcional da câmera ao vivo; não bloqueia F3 | — (opcional) |
| **F3 — Medição** | Coleta das repetições (piloto→N, warm-up por dado), energia por software | `data/raw/*.csv` completo e reprodutível |
| **F4 — Análise e entrega** | Estatística, tabelas/gráficos, relatório, dataset, vídeo | Entregáveis-base concluídos |

- **Regra:** nenhuma medição final antes de a F1 estar funcional. Desde a v1.4 (D33), a **F3 segue
  direto da F1** com a mesma ingestão de imagens; a câmera ao vivo é extra.


## 16. Emenda v1.3 — Desenho experimental e coleta de métricas

- **Contexto:** hardware em mãos (ESP32 LX6 e RPi Zero 2 W, issue #3); dataset definido pelo Filipe
  (Construction-PPE, issue #4); o modelo **já executa no Raspberry Pi**. O INA219 chegou a ser pedido ao
  orientador, mas **não será usado** — a medição física aumentaria demais a complexidade (reafirma D24).
- **Decisões:** D26–D32 (tabela §12.1). O desenho operacional — variáveis, experimentos E0–E6, formato
  do CSV, modelo de potência e checklist de preparação — está em [`experimentacao.md`](experimentacao.md).
- **O que NÃO muda:** pergunta, H1/H0, métricas, janela pré+inf+pós, estatística (Mann-Whitney U +
  Cliff's δ + bootstrap), regra de outliers (opção A) e entregáveis.
- **O que muda / deve ser declarado:**
  1. D9 reinterpretada: existe **um** treino por configuração de classes (o COCO não tem classes de EPI),
     feito antes e fora das medições; nenhuma plataforma recebe modelo diferente.
  2. Cronometragem passa a ser por software (D28), já que não há instrumento externo para registrar as
     bordas de GPIO.
  3. A energia de MCU (chip) e SBC (placa) tem escopos distintos → *overhead* de placa do MCU como
     parâmetro e H1 testada também no pior caso para o MCU (D29).
  4. Pré/pós no SBC (Python/NumPy) e no MCU (C/C++) diferem em linguagem → análise secundária só com
     `t_inf` (ameaça declarada).


## 17. Emenda v1.4 — Câmera ao vivo como extra; ingestão direta de imagens

- **Decisão (D33):** a integração da câmera ao vivo deixa de ser fase obrigatória e passa a **extra**.
  A validação (F1) e a medição (F3) usam **ingestão direta das imagens no modelo**, dentro do ESP32
  (Arduino core ou ESP-IDF) e do Raspberry Pi.
- **Como:** as imagens são pré-processadas uma vez no PC (letterbox `r×r`, BGR uint8, `.bin`, D30).
  - **ESP32:** o PC envia cada `.bin` pela **serial USB-UART** com um quadro simples (cabeçalho + CRC32);
    o firmware guarda a imagem em um buffer em RAM, confirma e só então abre a janela cronometrada
    (pré + inferência + pós). Os resultados voltam pela serial **depois** da janela.
  - **RPi:** os `.bin` são lidos do disco para a RAM antes do laço (como já faz `bench_collect.py`).
- **Por que serial e não flash:** 50 imagens a 160×160 ocupam ~3,8 MB, mais que a partição de dados
  típica do ESP32; uma imagem por vez cabe em RAM/PSRAM. Embutir um subconjunto na flash fica como
  alternativa, registrada no `meta.json`.
- **O que NÃO muda:** pergunta, H1/H0, métricas, janela, estatística, modelo de potência, entregáveis.
- **O que muda:** a fase F2 sai do caminho crítico; o vídeo de demonstração mostra a ingestão de imagens
  e as detecções (câmera só se o extra for feito); a ameaça "pipelines de câmera diferentes" sai do teste
  de H1, e entra a ameaça "ingestão por serial × disco" (mitigada por manter a transferência fora da janela).


## 18. Emenda v1.5 — Acurácia como condição de validade; E0, treino e quantização

**Motivação.** Os resultados da v1.3 no RPi cobriam latência e energia, mas não a acurácia do artefato. A
revisão de literatura ([`revisao/README.md`](revisao/README.md)) e o E0 medido mostraram que (i) na área, energia
e latência só são resultados válidos acompanhados da acurácia do modelo; (ii) o artefato int8 usado retém
~30% do mAP@50 do fp32 (4,7% de mAP@50 com a entrada RGB efetivamente usada no RPi); (iii) as 80 épocas de
treino não têm respaldo citável.

**D34 — Estrutura de avaliação.** Para cada combinação {plataforma × precisão}, reportar lado a lado:
mAP@50 e mAP@50:95 (teste, IC 95% por *bootstrap* de imagens), AP@50 do colete, latência por fase,
throughput, energia estimada, inferências/J e EDP. A acurácia é **condição de validade**: o alvo de
referência é o do MLPerf Inference (≥ 99% do FP32) [Reddi et al. 2020]; um artefato abaixo dele é reportado
com sua retenção de mAP e seus resultados de energia são rotulados "artefato degradado". Não se usa métrica
composta escalar (não há precedente em detecção; FemtoDet e POQ reportam eixos separados/Pareto).

**D35 — Hipóteses.** H1 é testada com o **mesmo** artefato int8 nas duas plataformas (acurácia idêntica,
E1); H2 (trade-off int8 × fp32) é testada no SBC. O fp32 no ESP32 (3,5 MB; FPU escalar sem SIMD; exige PSRAM)
é tentativa condicional: mede-se uma latência preliminar e documenta-se a viabilidade.

**D36 — Protocolo do E0.** `scripts/eval_map_tflite.py`: split de teste (141 imagens, 1.251 anotações),
pré/pós idênticos aos dispositivos, **ordem BGR**, conf 0,001 e NMS 0,65 para mAP (limiares de implantação
0,25/0,45 só para demonstração), AP por classe, IC por *bootstrap* (B = 1000) [Efron & Tibshirani 1993].
Todo número de mAP é rotulado com o split (o 59,9%/29,0% anterior era de validação).

**D37 — Protocolo de treino.** Substitui as 80 épocas fixas: teto declarado, avaliação de mAP de validação
a cada época, parada antecipada por paciência [Prechelt 1998], checkpoint pelo melhor mAP de validação,
cronograma do YOLOX (warm-up; Mosaic/MixUp desligados nas últimas épocas) [Ge et al. 2021], ≥ 1 seed com
curva completa e 2ª seed se houver tempo [Henderson et al. 2018]; reportar todos os hiperparâmetros
[Dodge et al. 2019; Pineau et al. 2021]. Justificativa de orçamento: fine-tuning a partir de pesos
pré-treinados requer bem menos iterações que treino do zero [He et al. 2019].

**D38 — Protocolo de quantização (revisado em 2026-10-23).** A proposta assinada exclui "otimização/tuning da
quantização" (§6.2). Por isso a v1.5 faz **apenas correções de erros de execução**: ordem de canais BGR em
treino, calibração, avaliação e dispositivos, e calibração com as 750 imagens já previstas em D17 (a primeira
rodada usou 100 imagens em RGB). Pesos por canal e ativações por tensor (padrão do TFLite), sem ajustes. Não
se investiga a causa do colapso (o diagnóstico Q2 — variante ReLU — proposto inicialmente foi **retirado**).
Se o int8 corrigido continuar com acurácia inaceitável, aplica-se o risco previsto na proposta (§9: "modelo
não performar em algum dos dispositivos → utilizar outras versões quantizadas do Yolo"). O artefato só segue
para E1–E4 após o E0.

**D39 — Reaproveitamento.** As latências/energias medidas no RPi na v1.3 caracterizam o grafo SiLU e
continuam válidas como tal (dependem do grafo, não dos pesos); não podem ser apresentadas como desempenho de
um detector útil. Se o artefato final mantiver o grafo (Q1), basta refazer E0/E1 e repetir E3/E4 para
confirmar; se mudar o grafo (Q2), refazer E2–E4.

**D40 — Firmware.** O projeto ESP-IDF commitado tem `CONFIG_IDF_TARGET="esp32s3"` e PSRAM octal, em
conflito com D23 (ESP32 LX6). Ajustar o alvo para `esp32`, confirmar o módulo/PSRAM e a resolução (D31).


## 19. Alinhamento com a proposta assinada e situação dos resultados da v1.3 (2026-10-23)

### 19.1 O que é da proposta e o que é emenda

| Item | Proposta assinada | Projeto atual | Natureza |
|---|---|---|---|
| Objetivo geral | Comparar sistematicamente desempenho e eficiência energética do YOLOX-Nano quantizado em uma MCU e um SBC | Mantido sem alteração | Original |
| Pergunta e H1 | Latência, throughput e energia MCU × SBC; H1: SBC com maior throughput e energia igual ou maior | Mantidas | Original |
| Tarefa | "classificação de EPIs" | Detecção (caixas), pois o YOLOX é detector | Emenda v1.0 (D1) |
| MCU | ESP32-S3 | ESP32 (LX6), hardware disponível | Emenda v1.1 (D23) |
| Energia | Medidor inline (ou proxy, Plano B) | Estimativa por software | Emenda v1.2 (D24) — é o Plano B previsto |
| Repetições | 30 por plataforma | Piloto define N, piso 30 | Emenda v1.0 (D19, D22) |
| Entrada | Dataset de EPIs | Imagens do dataset ingeridas no modelo; câmera é extra | Emenda v1.4 (D33) |
| Variáveis dependentes | Latência, throughput, energia | + **acurácia (mAP) como condição de validade** | Emenda v1.5 (D34), fundamentada na literatura |
| Treino | Não previsto | Um fine-tuning (o COCO não tem classes de EPI), com protocolo documentado | Emenda v1.3 (D27) e v1.5 (D37) |
| Quantização | *Tuning* excluído (§6.2) | Só correções de execução (BGR, 750 imagens); sem investigação | Respeita §6.2 (D38 revisado) |
| Modelo não performar | Usar outras versões quantizadas do YOLO | Mantido como Plano B | Original |
| int8 × fp32 (H2) | Não previsto | Comparação secundária no SBC; o fp32 é necessário de qualquer forma como referência de acurácia | Emenda v1.5 (D35), exploratória |

Pendente para o orientador: o **requisito da aplicação** (frequência de inspeção e acurácia mínima do colete)
não consta da proposta e é necessário para julgar se cada plataforma "serve".

### 19.2 Situação dos resultados da primeira rodada no RPi (commit bc05622)

| Resultado | Situação | Observação |
|---|---|---|
| Latência por fase, int8 4 threads (pré 33,2 / inf 109,5 / pós 6,0 / total 148,8 ms, medianas) | **Válido** | Caracteriza o grafo; não depende dos pesos nem da ordem de canais. Ressalva: uma única imagem (D30 previa 50) |
| Escalonamento 1/2/4 threads e Amdahl (E3b) | **Válido** (exploratório) | Mesma ressalva |
| Latência fp32 4 threads (inf 236,6 / total 279,6 ms) | **Válido** | Grafo fp32 |
| 60 s contínuos, temperatura, `throttled=0x0` (E4) | **Válido** | — |
| Linha de base ociosa (rotulada "E0" nos arquivos) | **Válido** | Renomear: E0 no protocolo é a acurácia |
| Memória: int8 1,16 MB, fp32 3,49 MB, RSS 100 MB | **Válido** | — |
| Estatística (Mann-Whitney, Cliff, KW p = 0,83, ACF) | **Método válido** | Recalcular com os dados finais |
| Energia por inferência 253 mJ (int8) / 546 mJ (fp32) | **Corrigir** | Usou só `t_inf`; a janela do protocolo (D5/D6) dá **292 mJ / 589 mJ**, já calculados no relatório |
| Throughput 9,1 inf/s | **Corrigir** | É `1/t_inf`; pela janela completa: **6,7 inf/s** |
| "int8 economiza energia em relação ao fp32" | **Não reportável** | O int8 medido tem 4,7% de mAP@50 (entrada RGB); comparação só com o artefato corrigido (H2) |
| Concordância int8 × fp32 de 13,6% ("E1" nos arquivos) | **Substituído** pelo E0 | Usou 30 cópias deslocadas de uma imagem, não o dataset |
| *Break-even* MCU × SBC (fig. 3) | **Válido como sensibilidade** | Não é medição do MCU; retirar a projeção ESP32-S3 (fora do escopo, D23) |
| Curva de energia por ciclo com "referência MCU t = 500 ms" (fig. 4a) | **Hipotético** | Valor assumido, não medido |
| Qualquer resultado do ESP32 | **Inexistente** | H1 ainda não pode ser testada |

Em resumo: a caracterização do SBC (latência, threads, térmica, memória) é aproveitável; energia e throughput
precisam ser recalculados pela janela completa; nada pode ser apresentado como desempenho de detector até o
artefato int8 corrigido passar no E0; e o MCU ainda precisa ser medido.

---

## Referências

Fontes da v1.5 (verificadas e auditadas): [`revisao/referencias_consolidadas.bib`](revisao/referencias_consolidadas.bib) e
[`revisao/README.md`](revisao/README.md). Fontes primárias anteriores em [`pesquisa-protocolo.md`](pesquisa-protocolo.md). Principais:
MLPerf Tiny (arXiv:2106.07597), Noronha et al. 2025 (arXiv:2505.15622), Maricq et al. OSDI'18,
NIST TN 1830, datasheets **ESP32 (LX6)**/**ESP32-S3** (contexto) e RPi Zero 2 W, repositórios
Espressif/YOLOX/NCNN, datasets CHV/SH17/SFCHD/Construction-PPE. O datasheet do **INA219** permanece
citado apenas como **contexto histórico** (método abandonado em D24).
