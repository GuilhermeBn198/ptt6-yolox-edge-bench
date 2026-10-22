# Revisão de literatura e auditoria metodológica — métricas, quantização e treino (protocolo v1.5)

> Data: 2026-10-22. Processo: três pesquisas independentes (métrica principal; PTQ int8 em detectores;
> orçamento de épocas), uma auditoria de citações, uma revisão metodológica e uma contra-revisão.
> Fontes restritas a periódicos, conferências, arXiv de grupos reconhecidos e documentação oficial de
> benchmark (MLCommons); metadados conferidos em Crossref/OpenAlex/arXiv. Relatórios completos nesta pasta:
> [01](01-metrica-principal.md) · [02](02-quantizacao-ptq-deteccao.md) · [03](03-epocas-fine-tuning.md) ·
> [04 auditoria](04-auditoria-citacoes.md) · [05 revisão](05-revisao-metodologica.md) ·
> [06 contra-revisão](06-contra-revisao.md) · [bib consolidado](referencias_consolidadas.bib).
> As decisões resultantes estão no [`protocolo.md`](../protocolo.md) §18 (D34–D40).

## 1. Qual é a métrica principal? Energia, acurácia ou as duas?

**As duas, com a acurácia como condição de validade e os eixos reportados lado a lado.**

- Os benchmarks de referência da área não aceitam latência/energia isoladas: no **MLPerf Inference** cada
  tarefa tem um *quality target* (tipicamente 99% da acurácia FP32 de referência; Tab. I e §III.B), e um
  resultado de desempenho só é válido se esse alvo for atingido [reddi2020mlperf]. O **MLPerf Tiny** mede
  acurácia, latência e energia, e exige que cada modelo cumpra uma acurácia mínima para o escore valer
  [banbury2021mlperftiny].
- Trabalhos de detecção voltados a energia reportam **acurácia e energia como eixos separados de um
  trade-off** (fronteira de Pareto, comparações iso-acurácia/iso-energia), não como um número único
  [tu2023femtodet; deputter2025poq]. O detector YOLO quantizado para MCU mais próximo do nosso caso reporta
  mAP ao lado da energia por inferência [moosmann2023tinyissimoyolo]. Não encontramos precedente de métrica
  escalar composta (ex.: "mAP por joule") em detecção — **não a usaremos**.
- Surveys de TinyML e de quantização tratam a avaliação como multiobjetivo (acurácia, latência, energia,
  memória) [ray2022tinyml; abadade2023comprehensive; gholami2021survey; menghani2023efficient].

**Consequência para o PTT6:** (a) a comparação **entre plataformas** usa o mesmo artefato, então a acurácia
é igual nas duas (verificada pelo E1) e a energia/latência discriminam as plataformas — mas o resultado só
tem significado aplicado se o artefato atingir uma acurácia de referência; (b) a comparação **int8 × fp32**
é um trade-off acurácia × energia e precisa ser reportada como tal. O alvo do MLPerf (≥ 99% do FP32) entra
como **referência externa**, não como corte inventado por nós; a retenção de mAP é reportada como variável
contínua com IC por *bootstrap* [efron1993bootstrap].

## 2. O que a literatura diz sobre a acurácia do modelo quantizado — e o que medimos

- PTQ int8 bem executada em detectores mantém a queda de mAP abaixo de ~1 p.p. (ex.: −0,35/−0,24/−0,28 p.p.
  em três detectores COCO, Tab. 7 de [wu2020integer]; SSD-MobileNet 22,1→21,7 mAP em [jacob2018quantization]).
- Avaliação correta: mAP@50 e mAP@50:95 do int8 e do fp32 **no mesmo split de teste**, delta absoluto e
  relativo [jacob2018quantization; nagel2021whitepaper; wu2020integer].
- Calibração: conjuntos da ordem de 100 lotes/imagens costumam bastar para estimar faixas de ativação
  [krishnamoorthi2018quantizing; liang2021pruning] — o tamanho isolado dificilmente explica um colapso.
- Ativações não-ReLU (sigmoide/Swish ≈ SiLU) sem suporte dedicado podem ter grande impacto na acurácia
  quantizada [nagel2021whitepaper] — relevante porque o artefato usado é o YOLOX-Nano padrão (SiLU), não a
  variante ReLU prevista.

**Medição própria (E0, `scripts/eval_map_tflite.py`, protocolo COCO, conf 0,001, NMS 0,65):**

| Artefato | Ordem de canais | split | mAP@50 | mAP@50:95 |
|---|---|---|---:|---:|
| fp32 | BGR (como no treino) | val | 60,3% | 29,1% |
| fp32 | BGR | test | 55,0% | 27,2% |
| fp32 | RGB (como roda no RPi) | test | 48,6% | 23,3% |
| int8 | BGR | test | 16,7% | 6,4% |
| **int8** | **RGB (o que foi medido no RPi)** | test | **4,7%** | **1,5%** |

- Os 59,9%/29,0% reportados no roteiro eram de **validação** (reproduzimos 60,3%/29,1%).
- O int8 retém ~30% do mAP@50 do fp32 (BGR) — muito fora da faixa de PTQ bem executada; com a entrada RGB
  usada no RPi, retém 8,5%. AP@50 do colete: 88,8% (fp32) → 47,1% (int8).
- Diagnóstico: ler os tensores internos **antes** da concatenação final dá 17,1% de mAP@50 (≈ 16,7%), logo o
  colapso **não** vem da concatenação da saída (hipótese inicial refutada). Os pesos já são quantizados por
  canal (110 de 113 convoluções). Restam como causas a testar: ativações SiLU por tensor e calibração com
  100 imagens em RGB (o YOLOX treina em BGR).
- **Implicação:** as medições de energia/latência do RPi foram feitas com um artefato sem utilidade de
  detecção. Latência e energia dependem do grafo, não dos pesos — continuam válidas como medida do grafo
  SiLU —, mas não podem ser apresentadas como desempenho de um detector de EPI até o artefato ser corrigido.

## 3. 80 épocas têm respaldo?

**Não.** Nenhuma fonte revisada fixa 80 épocas para fine-tuning de detectores em datasets pequenos.

- As 300 épocas do YOLOX são de treino **do zero** no COCO (~118 mil imagens), com 5 épocas de warm-up e
  Mosaic/MixUp desligados nas 15 finais; o artigo não trata de fine-tuning [ge2021yolox].
- Partindo de pesos pré-treinados, o fine-tuning usa orçamentos bem menores que o treino do zero
  (schedule 2× ≈ 24 épocas COCO vs 6× ≈ 72 do zero) [he2019rethinking].
- Estudos de EPI com YOLO usam de centenas a 1000 épocas, escolhidas empiricamente e em geral com parada
  antecipada pela validação [otgonbold2022shel5k; ali2022labsafety].
- Boas práticas: parar/selecionar pelo erro de validação [prechelt1998earlystopping], reportar curvas,
  hiperparâmetros e orçamento [dodge2019showyourwork; pineau2021reproducibility] e múltiplas seeds
  [henderson2018deeprl].

> **Revisão de 2026-10-23:** o diagnóstico com variante ReLU (Q2) foi retirado por estar fora do escopo da
> proposta (§6.2 exclui *tuning* da quantização). Ficam só as correções de execução (BGR, 750 imagens) e, se
> necessário, o Plano B da proposta (outra versão quantizada do YOLO).

**Protocolo adotado (D37):** épocas como hiperparâmetro com teto declarado; mAP de validação a cada época;
parada antecipada por paciência; checkpoint pelo melhor mAP de validação; ≥ 1 seed com curva completa
(2ª seed se houver tempo, reportando min–máx); todos os hiperparâmetros no artigo.

## 4. O que cada revisão mudou

| Etapa | Correção |
|---|---|
| Auditoria | ~45 DOIs/arXiv conferidos; 13 afirmações centrais lidas no texto completo e confirmadas; removidos Bhushan 2025 (classificação VWW, não detecção) e Wu 2021 (data center); autores de Lo et al. 2022 corrigidos; TinyissimoYOLO roda no MAX78000 (não GAP9); Gholami consolidado |
| Revisão metodológica | Refutou a hipótese da concatenação; identificou SiLU vs ti-lite; propôs H1 condicionada à acurácia |
| Contra-revisão | Removeu limiares inventados (50%/2-terços) em favor de retenção contínua + referência MLPerf; corrigiu "ESP32 sem FPU" (o LX6 tem FPU escalar; falta SIMD float e memória); vetou métrica composta escalar; ressalva de escopo em Krishnamoorthi Tab. 2 (só pesos, pior caso) |
| Verificação final (autor) | Pesos já são per-channel → "requantizar per-channel" não é experimento novo; o firmware ESP-IDF commitado aponta para `esp32s3` (conflita com D23) |

## 5. Limitações desta revisão

- Não há estudo publicado com a mesma combinação (YOLOX-Nano int8, EPI, ESP32 LX6 + RPi Zero 2 W).
- Vários estudos aplicados de EPI foram verificados só por metadados/resumo (evidência fraca, contexto).
- Os valores de potência do RPi Zero 2 W no modelo de energia vêm de medições técnicas publicadas fora de
  periódicos (Geerling, CNX, Tom's Hardware); não há fonte acadêmica equivalente localizada — declarado como
  limitação no artigo.
