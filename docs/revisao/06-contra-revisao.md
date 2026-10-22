# Contra-revisão (advogado do diabo) da revisão metodológica do PTT6

**Papel:** contestar, com evidência, os pontos centrais da proposta de revisão metodológica (`vb99ade2e_relatorio_revisao_metodologica.md`) e da auditoria de citações (`v678cbb1d_auditoria_citacoes_PTT6.md`), usando apenas literatura acadêmica (OpenAlex, Crossref, arXiv, texto completo via `fetch_article_fulltext`) e documentação oficial de benchmark (MLCommons/MLPerf). Fatos de hardware do ESP32 (FPU escalar, SRAM) foram conferidos via busca geral, não como citação acadêmica, e são tratados como verificação de especificação, não como achado de literatura.

---

## 1) Os limiares de validade do int8 em duas camadas (≥95% e ≥50%; "vest" não cai mais de 2/3)

**Objeção.** A revisão apresenta a Camada 1 (≥95% do mAP FP32) como "alinhada ao MLPerf" e a Camada 2 (≥50%; `vest` não cai mais que 2/3) como critério novo, mas a Camada 2 **não tem nenhuma origem na literatura verificada** — nem a própria revisão alega isso; ela a descreve como proposta própria para preencher uma lacuna. Isso é honesto, mas o relatório final do PTT6 não pode apresentar "50%" e "2/3" como limiares com peso de evidência externa: são escolhas de projeto, e precisam ser rotuladas como tal, não como "critério da literatura adaptado". A literatura efetivamente verificada diz o oposto do que a Camada 2 sugere como prática aceitável: Jacob et al. 2018 reporta, para SSD-MobileNet, queda de 22,1 para 21,7 mAP (**-1,8% relativo**), e Wu et al. 2020 reporta deltas de PTQ de -0,35%, -0,24% e -0,28% (todos abaixo de 1 p.p. absoluto) para três detectores padrão. Não existe, nessas fontes nem em nenhuma outra identificada na auditoria de citações, um precedente de "aceitar comparações de eficiência com um artefato que retém apenas 30-50% do mAP original" — a Camada 2 autoriza exatamente esse uso, sob o rótulo de "degradado, mas comparável", o que é uma analogia enganosa ao enquadramento do MLPerf: lá, abaixo do quality target o escore **não é válido**, ponto final — não existe uma segunda camada de "válido mas degradado" no MLPerf Inference nem no MLPerf Tiny.

**Alternativa mais honesta para um artigo regional.** Em vez de duas camadas com números fixos (95%/50%/2-terços) sem lastro de precedente, a estrutura mais defensável e mais alinhada à prática real da literatura de eficiência em detecção de borda é:

1. Tratar a retenção de mAP (int8/FP32, com IC por bootstrap) como **variável de resposta contínua a ser reportada**, não como um portão binário com limiares arbitrários — isso evita o problema de "por que 50% e não 40%?" que um revisor de banca vai perguntar.
2. Usar o critério do MLPerf (dentro de ~1% / "quality target atingido") **apenas como referência de comparação com quantização bem executada**, não como um portão que o PTT6 precisa necessariamente passar para reportar algo. Isso é exatamente o que FemtoDet (Liu et al., ICCV 2023) e POQ (2025) fazem na prática: nenhum dos dois define um portão binário de aceitação de acurácia — ambos reportam a fronteira de trade-off energia-performance diretamente e deixam o leitor julgar a posição relativa dos pontos.
3. Reportar a fronteira de Pareto acurácia×energia com **todos** os pontos medidos (FP32 e int8, nas duas plataformas), marcando explicitamente quais pontos são "úteis na prática" vs. "diagnóstico", sem fingir que um limiar numérico fixo decide isso por si.
4. Se o objetivo é comunicar que o artefato int8 atual não serve para uso real, a forma mais honesta é declarativa, ancorada no caso de uso ("para EPI, um detector com AP50 de 'vest' caindo de 88,8% para 47,1% não é aceitável para qualquer aplicação de segurança"), não um limiar genérico de "2/3" sem base.

**Evidência de apoio.** FemtoDet define explicitamente a tarefa como um "trade-off entre energia e performance" a ser balanceado, não como um portão de aceitação: "This paper aims to serve as a baseline by designing detectors to reach tradeoffs between energy and performance from two perspectives". O MLPerf Inference (já confirmado na auditoria) define um quality target binário — atingiu ou não — sem uma segunda camada intermediária de "aceitável, mas degradado"; a Camada 2 da revisão é uma extensão não suportada diretamente por essa fonte.

---

## 2) "Testar H1 em FP32 × FP32" é viável no ESP32?

**Objeção parcial.** A premissa de que o ESP32 LX6 "não tem FPU vetorial" está correta, mas a conclusão implícita de que FP32 é inviável por "falta de FPU" não é precisa: o Xtensa LX6 integra, sim, uma unidade de ponto flutuante de precisão simples (hardware FPU), confirmada pelo datasheet Tensilica/Cadence do núcleo (opção "IEEE 754-compliant single-/double-precision scalar floating-point unit") e por discussões técnicas da própria Espressif. O que o LX6 **não tem** é uma FPU vetorizada/SIMD — cada operação float é executada **escalarmente**, uma instrução por vez (`add.s`, `mul.s`), sem paralelismo de dados como em VFP/NEON de um Cortex-A ou mesmo em algumas variantes Cortex-M com DSP/SIMD. Isso é a razão real pela qual FP32 em redes com milhões de parâmetros é lento no ESP32, não a ausência total de hardware de ponto flutuante.

**Viabilidade de memória.** O relatório já identifica corretamente a restrição crítica: 520 KB de SRAM interna contra um modelo FP32 de 3,5 MB — isso por si só inviabiliza manter os pesos inteiros em SRAM; seria necessário executar a partir de PSRAM (até 4-8 MB, se a variante for WROVER — o que, segundo a lacuna já registrada no projeto, ainda não foi confirmado). Mesmo com PSRAM disponível, o acesso a pesos armazenados fora da SRAM interna via o barramento SPI da PSRAM é ordens de magnitude mais lento que acesso à SRAM, o que degrada ainda mais a latência de um modelo FP32 de 3,5 MB rodando camada a camada.

**O que a literatura de MCU diz sobre FP32 vs int8 em TFLite Micro.** O framework TFLite Micro (já citado no projeto) suporta nativamente os dois formatos — "TFLiteMicro supports computation in both ﬂoating point in binary32 format and ﬁxed point on 8-bit integers" — mas os números empíricos de Novac et al. (2021, *Sensors*), medidos em microcontroladores Cortex-M4F (que **têm** FPU escalar, mas ainda assim são ordens de grandeza mais simples que o YOLOX-Nano) mostram que mesmo em redes residuais pequenas o tempo de resposta em FP32 chega a ser várias vezes maior que em int8 para o mesmo hardware — a Figura 12 do artigo relata tempos de resposta de até a faixa de milhares de milissegundos em FP32, contra valores bem menores em int8, para redes com dezenas de filtros por camada, muito menores que o YOLOX-Nano. Extrapolar esse padrão para um detector de milhões de parâmetros no ESP32 sugere uma latência de inferência FP32 da ordem de minutos (não segundos), o que compromete o valor prático de qualquer comparação de energia/throughput em FP32×FP32 no ESP32 — mesmo que seja tecnicamente executável.

**Estrutura correta de comparação, proposta.** A revisão já aponta a direção certa ao reformular H1 condicionando-a a um "portão de acurácia", mas é preciso ser explícito sobre o ESP32-FP32:

1. **Comparação principal (H1):** int8 corrigido (após as mitigações do item (b) do relatório original) em ambas as plataformas — essa é a comparação que efetivamente testa "qual dispositivo é mais eficiente sob a mesma precisão numérica", e é a única comparação viável nas duas pontas em tempo hábil.
2. **FP32 no RPi:** usado como referência de acurácia "sem perdas de quantização" e como limite superior de latência/energia — o RPi, com FPU vetorial (NEON) e SO completo, executa FP32 em tempo prático.
3. **FP32 no ESP32:** só deve ser tentado se, após a correção de canal e eventual uso de PSRAM, uma medição preliminar mostrar que a latência fica dentro de uma ordem de grandeza utilizável (ex. segundos, não minutos) por inferência — caso contrário, deve ser documentado como "testado e descartado por inviabilidade prática" (com o número de latência medido como evidência), não omitido silenciosamente nem forçado a compor H1.

---

## 3) Métrica principal: portão de acurácia vs. métrica composta (ex. mAP/joule)

**Concordância com reserva.** A recomendação da revisão (acurácia como portão, energia/latência como eixo secundário, sem métrica composta escalar) está bem fundamentada e é reforçada por duas fontes adicionais verificadas nesta contra-revisão, que não estavam nos três relatórios originais:

- **FemtoDet** (Liu et al., ICCV 2023) trata explicitamente o problema como um trade-off de dois eixos a serem reportados e balanceados por desenho arquitetural, não como uma métrica única — reforçando que mesmo trabalhos cujo foco central é "detecção eficiente em energia" não colapsam os dois eixos em um número só.
- **POQ** (2025, IEEE Access) formaliza a necessidade de permitir "comparações iso-acurácia e iso-energia" para quantização, o que pressupõe tratar os dois eixos separadamente em vez de combiná-los: "Efficient deployment of deep learning models on resource-constrained devices requires balancing accuracy with energy consumption and/or latency" — a própria formulação do problema ("balancing", não "a single score of") é consistente com a recomendação de Pareto/portão.

**Não existe, nas fontes verificadas, respaldo para uma métrica composta do tipo "mAP por joule" ou "detecções úteis por joule" em detecção de objetos.** A busca dirigida a esse termo específico não retornou nenhum trabalho de detecção que reporte essa métrica como principal; os trabalhos mais próximos de "eficiência energética + detecção" (SqueezeDet, YOLO Nano, FemtoDet) reportam parâmetros/FLOPs/energia e mAP **lado a lado**, nunca como razão única. Isso reforça — em vez de apenas repetir — a recomendação já presente na revisão.

**Estrutura principal recomendada (única):**
1. mAP50 e mAP50:95 por combinação {plataforma × precisão}, com IC por bootstrap — eixo de qualidade, reportado primeiro.
2. Latência (ms/inferência) e energia estimada (mJ/inferência) — eixo de custo, condicionado à combinação que passou (ou não) no critério de qualidade mínima, mas **sempre reportado para todas as combinações**, com marcação clara de quais são comparações "úteis" e quais são "diagnóstico".

**Métricas secundárias recomendadas:** eficiência (inferências/J) apenas como *derivada* das duas primeiras (não como métrica de decisão isolada) e AP50 por classe para a classe focal (`vest`), por ser a métrica mais interpretável para a aplicação (segurança do trabalhador), separada da métrica agregada.

---

## 4) Diagnóstico causal (SiLU): a evidência de Nagel 2021 é suficiente?

**Objeção.** Não é suficiente como prova — e a própria estrutura da auditoria de citações já trata isso corretamente como "hipótese sustentada por citação relevante", não como causa comprovada. A passagem de Nagel et al. 2021 ("Other more complex activation functions, such as sigmoid or Swish... require more dedicated support. If this support is not available, we need to add a quantization step before and after the non-linearity... This can have a big impact on the accuracy of quantized model", conforme reproduzida na auditoria de citações) é uma afirmação de mecanismo geral, não um experimento sobre o YOLOX-Nano especificamente, e não distingue entre pelo menos três fatores concorrentes que coexistem no artefato atual do PTT6: (i) ativação SiLU vs. ReLU, (ii) calibração em RGB com 100 imagens (vs. 750 em BGR previstas), e (iii) granularidade per-tensor vs. per-channel. O próprio E0 só eliminou **um** mecanismo (concatenação de saída), não discriminou entre esses três.

**Experimento mínimo para discriminar as três causas, viável no prazo:** um desenho fatorial 2×2×2 incompleto (ou, mais realista para o orçamento, uma busca sequencial "one-factor-at-a-time" a partir do artefato atual) usando o pipeline de PTQ já existente, sem retreinar o modelo:

1. **Requantizar com calibração BGR de 750 imagens** (mesmo artefato FP32, mesma arquitetura SiLU) — isola o efeito de calibração/canal, que o próprio E0 já indica ser dominante no canal (fator "RGB vs BGR" já medido: 3,6× em mAP50) mas que ainda não foi separado do efeito do *tamanho* da amostra de calibração (100 vs. 750).
2. **Requantizar o mesmo FP32 com quantização per-channel dos pesos** (mantendo SiLU, calibração BGR) — isola o efeito de granularidade, testável com o conversor TFLite padrão (per-channel já é suportado nativamente para pesos de convolução) sem re-treinar.
3. **Treinar/obter a variante `ti-lite` (ReLU)** do YOLOX-Nano e repetir o PTQ nas mesmas condições (BGR, 750 imagens, per-tensor) — isola o efeito da ativação, mas é a etapa mais cara: exige re-treinar do zero ou fine-tuning a partir de um checkpoint ti-lite, se disponível publicamente, e não é um ajuste de poucos minutos como (1) e (2).

**Viabilidade no prazo.** As etapas (1) e (2) são de baixo custo computacional (conversão TFLite + reavaliação de mAP, sem GPU de treino, da ordem de minutos a poucas horas cada) e deveriam ser priorizadas antes de qualquer afirmação causal mais forte no artigo. A etapa (3) é a mais informativa para confirmar a hipótese central (SiLU vs. ReLU), mas também a mais cara; dado o cronograma de 15 semanas e as outras pendências (treino, bootstrap, dois dispositivos embarcados), ela só deve ser tentada **depois** de (1) e (2), e se o tempo não permitir, deve ser declarada como trabalho futuro, com o resultado de (1)+(2) decidindo se a questão fica "em grande parte explicada" ou "ainda em aberto".

---

## 5) Protocolo de treino: épocas + early stopping + 2-3 seeds, viável com RTX 4050?

**Avaliação.** O conjunto de treino do PTT6 (1.142 imagens) é pequeno o suficiente para que o fine-tuning do YOLOX-Nano seja rápido em qualquer GPU de 2024+ com pelo menos 6 GB de VRAM, o que inclui a RTX 4050 — a própria revisão já nota isso ao comparar com as 300 épocas de treino do zero no COCO (~118 mil imagens) do artigo original do YOLOX. Rodar dezenas a poucas centenas de épocas de fine-tuning em ~1.100 imagens em resolução 416 é uma carga de treino ordens de magnitude menor que o treino original, e 2-3 repetições independentes do fine-tuning (não do pipeline completo de quantização/E0, como a revisão já ressalva corretamente) são plausíveis dentro de um orçamento de poucos dias de GPU, não semanas.

**Mínimo aceitável se não houver tempo para seeds múltiplas:** caso o cronograma não comporte 2-3 seeds mesmo só para o fine-tuning, o mínimo defensável para um artigo — alinhado a Dodge et al. 2019 e à prática já citada na revisão — é:
1. Uma única seed, mas com a curva de validação completa documentada (mAP50/mAP50:95 por época) e o critério de parada/seleção de checkpoint explícitos.
2. Declaração explícita da seed única como limitação, não como se fosse resultado estável — isso já está no texto da revisão, mas deve ser reforçado no corpo do artigo (não só nas limitações finais), porque a variância entre seeds em redes de detecção pequenas treinadas em poucos dados tende a ser nada desprezível.
3. Se restar algum tempo de GPU, preferir rodar uma segunda seed "de verificação" (não três) e reportar a faixa (min-max) em vez de descartar completamente a ideia de quantificar variância — um IC formal com apenas duas observações não é estatisticamente robusto, mas uma faixa de dois pontos já comunica a existência de variância ao leitor, o que uma seed única não comunica.

---

## 6) Algo que o revisor de citações aprovou mas não deveria (ou removeu mas deveria ficar)

**Aprovado sem ressalva suficiente: Krishnamoorthi 2018, Tabela 2 (MobileNetV1 caindo a 0,001 de acurácia).** A auditoria de citações confirma a citação como literalmente correta — a Tabela 2 realmente mostra MobileNetV1 caindo a acurácia de nível de chance sob quantização assimétrica per-layer — mas não assinala que esse resultado é de **quantização apenas dos pesos** (não da ativação+peso conjunta, que é o caso de um int8 full-integer como o do PTT6), nem que é uma configuração deliberadamente ingênua (assimétrica, per-layer, sem calibração de ativações) que o próprio whitepaper usa como exemplo de "pior caso" a ser evitado, não como comportamento típico de PTQ. O relatório de PTQ usa esse número para sustentar a plausibilidade de "colapso para quase zero" — mas a Camada 2 da revisão (retenção de 30%, não colapso total) e o restante da literatura confirmada (Jacob et al., Wu et al.: quedas de ~1 p.p.) mostram que o "colapso total" é a exceção patológica, não a expectativa central de PTQ em detectores. O auditor confirmou a citação "corretamente" no sentido literal, mas deveria ter sinalizado esse descompasso de escopo (peso-apenas vs. full-integer; pior-caso vs. caso-típico) como uma ressalva de uso, da mesma forma que fez, corretamente, com o caso de Liang et al. 2021 (atribuição indevida de "per-canal").

**Não contesto a remoção de Bhushan et al. 2025** (autoria divergente + conteúdo de classificação binária, não detecção) nem a de Wu et al. 2021 (Sustainable AI, extrapolação forçada) — ambas as remoções são bem fundamentadas na auditoria e o `.bib` consolidado já reflete essa remoção. Também não encontrei, na checagem desta contra-revisão, nenhuma citação que tenha sido removida do `.bib` consolidado e que devesse ter ficado.

---

## Decisões recomendadas

1. Substituir os limiares fixos de "Camada 1 (≥95%) / Camada 2 (≥50%; vest ≤2/3)" por um relato contínuo da retenção de mAP (com IC por bootstrap), usando o critério do MLPerf apenas como referência de comparação externa, não como portão binário de validade do artefato do PTT6.
2. Rotular explicitamente, em qualquer figura ou tabela de Pareto, os pontos "úteis na prática" vs. "diagnóstico", sem depender de um número de corte arbitrário para decidir essa marcação.
3. Reconhecer explicitamente que o ESP32 LX6 **tem** FPU escalar de precisão simples (não vetorizada) — corrigir a justificativa de inviabilidade do FP32 no ESP32 para "ausência de paralelismo SIMD + 520 KB de SRAM vs. 3,5 MB de modelo", não "ausência de FPU".
4. Medir uma latência preliminar de FP32 no ESP32 antes de decidir se entra no desenho experimental como comparação principal, secundária ou é descartada e apenas documentada como inviável.
5. Fixar a estrutura de comparação em três blocos: (i) int8 corrigido nas duas plataformas como teste principal de H1; (ii) FP32 no RPi como referência de acurácia/latência; (iii) FP32 no ESP32 só como tentativa condicional, documentada como viável ou não.
6. Adotar como métrica principal única mAP (test, com IC) + latência + energia reportados lado a lado — nunca uma métrica composta escalar (ex. mAP/joule) — por ausência de precedente na literatura de detecção revisada (FemtoDet, POQ e demais fontes confirmadas).
7. Antes de qualquer alegação causal mais forte sobre SiLU, executar pelo menos a requantização com calibração BGR de 750 imagens e a requantização per-channel (baixo custo, sem retreino); tratar o teste da variante ti-lite (ReLU) como etapa condicional ao tempo restante e, se não couber no prazo, declará-la como trabalho futuro.
8. Manter o fine-tuning com pelo menos uma seed documentada com curva de validação completa e critério de parada explícito; se o cronograma permitir, rodar uma segunda seed "de verificação" e reportar a faixa min-max, em vez de omitir qualquer medida de variância.
9. Ao citar Krishnamoorthi 2018 (Tabela 2) no relatório de PTQ, acrescentar a ressalva de escopo (quantização apenas de pesos, configuração de pior caso) para não sugerir que "colapso a nível de chance" é o comportamento típico esperado de PTQ int8 full-integer em detectores.
10. Manter as remoções já feitas no `.bib` consolidado (Bhushan et al. 2025; Wu et al. 2021 Sustainable AI) — nenhuma reversão recomendada nesta contra-revisão.
