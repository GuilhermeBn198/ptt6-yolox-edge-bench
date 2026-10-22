# Auditoria de citações — PTT6 (UFRR): Três relatórios de literatura

**Papel:** Revisor/auditor de citações. **Escopo:** os três relatórios (métrica principal, PTQ em detecção, orçamento de épocas) e seus três `.bib`. Fontes de verificação: Crossref (metadados/DOI), arXiv (texto completo via `export.arxiv.org`/`arxiv.org/pdf`), OpenAlex (metadados/abstract, com `api_key`), e leitura de texto completo (PDF) para as afirmações centrais listadas no escopo da tarefa.

**Importante sobre escopo processado:** Dos ~45 DOIs/arXiv IDs citados nos três `.bib`, foram verificados os metadados (DOI resolve, título/autores/ano/veículo) de **todos**. Foi lido o **texto completo** (não apenas resumo) de 13 trabalhos centrais: Reddi et al. 2020 (MLPerf Inference), Banbury et al. 2021 (MLPerf Tiny), Ge et al. 2021 (YOLOX), He et al. 2019 (Rethinking ImageNet Pre-training), Jacob et al. 2018, Krishnamoorthi 2018, Nagel et al. 2021, Wu et al. 2020 (Integer Quantization), Liang et al. 2021 (Pruning and Quantization survey), Pineau et al. 2021, Moosmann et al. 2023 (TinyissimoYOLO), Wu et al. 2021 (Sustainable AI) e Bhushan et al. 2025. Para os ~14 artigos aplicados de EPI/capacete e para Henderson 2018, Dodge 2019, Prechelt 1998, Gholami 2021/2022, Krishnamoorthi/Nagel (seções secundárias), MQBench, Kuzmin 2023 e Mittal 2024, a verificação ficou limitada a metadados (Crossref/arXiv/OpenAlex) — consistente, aliás, com o que os próprios relatórios já haviam declarado como limitação ("texto completo não disponível", nos casos de Nath 2020, Mittal 2024, Kuzmin 2023 e quatro artigos de EPI).

---

## 1. Veredito geral por relatório

### 1.1 Relatório "Métrica principal" (energia × acurácia)
As duas citações centrais ao argumento (Reddi et al. 2020 — MLPerf Inference; Banbury et al. 2021 — MLPerf Tiny) foram **confirmadas literalmente**: a Tabela I do MLPerf Inference traz exatamente "99% OF FP32 (76.456%) TOP-1 ACCURACY" (ResNet-50) e "99% OF FP32 (0.20 MAP)" / "99% OF FP32 (0.22 MAP)" (SSD-ResNet34 / SSD-MobileNet-v1), e a Seção III.B contém literalmente a frase citada ("We require that almost all implementations achieve a quality target within 1% of the FP32 reference model's accuracy"). O MLPerf Tiny contém literalmente "Each model has a minimum accuracy that must be met for the score to be valid". Essas duas citações sustentam integralmente o argumento central do relatório.

Dois problemas pontuais:
- **Moosmann et al. 2023 (TinyissimoYOLO)** foi deployado principalmente no **MAX78000** (Analog Devices) — não no GAP9. O relatório, na seção de lacunas, erra ao dizer que o artigo usa "GAP9/outro MCU"; GAP9 é citado no artigo apenas como um dos MCUs *compatíveis* em trabalho futuro/possível, não a plataforma de avaliação usada. Correção necessária no texto do relatório (não afeta a citação em si, que é válida).
- **Wu et al. 2021 (Sustainable AI)** é sobre a pegada de carbono de AI em escala de data center (treinamento e manufatura de hardware), não sobre reportar acurácia e energia lado a lado em modelos de borda/TinyML. A afirmação atribuída ("discute a necessidade de reportar custo energético par a par com desempenho de modelo... para evitar otimização enganosa") é uma extrapolação forçada do conteúdo real do artigo — recomenda-se **remover** esta referência do argumento ou substituí-la por uma fonte que trate explicitamente do tema em nível de modelo/edge.
- **Bhushan et al. 2025 (Scientific Reports)** apresenta problemas mais sérios (ver seção 3) e deveria ser removido.

### 1.2 Relatório "PTQ em detecção" (causas do colapso de acurácia)
As citações fundacionais (Jacob et al. 2018, Krishnamoorthi 2018, Nagel et al. 2021) foram **verificadas literal e numericamente**:
- Jacob et al.: Tabela 4.4 confirma exatamente "100% floats 22.1 / 8 bits 21.7" (mAP, SSD-MobileNet, COCO) com "-1.8% relative" — e o contexto confirma que esse resultado é de **quantização com fine-tuning simulado (QAT)**, não PTQ ingênua, exatamente como o relatório caracteriza.
- Krishnamoorthi 2018: confirma literalmente "about 100 mini-batches are sufﬁcient for the estimates of the ranges of the activation to converge" — e a Tabela 2 (quantização só de pesos) mostra MobileNetV1 com quantização assimétrica per-layer caindo a **0.001** de acurácia top-1 (nível de chance para ImageNet/1000 classes) contra 0.709 em ponto flutuante — confirma a afirmação de "acurácia a níveis próximos do acaso".
- Nagel et al. 2021: confirma o mecanismo de magnitudes heterogêneas dentro do mesmo tensor forçando uma única escala per-tensor. **Achado relevante para o projeto:** o mesmo artigo de Nagel et al. (já citado no relatório, mas não explorado neste ponto) tem uma passagem direta sobre o problema de quantizar ativações não-ReLU: *"Other more complex activation functions, such as sigmoid or Swish... require more dedicated support. If this support is not available, we need to add a quantization step before and after the non-linearity... This can have a big impact on the accuracy of quantized model."* Essa passagem é **a explicação mais diretamente aplicável, na própria bibliografia já citada pelo relatório**, ao achado empírico novo (E0) de que o artefato do projeto é o YOLOX-Nano padrão com SiLU (110 LOGISTIC + 104 MUL no grafo, i.e., Swish/SiLU decomposto), não a variante ti-lite (ReLU) prevista no protocolo. O relatório de PTQ **citou Nagel et al. apenas para o argumento per-tensor/per-channel e não usou essa segunda passagem**, que é uma lacuna de uso da própria fonte, não um erro de citação.
- Wu et al. 2020: confirmado — Tabela 7 mostra, para os três detectores avaliados (Faster R-CNN, Mask R-CNN, RetinaNet), deltas de PTQ de -0.35%, -0.24%, -0.28% e de QAT de -0.51%, -0.37%, -0.13%, todos **abaixo de 1 p.p. absoluto**, confirmando a afirmação central do relatório.
- Liang et al. 2021: confirmado literalmente — "nVidia found a model calibrated with 125 images showed only 0.36% top-1 accuracy loss using GoogLeNet on the Imagenet dataset." **Porém**, o relatório atribui a esse resultado também "quantização per-canal" — essa qualificação **não está no texto de Liang et al.** no trecho em questão (o texto menciona apenas calibração por divergência KL do TensorRT, sem mencionar granularidade per-channel nesse ponto específico). Classificar como suporte **parcial**: o número (125 imagens / 0,36%) está correto; o atributo "per-canal" é uma adição não verificada pelo relatório.

**Confronto do diagnóstico do agente de PTQ com a evidência E0 (dado pelo usuário):** O relatório de PTQ hipotetizou que o colapso vem de uma quantização per-tensor de uma saída concatenada que mistura coordenadas de caixa (escala ~0–416) com objectness/probabilidades (escala 0–1) — ou seja, atribuiu o colapso à concatenação final da saída. A evidência empírica E0 (fornecida pelo usuário, não avaliada pelos três agentes de literatura) **refuta especificamente esse mecanismo**: ao ler os tensores internos do int8 **antes** da concatenação final, o mAP50 já é 17,1% (praticamente igual aos 16,7% medidos depois da concatenação), isto é, o colapso já está presente dentro da rede, não é introduzido pela concatenação. Além disso, a saída real não está na escala 0–416 que foi informada aos agentes — é a cabeça crua (offsets/log-wh e sigmoides) com uma única escala int8 (0,0275, faixa -2,6..4,4), o que também invalida a premissa numérica usada no raciocínio sobre "misturar grandezas 0-416 com 0-1".

O que a literatura efetivamente sustenta, à luz do que falta explicar (colapso interno à rede, não na saída):
1. **Ativações não-ReLU (SiLU/Swish) sob PTQ ingênua** — sustentado diretamente por Nagel et al. 2021 (trecho acima), e é a causa mais alinhada ao diagnóstico correto E0 (artefato é YOLOX-Nano padrão com SiLU, não a variante ti-lite com ReLU prevista no protocolo). Nenhum dos três relatórios usa essa citação para esse ponto específico — é uma lacuna de análise, não um erro de citação.
2. **Calibração pequena e descasada (100 imagens RGB vs. 750 planejadas, treino em BGR)** — a literatura (Liang et al. 2021, Krishnamoorthi 2018) sustenta moderadamente que o tamanho por si só (100 vs. 750) dificilmente é a causa dominante (caso de 125 imagens com 0,36% de perda), mas nenhuma das fontes trata do **descasamento de canal de cor** (calibração em RGB vs. treino original em BGR do YOLOX) — esse é um fator adicional não coberto pela literatura revisada e deveria ser testado experimentalmente pelo projeto, não assumido via citação.
3. **Quantização per-tensor (granularidade única) do tensor de saída/dos pesos** — sustentado por Jacob et al. 2018, Krishnamoorthi 2018 e Nagel et al. 2021, mas como fator **adicional**, não como mecanismo único: a evidência E0 já mostra colapso antes da concatenação, então a granularidade da camada final não é a explicação suficiente; granularidade de camadas internas (incluindo as ativações SiLU) é mais provável.

O relatório de PTQ já havia, de forma independente, recomendado "reexaminar a exportação" e havia identificado corretamente, como lacuna, que a arquitetura do projeto poderia divergir da assumida — mas não tinha acesso à informação (fornecida ao auditor, não aos três agentes) de que a rede é SiLU-based, nem à leitura fina da passagem de Nagel sobre Swish. Isso deve ser registrado como atualização, não como erro do relatório original.

### 1.3 Relatório "Orçamento de épocas"
Todas as afirmações centrais sobre o YOLOX foram **confirmadas literalmente no texto completo**:
- "We train the models for a total of 300 epochs with 5 epochs warm-up on COCO train2017... cosine lr schedule" — confirmado.
- "We adopt the MixUp and Mosaic implementation in our model and close it for the last 15 epochs" — confirmado.
- Ausência de qualquer menção a "fine-tun[ing]", "custom dataset" ou "small dataset" no PDF completo do YOLOX — confirmado (busca textual direta nesta auditoria não encontrou essas substrings).
- He et al. 2019: "pre-training takes 100 epochs in ImageNet, and fine-tuning adopts the 2× schedule (∼24 epochs over COCO) and random initialization adopts the 6× schedule (∼72 epochs over COCO)" — confirmado literalmente.
- He et al. 2019: "Original Mask R-CNN models in Detectron were fine-tuned with 90k iterations (namely, '1× schedule') or 180k iterations ('2× schedule')" — confirmado literalmente.
- He et al. 2019: "our results hold even when: (i) using only 10% of the training data" — confirmado literalmente.
- Pineau et al. 2021: confirmado (checklist de reprodutibilidade do NeurIPS 2019, linguagem do artigo bate com a citação).

Os artigos aplicados de detecção de EPI/capacete (SHEL5K, Ali et al. 2022, Kim et al. 2023, Hayat 2022, Lo et al. 2022, Ahmed et al. 2023, Kisaezehra et al. 2023, Lee & Lee 2023, Han & Zeng 2022) têm **todos os DOIs e metadados (título/autor/ano/veículo) confirmados via Crossref** — nenhum DOI inválido ou apontando para outro artigo foi encontrado nessa lista. Quatro deles (Hayat 2022, Lo et al. 2022, Ahmed et al. 2023, Kisaezehra et al. 2023) tiveram, segundo a declaração do próprio relatório, apenas o resumo lido — essa limitação já estava corretamente sinalizada nas "Lacunas" do relatório original e é reafirmada aqui.

---

## 2. Divergência de autoria confirmada: Lo et al. (10.3390/su15010391)

O `.bib` de épocas lista os autores como **"Lo, Jung-Hua; Lin, Lian-Kang; Hung, Chu-Chun"**. O registro oficial no Crossref (confirmado nesta auditoria) lista os autores como:
- **Jye-Hwang Lo** (não "Jung-Hua Lo")
- **Lee-Kuo Lin** (não "Lian-Kang Lin")
- Chu-Chun Hung (confere)

Título, DOI, ano (publicação online 26/12/2022; issue nominal do periódico = Sustainability 15(1), 2023) e veículo conferem. **Erro de nomes próprios confirmado e corrigido no .bib consolidado.**

## 3. Bhushan et al. 2025 — recomendação de remoção

Dois problemas confirmados por leitura do texto completo (versão "Article in Press", recebido em 19/07/2025, aceito em 06/11/2025, Scientific Reports):
1. **Divergência de autoria**: o `.bib` da métrica principal lista "Bhushan, Shashi; Koppuravuri, S.; Prasanthi, G.; others", mas os autores reais são **Ch Madhu Bhushan, Priya Koppuravuri, Nomitha Prasanthi, Firoj Gazi, Md Muzakkir Hussain, Mohammad Abdussami, Aguru Aswani Devi, Jamilurahman Faizi** (SRM University AP e Nangarhar University) — os sobrenomes batem mas os prenomes/iniciais citados no `.bib` são incorretos/inventados.
2. **Conteúdo não corresponde ao uso que o relatório faz dele**: o artigo trata de classificação binária (Visual Wake Words — presença/ausência de pessoa) com MobileNetV2 quantizado, **não de detecção de objetos multi-classe com métrica mAP** (a string "mAP" não ocorre nenhuma vez no texto completo). O relatório cita este trabalho para sustentar que "eficiência energética e desempenho de detecção" são tratados como objetivos conjuntos de avaliação — o artigo de fato trata eficiência energética e acurácia de classificação conjuntamente, mas rotulá-lo como suporte para "detecção de objetos" é impreciso.

**Recomendação: remover esta referência** do argumento sobre detecção de objetos, ou citá-la apenas e explicitamente como exemplo de classificação binária em TinyML (VWW), não detecção.

## 4. Gholami — duplicidade de DOI entre bibs

O `.bib` da métrica principal cita `gholami2022survey` com DOI `10.1201/9781003162810-13` (capítulo do livro "Low-Power Computer Vision", Gholami et al., confirmado via Crossref: mesmo título, mesmos autores, ano 2022). O `.bib` de PTQ cita `gholami2021survey` com `eprint 2103.13630` (arXiv, confirmado via OpenAlex: mesmo título, mesmos autores, ano 2021). **Ambos resolvem corretamente para identidades reais e distintas (capítulo de livro vs. preprint), mas são a mesma obra em dois formatos de publicação** — não é um erro de citação, mas uma duplicidade que deveria ser consolidada sob uma única chave no `.bib` final do artigo (mantida aqui como uma entrada única, usando a versão arXiv por ser de acesso aberto e verificável).

## 5. Outras fontes com leitura limitada (confirmadas apenas por metadados)

- **Nath, Behzadan & Paal (2020)**, 10.1016/j.autcon.2020.103085: DOI/metadados confirmados (Automation in Construction, autores corretos). Texto completo não acessado nesta auditoria nem no relatório original — declarado como lacuna em ambos.
- **Mittal (2024)**, 10.1007/s10462-024-10877-1: DOI/metadados confirmados (Artificial Intelligence Review, autor único Mittal). Apenas metadados verificados — consistente com a lacuna já declarada no relatório de PTQ.
- **Kuzmin et al. (2023)**, arXiv 2303.17951: DOI/metadados confirmados (FP8 versus INT8...). **Nota**: a ordem de autores retornada por metadados automatizados (OpenAlex) aparece como "Mart van Baalen, Kuzmin, Andrey, ..." — isso é provavelmente um artefato de parsing de metadados (comum em entradas OpenAlex para esse período), não uma mudança real de autoria; o `.bib` já lista corretamente Kuzmin como primeiro autor, compatível com a página do arXiv. Apenas metadados verificados nesta auditoria (consistente com a lacuna já declarada).
- **MQBench (Li et al. 2021)**, arXiv 2111.03759: DOI/metadados confirmados. Apenas metadados verificados.
- **Quatro artigos de EPI/capacete** (Hayat 2022, Lo et al. 2022, Ahmed et al. 2023, Kisaezehra et al. 2023): DOI/metadados confirmados; apenas resumo lido (consistente com a lacuna já declarada no relatório de épocas).
- **Henderson et al. 2018, Dodge et al. 2019, Prechelt 1998**: DOI/metadados confirmados via Crossref (título/autores/veículo corretos); não foi lido o texto completo nesta auditoria — as afirmações atribuídas (variância entre seeds; relatar curvas de desempenho vs. orçamento computacional; parada antecipada por validação) são consistentes com o conteúdo amplamente conhecido e citado desses trabalhos de referência, mas ficam classificadas como "não_verificável" por leitura direta nesta rodada.

## 6. Itens sem problema encontrado

Todos os demais DOIs e arXiv IDs citados nos três `.bib` (MLPerf Tiny/Inference, YOLOX, YOLOv4, SGDR, Mask R-CNN, Rethinking ImageNet Pre-training, David et al. TFLite Micro, Ray 2022, Abadade et al. 2023, Alajlan & Ibrahim 2022, Schizas et al. 2022, Menghani 2023, Shuvo et al. 2023, Merenda et al. 2020, Wong et al. 2019 YOLO Nano, García-Martín et al. 2019, Lin et al. 2020 MCUNet, SHEL5K, Ali et al. 2022, Kim et al. 2023, Lee & Lee 2023, Han & Zeng 2022) resolveram para o título, autores, ano e veículo exatamente como citados nos `.bib`. Nenhum DOI inválido, redirecionado para outro artigo, ou com ano/veículo divergente foi encontrado além dos dois casos documentados nas seções 2 e 3.

---

## 7. Lacunas desta auditoria

- Não foi possível ler o texto completo de Henderson et al. 2018, Dodge et al. 2019, Prechelt 1998, Gholami et al. 2021/2022 (seções além do abstract), MQBench, Kuzmin 2023, Mittal 2024, Nath et al. 2020, e dos quatro artigos de EPI com acesso limitado a resumo — apenas metadados (DOI/título/autores/ano/veículo) foram confirmados. Essas afirmações estão marcadas como "não_verificavel" (não "não", pois nada indica erro — apenas não houve confirmação por leitura direta do texto completo dentro do orçamento desta tarefa).
- A checagem de metadados usou Crossref como fonte primária e OpenAlex/arXiv como fallback para DOIs 10.48550/arXiv.* (que o Crossref não indexa diretamente); não foi necessário solicitar novo acesso de rede além do já disponível (Crossref, arXiv, OpenAlex, api.crossref.org, export.arxiv.org, arxiv.org/pdf — todos já permitidos).
- A extração de texto de PDFs via `pypdf` pode falhar silenciosamente em tabelas complexas (ex.: a Tabela I do MLPerf Inference só foi recuperada integralmente após busca textual direcionada); isso não compromete as confirmações reportadas acima, mas significa que passagens tabulares não localizadas por busca textual direta podem não ter sido vistas.
