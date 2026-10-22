# Orçamento de épocas para fine-tuning do YOLOX-Nano no PTT6: existe respaldo na literatura para 80 épocas?

## Síntese

O artigo original do YOLOX (Ge et al., 2021, arXiv:2107.08430) treina o modelo **do zero** (pesos
aleatórios do backbone CSPDarknet, sem herdar pesos de classificação ImageNet) por **300 épocas**
no COCO *train2017* (~118 mil imagens), com 5 épocas de warmup linear, SGD com momentum 0,9,
*cosine learning-rate schedule* e, nas últimas 15 épocas, desligamento do Mosaic/MixUp para
estabilizar a convergência final. Esse número foi confirmado lendo o texto completo do artigo
(seção "Implementation details"): nenhuma passagem do artigo discute fine-tuning em datasets
customizados ou pequenos, nem propõe um orçamento de épocas para esse cenário — buscas textuais
pelos termos "fine-tun", "custom dataset" e "small dataset" no PDF completo não retornaram
nenhuma ocorrência. Ou seja, **as 300 épocas do YOLOX não são, e nunca foram apresentadas como,
uma referência para fine-tuning**; são o orçamento de um treino do zero em um dataset de escala
muito maior (118 mil vs. 1.142 imagens de treino do PTT6, cerca de 100x mais dados).

A literatura de transfer learning para detecção converge para a conclusão oposta à leitura
ingênua de "copiar as 300 épocas do paper": quando se parte de pesos pré-treinados, o orçamento
de treino necessário cai em uma ordem de grandeza. He, Girshick e Dollár (Rethinking ImageNet
Pre-training, ICCV 2019, arXiv:1811.08883) mostram que, mantendo o pré-treino ImageNet (100
épocas), o fine-tuning subsequente no COCO usa o "*2× schedule*" (~24 épocas), enquanto o
treino do zero exige o "*6× schedule*" (~72 épocas) para alcançar paridade de acurácia — e
que a terminologia "1×/2×" do Detectron/Mask R-CNN corresponde a 90 mil/180 mil iterações, ou
seja, da ordem de 12–24 épocas no COCO, não centenas. O mesmo artigo mostra que mesmo o treino
do zero permanece robusto usando apenas 10% dos dados de treino, desde que o número de
*iterações* (atualizações de gradiente), não de épocas nominais, seja preservado, o parâmetro
que realmente importa é quantas vezes o otimizador atualiza os pesos, e isso depende do
tamanho do dataset e do batch size, não de um número fixo copiado de outro paper com um
dataset 100x maior.

A literatura aplicada de detecção de Equipamentos de Proteção Individual (EPI) com YOLO não
oferece uma convenção fixa de épocas, mas ilustra a prática real do campo. O dataset
SHEL5K (Otgonbold et al., 2022, Sensors, 10.3390/s22062315) reporta ter determinado
"empiricamente" 1000 épocas para a família YOLOv3 e 500 para YOLOv4/YOLOv5/YOLOR — números
discrepantes entre si e sem um critério de parada formal declarado. Já o estudo de monitoramento
de segurança em laboratório com YOLOv5/YOLOv7 (Ali et al., 2022, Sensors, 10.3390/s22228820),
que usa um dataset de 481 imagens, ordem de grandeza mais próxima das 1.142 imagens de treino
do PTT6 do que o COCO —, fixou um **teto** de 300 épocas mas aplicou **parada antecipada**
monitorando overfitting; ou seja, o número efetivo de épocas foi decidido pela curva de
validação, não por um valor nominal copiado de outro artigo. O estudo de reconhecimento de
riscos em canteiro de obras com YOLOv5/v8 (Kim et al., 2023, Sustainability, 10.3390/su152015179)
relata explicitamente ter otimizado a acurácia "alterando épocas, otimizadores e condições de
hiperparâmetros", tratando o número de épocas como hiperparâmetro de busca, não como constante.
Vários outros estudos de detecção de EPI/capacete com datasets de 5 a 11 mil imagens
(10.3390/app12168268; 10.3390/su15010391; 10.32604/iasc.2023.031359; 10.3390/su151813990;
10.3390/s23020944; 10.1109/access.2021.3138407) reportam mAP alto mas não detalham no texto
disponível o número de épocas nem o critério de seleção do checkpoint, uma lacuna de relato
que é, por si, um dado relevante para o diagnóstico do PTT6.

Essa lacuna de relato é exatamente o que a literatura de reprodutibilidade em ML identifica como
problema sistêmico. O relatório do programa de reprodutibilidade do NeurIPS 2019 (Pineau et al.,
2021, JMLR 22(164), arXiv:2003.12206) introduziu um checklist obrigatório cobrindo código,
dados, hiperparâmetros e protocolo experimental justamente para reduzir a prevalência de
resultados não reprodutíveis/não justificados em ML. Dodge et al. (Show Your Work, EMNLP-IJCNLP
2019, 10.18653/v1/d19-1224) recomendam reportar curvas de desempenho em função do orçamento
computacional e dos hiperparâmetros testados — não apenas o resultado final de uma configuração
escolhida sem justificativa. Henderson et al. (Deep RL That Matters, AAAI 2018,
10.1609/aaai.v32i1.11694), embora focado em RL, é a referência padrão para o argumento de que
uma única execução (uma seed) não é suficiente para sustentar conclusões sobre desempenho de um
modelo treinado por otimização estocástica, a variância entre execuções pode ser maior que a
diferença entre configurações. E Prechelt (Early Stopping, But When?, 1998, 10.1007/3-540-49430-8_3)
formaliza o critério clássico, teoricamente motivado, de parar o treino (ou selecionar o
checkpoint) pelo erro em um conjunto de validação separado, e não por um número de épocas fixado
a priori por analogia com outro paper/dataset.

## Resposta à pergunta do track

**As 80 épocas usadas pelo colega não têm respaldo direto na literatura revisada**, nem como
cópia do protocolo do YOLOX (que treina do zero por 300 épocas em um dataset ~100x maior, sem
qualquer orientação de fine-tuning), nem como aplicação de uma convenção estabelecida de
transfer learning (os schedules 1x–6x do Detectron/Mask R-CNN, discutidos por He et al. 2019,
situam o fine-tuning típico em ~12–24 "épocas-equivalentes" no COCO, isto é, bem abaixo de 80,
quando a métrica relevante é o número de iterações de gradiente observadas, não épocas nominais
em datasets de tamanhos muito diferentes). A literatura aplicada de EPI com YOLO tampouco
estabelece 80 como referência: os valores relatados variam de poucas dezenas até 1000 épocas
nominais, e nos poucos estudos que detalham o protocolo, o número de épocas é tratado como
hiperparâmetro de busca, limitado por um teto e decidido na prática por early stopping
monitorando a curva de validação — não como um valor fixo justificado por citação. Em suma: 80
épocas não é metodologicamente errado por si só, mas também não decorre de nenhuma citação
válida; o problema real apontado pela literatura de reprodutibilidade é que esse número deveria
ter sido *justificado empiricamente* (curva de mAP de validação, critério de parada, seed(s)) e
não escolhido/relatado sem essa justificativa.

## Recomendação metodológica concreta para o PTT6

1. Reexecutar (ou, no mínimo, reportar a posteriori) o fine-tuning do YOLOX-Nano monitorando
   mAP@50 (e idealmente mAP@50:95) no split de validação a cada época, e plotar a curva de
   convergência (treino e validação) no artigo — adotando o critério de parada por validação de
   Prechelt (1998) em vez de um número de épocas fixo.
2. Selecionar o checkpoint final pelo melhor mAP de validação (não necessariamente a última
   época), documentando explicitamente esse critério, prática consistente com He et al. (2019)
   e com o checklist de reprodutibilidade de Pineau et al. (2021).
3. Justificar o orçamento de épocas pela escala do dataset do PTT6 (1.142 imagens de treino),
   não pelo número usado no artigo do YOLOX para COCO (118 mil imagens): por analogia com os
   schedules de fine-tuning discutidos por He et al. (2019), um orçamento da ordem de dezenas de
   "passagens efetivas" sobre o dataset (equivalente a um número bem menor de iterações de
   gradiente que o treino do zero no COCO) é o ponto de partida esperado, a ser then ajustado
   por early stopping, e não um número copiado de outro paper com dataset de escala diferente.
4. Reportar o número de seeds/execuções usadas e, idealmente, rodar pelo menos 2–3 seeds
   independentes para o fine-tuning final, reportando média e dispersão do mAP, seguindo o
   alerta de Henderson et al. (2018) sobre a variância entre execuções em otimização estocástica.
5. Reportar explicitamente no artigo: número de épocas efetivamente treinadas, critério de
   parada, hiperparâmetros de otimização (lr, schedule, batch size, warmup), uso ou não de
   Mosaic/MixUp e em que épocas foram desligados, e o(s) seed(s) usado(s), seguindo as
   recomendações de relato de Dodge et al. (2019) e o checklist de Pineau et al. (2021).
6. Tratar o número de épocas como hiperparâmetro de busca (como fazem os estudos de EPI/YOLO
   revisados), e não como constante herdada de outro artigo, dado que nenhuma das fontes revisadas
  , nem o próprio YOLOX, nem a literatura de transfer learning, nem os estudos aplicados de
   EPI, sustenta 80 (ou qualquer outro valor específico) como padrão para fine-tuning em
   datasets desse porte.
7. Caso o tempo de GPU seja limitado, documentar o compromisso explicitamente (ex.: "orçamento
   de treino limitado pela disponibilidade de GPU a N épocas; curva de validação não platô
   confirmado") em vez de apresentar 80 épocas como escolha metodológica definitiva sem
   evidência de convergência.

## Tabela de evidências

| Afirmação | Referência | DOI/arXiv | Onde no texto | Força |
|---|---|---|---|---|
| O YOLOX é treinado do zero (pesos aleatórios/backbone CSPDarknet) por 300 épocas no COCO train2017, com 5 épocas de warmup, SGD, lr inicial 0,01 escalado linearmente pelo batch size, e cosine LR schedule. | Ge, Z.; Liu, S.; Wang, F.; Sun, J. YOLOX: Exceeding YOLO Series in 2021. | arXiv:2107.08430 | Seção 'Implementation details' (texto completo baixado via arXiv): trecho 'We train the models for a total of 300 epochs with 5 epochs warm-up on COCO train2017 ... cosine lr schedule'. | forte |
| O YOLOX desliga Mosaic e MixUp nas últimas 15 épocas do treino de 300 (estratégia de 'forte aumento de dados, depois fine-tune sem aumento forte'), ganhando 2,4 AP sobre a baseline. | Ge, Z.; Liu, S.; Wang, F.; Sun, J. YOLOX: Exceeding YOLO Series in 2021. | arXiv:2107.08430 | Seção 3 (Strong augmentation): trecho 'We adopt the MixUp and Mosaic implementation in our model and close it for the last 15 epochs, achieving 42.0% AP'. | forte |
| O artigo do YOLOX não contém nenhuma recomendação, protocolo ou menção a fine-tuning em dataset customizado/pequeno; as 300 épocas (+5 warmup, -15 sem aug. forte) são um orçamento definido para pré-treino do zero no COCO (118k imagens), não um padrão de fine-tuning. | Ge, Z.; Liu, S.; Wang, F.; Sun, J. YOLOX: Exceeding YOLO Series in 2021. | arXiv:2107.08430 | Busca textual no PDF completo pelos termos 'fine-tun', 'custom dataset', 'small dataset': nenhuma ocorrência. | forte |
| Transferir pesos pré-treinados no ImageNet para detecção permite usar orçamentos de treino muito mais curtos que o treino do zero: o pré-treino consome 100 épocas de ImageNet, mas o fine-tuning subsequente no COCO usa apenas o 'schedule 2x' (~24 épocas), enquanto o treino do zero precisa de ~72 épocas (schedule 6x) para igualar a acurácia. | He, K.; Girshick, R.; Dollár, P. Rethinking ImageNet Pre-training. | 10.1109/iccv.2019.00502 (arXiv:1811.08883) | Seção 3.1 'Normalization'/legenda da figura de comparação: trecho 'pre-training takes 100 epochs in ImageNet, and fine-tuning adopts the 2× schedule (~24 epochs over COCO) and random initialization adopts the 6× schedule (~72 epochs over COCO)'. | forte |
| A terminologia de orçamento de treino em detecção ('1x schedule' = 90 mil iterações, '2x' = 180 mil) vem do Detectron/Mask R-CNN e corresponde, no COCO (~118k imagens, batch típico), a algo da ordem de 12–24 épocas — ou seja, ordens de grandeza abaixo de 300 épocas mesmo quando o treino é longo. | He, K.; Girshick, R.; Dollár, P. Rethinking ImageNet Pre-training. | 10.1109/iccv.2019.00502 (arXiv:1811.08883) | Seção 'Learning rate scheduling': trecho 'Original Mask R-CNN models in Detectron were fine-tuned with 90k iterations (namely, "1x schedule") or 180k iterations ("2x schedule")'. | forte |
| Mesmo treinando do zero (sem pré-treino ImageNet), os resultados se mantêm robustos usando apenas 10% dos dados de treino, desde que o número de iterações seja ajustado, ou seja, o fator decisivo no orçamento de treino é a quantidade de iterações/atualizações de gradiente vista pelo otimizador, não um número fixo de épocas copiado de outro cenário. | He, K.; Girshick, R.; Dollár, P. Rethinking ImageNet Pre-training. | 10.1109/iccv.2019.00502 (arXiv:1811.08883) | Resumo/Seção de experimentos de robustez: trecho 'our results hold even when: (i) using only 10% of the training data'. | forte |
| A literatura de referência para detecção de EPI/segurança em obra usa YOLO com datasets de tamanho comparável (ou maior) ao do projeto (1699 a 11000 imagens), mas raramente documenta um critério formal de parada; quando reportam o número de épocas, ele varia enormemente entre estudos (de poucas centenas a 1000), sem uma convenção única na área. | Otgonbold, M.; Gochoo, M.; Alnajjar, F. et al. SHEL5K: An Extended Dataset and Benchmarking for Safety Helmet Detection. | 10.3390/s22062315 | Seção de resultados: trecho 'the suitable number of epochs for training the YOLOv3 models ... was 1000, while for the other models, namely YOLOv4, YOLOv5-P5, and YOLOR, the value was 500' (determinado empiricamente, sem critério de validação explicitado). | moderada |
| Em um estudo de detecção de EPI com YOLOv5/YOLOv7 e dataset pequeno (481 imagens, ordem de grandeza próxima às 1142 imagens de treino do projeto PTT6), o orçamento nominal foi de até 300 épocas, mas com parada antecipada (early stopping) monitorando overfitting, i.e., o número efetivo de épocas usado foi decidido pela curva de validação, não fixado a priori. | Ali, L.; Alnajjar, F.; Parambil, M. et al. Development of YOLOv5-Based Real-Time Smart Monitoring System for Increasing Lab Safety Awareness in Educational Institutions. | 10.3390/s22228820 | Seção 'Analysis of Experimental Results': trecho 'The number of epochs that were considered necessary for training all of the models was 300; however, an early stopping condition was utilized to prevent model overfitting'. | forte |
| Em outro estudo de reconhecimento de riscos de segurança em canteiro de obras com YOLOv5/YOLOv8, a acurácia foi otimizada 'alterando épocas, otimizadores e condições de hiperparâmetros', reforçando que a prática aceita na área é tratar o número de épocas como hiperparâmetro buscado empiricamente (validação), não copiado de outro paper. | Kim, D.; Kim, J.; Jeong, J. Application of YOLO v5 and v8 for Recognition of Safety Risk Factors at Construction Sites. | 10.3390/su152015179 | Resumo do artigo: trecho 'accuracy was improved by changing epochs, optimizers, and hyperparameter conditions'. | fraca |
| Estudos de detecção de capacete/EPI com datasets de 5000 a 11000 imagens reportam mAP de 92–97% sem detalhar no resumo o número de épocas de treino nem o critério de seleção do checkpoint, evidenciando uma lacuna comum de relato metodológico na área (consistente com a crítica de reprodutibilidade abaixo). | Hayat, A.; Morgado-Dias, F. Deep Learning-Based Automatic Safety Helmet Detection System for Construction Safety; Lo, J.; Lin, W.; Hung, W. Real-Time Personal Protective Equipment Compliance Detection Based on Deep Learning Algorithm. | 10.3390/app12168268; 10.3390/su15010391 | Resumos (texto completo não disponível em acesso aberto): nenhuma menção a número de épocas, seeds ou critério de parada. | fraca |
| O relatório oficial do programa de reprodutibilidade do NeurIPS 2019 recomenda explicitamente reportar código, dados, hiperparâmetros e os checklists de reprodutibilidade como parte da submissão, como forma de reduzir resultados não reprodutíveis em ML. | Pineau, J.; Vincent-Lamarre, P.; Sinha, K.; Larivière, V.; Beygelzimer, A.; d'Alché-Buc, F.; Fox, E.; Larochelle, H. Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program). | arXiv:2003.12206 (J. Mach. Learn. Res. 22(164), 2021) | Resumo e Seção 1: descrição do 'code submission policy' e do 'Machine Learning Reproducibility checklist' adotados pelo NeurIPS. | forte |
| Execuções únicas (single-run) de algoritmos de aprendizado profundo podem variar drasticamente apenas por mudança de seed aleatória, de forma que comparações e conclusões baseadas em uma única execução não são estatisticamente confiáveis; múltiplas seeds/execuções e reporte de variância são necessários. | Henderson, P.; Islam, R.; Bachman, P.; Pineau, J.; Precup, D.; Meger, D. Deep Reinforcement Learning That Matters. | 10.1609/aaai.v32i1.11694 | Resumo e discussão de variância entre seeds (AAAI 2018); trabalho amplamente citado como referência para a necessidade de múltiplas execuções em ML. | moderada |
| Recomendações de boas práticas de relato experimental em ML incluem descrever explicitamente o orçamento computacional/hiperparâmetros de busca, curvas de validação e os critérios usados para seleção do modelo final, para que outros pesquisadores possam reproduzir e comparar resultados de forma justa. | Dodge, J.; Gururangan, S.; Card, D.; Schwartz, R.; Smith, N. A. Show Your Work: Improved Reporting of Experimental Results. | 10.18653/v1/d19-1224 | Resumo e proposta do artigo (EMNLP-IJCNLP 2019): recomendação de reportar curvas de desempenho em função do orçamento computacional e dos hiperparâmetros testados. | moderada |
| A prática de parar o treino (ou escolher o checkpoint) pelo erro em um conjunto de validação separado, em vez de um número fixo de épocas, é um método clássico e teoricamente motivado para evitar overfitting em redes neurais. | Prechelt, L. Early Stopping, But When? | 10.1007/3-540-49430-8_3 | Capítulo completo (Lecture Notes in Computer Science, 1998): define critérios formais de parada antecipada baseados no erro de validação. | forte |
| O cosine learning-rate schedule usado pelo YOLOX (e por muitos detectores modernos) vem do método de 'warm restarts' com decaimento cosseno, originalmente proposto para acelerar a convergência de SGD sem necessidade de ajuste manual fino do número de épocas. | Loshchilov, I.; Hutter, F. SGDR: Stochastic Gradient Descent with Warm Restarts. | arXiv:1608.03983 | Resumo do artigo (ICLR 2017): proposta do decaimento cosseno de taxa de aprendizado com reinícios. | moderada |
| O YOLOv4 (citado pelo próprio YOLOX como fonte do MixUp/Mosaic e das técnicas de 'bag of freebies') documenta seu próprio protocolo de treino (épocas, augmentations, schedule) separadamente para o COCO, reforçando que cada detector da família YOLO define seu próprio orçamento de época para treino do zero no COCO, nenhum define um padrão universal para fine-tuning em datasets pequenos e customizados. | Bochkovskiy, A.; Wang, C.-Y.; Liao, H.-Y. M. YOLOv4: Optimal Speed and Accuracy of Object Detection. | arXiv:2004.10934 | Resumo e seção de setup experimental do artigo: protocolo de treino específico para COCO, com técnicas de 'bag of freebies' citadas também na seção 2 do YOLOX. | moderada |

## Lacunas identificadas

- Nenhuma fonte revisada por pares define um número "padrão" de épocas para fine-tuning de
  detectores YOLO em datasets de ~1 mil imagens; a prática de campo (estudos de EPI) usa valores
  heterogêneos (de poucas centenas a 1000), quase sempre combinados com early stopping ou busca
  de hiperparâmetros, nunca como constante fixa citável.
- Não foi possível obter o texto completo (apenas resumo) de quatro artigos de detecção de EPI
  (10.3390/app12168268, 10.3390/su15010391, 10.3390/su151813990, 10.32604/iasc.2023.031359 teve
  texto completo mas sem seção de épocas localizável por extração de PDF), o que limita a
  verificação direta do número de épocas/critério de parada usado nesses estudos — foram citados
  apenas pelo que consta no resumo (classificados como evidência "fraca").
- O artigo de monitoramento de conformidade de EPI por re-identificação de trabalhadores
  (10.1016/j.autcon.2022.104312) não pôde ser acessado em texto completo nem mesmo por extração
  padrão (requereria scraping de HTML específico fora do escopo das fontes permitidas) e não foi
  usado como evidência.
- Não foi encontrada, dentro das regras de fontes exigidas (periódicos/conferências revisados,
  preprints estabelecidos, documentação oficial de benchmarks), nenhuma citação que estabeleça
  diretamente "80 épocas" como escolha documentada em transfer learning de detectores YOLO em
  datasets pequenos, o que é consistente com a conclusão de que a escolha do colega carece de
  base citável.
