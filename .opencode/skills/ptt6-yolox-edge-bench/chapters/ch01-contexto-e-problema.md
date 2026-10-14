# Capítulo 1: Contexto, Problema e Objetivos

## Ideia Central
Comparar, de forma sistemática e quantificada, desempenho e eficiência energética de **YOLOX-Nano quantizado em int8** rodando em uma **MCU (ESP32 LX6)** e um **SBC (Raspberry Pi Zero 2 W)** para **detecção de EPIs** em cenários de **Edge AI**.

## Frameworks Introduzidos
- **Pergunta de pesquisa**: "Como se comparam latência, throughput e energia por inferência entre uma MCU (ESP32 LX6) e um SBC (Raspberry Pi Zero 2 W) ao executar o mesmo YOLOX-Nano (int8) para detecção de EPIs?"
  - Quando usar: como filtro de escopo de qualquer decisão do projeto.
  - Como: se a atividade não produz dado que responda latência/throughput/energia nas duas plataformas, está fora de escopo.
- **Hipótese H1**: o SBC tem throughput maior que a MCU, mas energia por inferência igual ou superior, por causa do maior consumo estático.
  - Quando usar: para orientar a análise estatística e a discussão dos resultados.
  - Como: testar a direção do throughput **e** da energia; reportar também inferências/joule.
- **Objetivo geral**: comparar sistematicamente desempenho e eficiência energética do YOLOX-Nano int8 em MCU e SBC representativos do espectro Edge AI.
- **Objetivos específicos** (4 itens):
  1. Implementar um pipeline de inferência **funcionalmente idêntico** nas duas plataformas.
  2. Medir **latência média** e **throughput** de inferência em cada plataforma.
  3. Estimar a **energia por inferência** (método por software, baseado em modelo de potência × tempo).
  4. Comparar a **eficiência energética normalizada (inferências por joule)**.

## Conceitos-Chave
- **Edge AI**: inferência de redes neurais diretamente em dispositivos de borda, de baixo custo e consumo restrito.
- **Compromisso MCU vs SBC**: escolha entre desempenho, consumo energético e complexidade de desenvolvimento, raramente quantificada de forma sistemática.
- **YOLOX-Nano**: detector leve baseado em YOLOX, escolhido por caber em dispositivos embarcados.
- **Quantização int8**: representação de pesos/ativações em inteiro de 8 bits, reduz tamanho e custo de computação.
- **Detecção de EPIs**: aplicação-alvo (Equipamentos de Proteção Individual) usando um dataset dedicado; saída em bounding boxes, avaliada por mAP.
- **Latência de inferência (ms)**: tempo de uma inferência.
- **Throughput (inferências/s)**: número de inferências por segundo.
- **Energia por inferência (mJ)**: energia consumida dividida pelo número de inferências.

## Modelos Mentais
- Use "MCU primeiro" quando o critério for eficiência energética e consumo estático — a MCU tende a ser mais econômica por inferência, mesmo mais lenta.
- Use "SBC primeiro" quando o critério for throughput e disponibilidade de recursos (memória, núcleos, SO).
- Pense no SBC como tendo **consumo estático mais alto**: mesmo ocioso, ele consome; esse fator domina a energia por inferência em cargas curtas.

## Anti-padrões
- **Otimizar/tunar a quantização**: está explicitamente fora de escopo; adiciona variável confundidora.
- **Usar mais de um modelo ou dataset**: quebra a comparabilidade controlada.
- **Comparar com GPU RTX 3090 ou nuvem**: foge do objetivo embarcado.
- **Tratar velocidade como único critério**: ignora a hipótese central (energia) e a eficiência normalizada.

## Exemplo Trabalhado
Cenário: você precisa decidir se adiciona suporte a um segundo modelo para "melhorar" o experimento. Aplique os objetivos: (1) só há orçamento/prazo para um pipeline idêntico; (2) H1 fala de **duas** plataformas com **o mesmo** modelo; (3) um segundo modelo tornaria a comparação dependente do modelo, não da plataforma. Decisão: **manter um único modelo (YOLOX-Nano int8)** e registrar a ideia em "trabalhos futuros".

## Principais Lições
1. O projeto é de **medição comparativa controlada**, não de treinamento de modelo.
2. Toda decisão de escopo é validada contra a pergunta de pesquisa e H1.
3. A métrica diferenciadora é **inferências por joule**, não apenas throughput.
4. A motivação é prática: Edge AI de baixo custo exige compromissos quantificados.

## Conexões
- **Ch 2**: o protocolo experimental operacionaliza esta pergunta e H1.
- **Ch 3**: o hardware (ESP32 (LX6), RPi Zero 2 W, fonte) materializa as duas plataformas.
- **Ch 4**: os resultados desta pergunta viram o relatório e o dataset publicados.
