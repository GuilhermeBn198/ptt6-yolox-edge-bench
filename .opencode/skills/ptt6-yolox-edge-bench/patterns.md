# Padrões e Técnicas — PTT6 YOLOX Edge Benchmark

## Pipeline funcionalmente idêntico
**Quando usar**: ao implementar a inferência na MCU e no SBC.
**Como**: definir a mesma tarefa (YOLOX-Nano int8 → detecção das classes de EPI), mesmas entradas e mesma saída lógica; isolar diferenças de runtime (TFLite Micro no MCU vs TFLite no SBC — mesma família int8, decisão D8) sem alterar o que é computado.
**Trade-offs**: garantir equivalência custa tempo de validação cruzada; ignorá-la invalida o benchmark.

## Protocolo experimental declarado
**Quando usar**: antes de coletar qualquer dado.
**Como**: registrar variável independente (plataforma), dependentes (latência ms, throughput inf/s, energia mJ), repetições por piloto/CV (piso 30), warm-up por dado, medidas de dispersão (mediana, IQR, IC), instrumentos (energia **estimada por software**; cronometragem por GPIO), ameaça (térmica) e mitigação.
**Trade-offs**: rigidez protege a validade, mas exige disciplina para não "ajustar" no meio.

## Desenvolvimento em fases (estático → tempo real)
**Quando usar**: ao planejar a implementação.
**Como**: validar primeiro o modelo com **imagens estáticas** (sem câmera) — mesma predição nas duas plataformas; só depois integrar a **câmera ao vivo**. Medir apenas depois do pipeline validado.
**Trade-offs**: atrasa o "tempo real", mas garante que a detecção funciona antes de somar a complexidade da câmera.

## Estimativa de energia por software (método principal)
**Quando usar**: para obter energia por inferência sem medidor físico.
**Como**: montar um **modelo de potência por modo** (`P_idle`, `P_ativo`, típicos de datasheet) e multiplicar pelos **tempos medidos por fase**: `E = Σ (P_modo × t_fase)` na janela (pré + inferência + pós). Registrar tensão e valores assumidos; usar o **mesmo modelo** nas duas plataformas.
**Trade-offs**: não captura transientes e a exatidão depende do datasheet — a comparação é **relativa**; declare a limitação.

## Verificação cruzada com medidor físico (opcional)
**Quando usar**: se um INA219/PPK2 estiver disponível.
**Como**: medir algumas execuções com o instrumento e comparar com a estimativa por software; reportar o desvio.
**Trade-offs**: aumenta a confiança, mas **não** é o método principal (medir tudo com hardware é inviável no prazo).

## Cronometragem por toggle de GPIO
**Quando usar**: medir latência sem overhead de software.
**Como**: alternar um GPIO no início/fim da inferência e capturar com instrumento/analisador.
**Trade-offs**: exige acesso ao pino e instrumentação; em troca, remove jitter do SO.

## Bateria com pausas de resfriamento
**Quando usar**: durante a coleta das repetições.
**Como**: agrupar execuções em baterias e pausar entre elas; executar em ambiente refrigerado.
**Trade-offs**: aumenta o tempo total de coleta; controla a ameaça térmica.

## Análise estatística descritiva
**Quando usar**: na fase de análise (semanas 10–11).
**Como**: aplicar o warm-up por dado; calcular mediana + IQR e média ± dp por plataforma e métrica; comparar ESP32 LX6 vs RPi Zero 2 W (Mann-Whitney U + tamanho de efeito + IC bootstrap); calcular inferências/joule; testar H1.
**Trade-offs**: estatística simples reduz poder inferencial, mas é adequada ao escopo mínimo.

## Normalização por joule
**Quando usar**: ao comparar eficiência entre plataformas.
**Como**: converter energia por inferência (mJ) em inferências por joule (1 / energia_por_inferência_em_J).
**Trade-offs**: exige estimativa de energia defensável; sem ela, declare o proxy usado.

## Fallback de modelo quantizado
**Quando usar**: quando o YOLOX-Nano não performar em algum dispositivo.
**Como**: testar outras versões quantizadas do YOLO mantendo o mesmo protocolo.
**Trade-offs**: pode introduzir novo artefato; documentar a troca e revalidar a comparabilidade.

## Publicação de dataset com DOI (Zenodo)
**Quando usar**: na consolidação pós-disciplina.
**Como**: publicar o CSV bruto com metadados e licença; obter DOI; citar no relatório.
**Trade-offs**: requer padronização dos dados e curadoria de metadados.
