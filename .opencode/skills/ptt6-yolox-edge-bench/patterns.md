# Padrões e Técnicas — PTT6 YOLOX Edge Benchmark

## Pipeline funcionalmente idêntico
**Quando usar**: ao implementar a inferência na MCU e no SBC.
**Como**: definir a mesma tarefa (YOLOX-Nano int8 → classes de EPI), mesmas entradas e mesma saída lógica; isolar diferenças de runtime (TFLite Micro vs ONNX/TFLite) sem alterar o que é computado.
**Trade-offs**: garantir equivalência custa tempo de validação cruzada; ignorá-la invalida o benchmark.

## Protocolo experimental declarado
**Quando usar**: antes de coletar qualquer dado.
**Como**: registrar variável independente (plataforma), dependentes (latência ms, throughput inf/s, energia mJ), 30 repetições/plataforma, descarte das 5 primeiras, medidas de dispersão (DP, IQR, CV), instrumentos (INA219 a 10 Hz; toggle GPIO), ameaça (térmica) e mitigação.
**Trade-offs**: rigidez protege a validade, mas exige disciplina para não "ajustar" no meio.

## Medição de energia inline (INA219)
**Quando usar**: para obter energia por inferência confiável.
**Como**: colocar o INA219 em série com a alimentação; fonte regulada; amostragem a 10 Hz; integrar potência no tempo da inferência.
**Trade-offs**: requer hardware dedicado; se falhar, aciona o Plano B (estimativa por datasheet).

## Estimativa de energia por datasheet (Plano B)
**Quando usar**: medidor físico indisponível ou defeituoso.
**Como**: energia ≈ potência do datasheet × tempo de execução medido; **declarar a limitação** no relatório.
**Trade-offs**: perde precisão, mas preserva a entrega e a comparabilidade relativa.

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
**Como**: descartar as 5 primeiras; calcular média, DP, IQR e CV por plataforma e métrica; comparar ESP32-S3 vs RPi Zero 2 W; calcular inferências/joule; testar H1.
**Trade-offs**: estatística simples reduz poder inferencial, mas é adequada ao escopo mínimo.

## Normalização por joule
**Quando usar**: ao comparar eficiência entre plataformas.
**Como**: converter energia por inferência (mJ) em inferências por joule (1 / energia_por_inferência_em_J).
**Trade-offs**: exige medição de energia confiável; sem ela, use estimativa declarada.

## Fallback de modelo quantizado
**Quando usar**: quando o YOLOX-Nano não performar em algum dispositivo.
**Como**: testar outras versões quantizadas do YOLO mantendo o mesmo protocolo.
**Trade-offs**: pode introduzir novo artefato; documentar a troca e revalidar a comparabilidade.

## Publicação de dataset com DOI (Zenodo)
**Quando usar**: na consolidação pós-disciplina.
**Como**: publicar o CSV bruto com metadados e licença; obter DOI; citar no relatório.
**Trade-offs**: requer padronização dos dados e curadoria de metadados.
