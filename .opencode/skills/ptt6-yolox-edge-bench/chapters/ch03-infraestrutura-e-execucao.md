# Capítulo 3: Infraestrutura, Cronograma e Riscos

## Ideia Central
O projeto exige **hardware específico** (ESP32-S3 + Raspberry Pi Zero 2 W + medidor INA219), executa em **11 semanas letivas** dentro de um cronograma de **15 semanas**, com **riscos mapeados** e Plano B obrigatório.

## Frameworks Introduzidos
- **Infraestrutura necessária**:
  - Marcado: **ESP32 / Arduino Mega** (usar ESP32-S3)
  - Especificação adicional: **Raspberry Pi Zero 2 W**, **medidor de potência inline (INA219)**, **fonte de alimentação regulada**
  - Não marcados (fora): Kit Raspberry Pi Zero W, FPGA + Quartus Prime Lite, estação com GPU RTX 3090, sensores extra
- **Cronograma (15 semanas)**:
  | Fase | Semanas | Atividades |
  |------|---------|------------|
  | Revisão e protocolo | 1–2 | Revisão bibliográfica; definição final do protocolo |
  | Implementação | 3–5 | Pipeline de inferência idêntico nas duas plataformas |
  | Execução dos experimentos | 6–9 | Coleta das 30 repetições por plataforma |
  | Análise dos dados | 10–11 | Estatística descritiva e comparação entre plataformas |
  | Redação do relatório | 12–13 | Relatório técnico conclusivo |
  | Vídeo e entrega final | 14–15 | Gravação do vídeo de demonstração e ajustes finais |
- **Riscos e Plano B**:
  | Risco | Prob. | Impacto | Plano B |
  |-------|-------|---------|---------|
  | Indisponibilidade/defeito do medidor de potência físico | média | alto | Estimar energia via datasheet × tempo de execução medido, declarando a limitação |
  | Modelo não performar em algum dispositivo | média | alto | Usar outras versões quantizadas do YOLO |

## Conceitos-Chave
- **ESP32-S3**: MCU alvo (representante do extremo de baixo consumo do espectro Edge AI).
- **Raspberry Pi Zero 2 W**: SBC alvo (representante do extremo de maior throughput/recursos).
- **INA219**: sensor de corrente/tensão para medir potência inline.
- **Fonte de alimentação regulada**: garante tensão estável durante a medição de energia.
- **Plano B**: caminho alternativo pré-aprovado para não bloquear a entrega quando um risco se materializa.
- **Proxy documentado**: medição substituta (ex.: estimativa por datasheet) com limitação explicitada.
- **Prazo**: 11 semanas letivas efetivas; cronograma declarado de 15 semanas.

## Modelos Mentais
- Use **"Plano B antecipado"** como seguro: para cada risco de impacto alto, o caminho alternativo já está decidido — não improvise no momento da falha.
- Pense na **regulação de energia** como pré-condição de medição: energia por inferência só é confiável com fonte estável e medição inline.
- Trate o **cronograma** como dependências, não como datas: implementação (3–5) bloqueia experimentos (6–9), que bloqueiam análise (10–11).

## Anti-padrões
- **Deixar a medição de energia para o fim do cronograma**: se o INA219 falhar na semana 9, não há tempo hábil — o Plano B já deve estar pronto para acionar.
- **Rodar experimentos sem protocolo finalizado (semanas 1–2)**: gera retrabalho e invalida dados.
- **Esquecer a fonte regulada**: ruído de alimentação contamina a medição de energia e a latência.
- **Improvisar o Plano B na hora do problema**: perde tempo e rigor; declare a limitação no relatório.

## Exemplo Trabalhado
Risco se materializa: **o INA219 chega com defeito na semana 8**, com experimentos previstos para 6–9.
1. Acionar o Plano B registrado: estimar energia = **potência do datasheet × tempo de execução medido** (cronometragem por GPIO mantida).
2. Declarar a limitação explicitamente no relatório e no README.
3. Manter o protocolo estatístico (30 repetições, descarte das 5 primeiras) — só a fonte da energia muda.
4. Não alterar escopo: a comparação MCU vs SBC continua válida com o proxy documentado.

## Principais Lições
1. O hardware mínimo é **ESP32-S3 + RPi Zero 2 W + INA219 + fonte regulada**.
2. O cronograma tem 6 fases encadeadas; cada uma bloqueia a seguinte.
3. Cada risco de impacto alto tem um Plano B pré-definido — use-o sem improvisar.
4. Estimar energia por datasheet é aceitável **desde que a limitação seja declarada**.

## Conexões
- **Ch 2**: hardware e cronograma viabilizam o protocolo experimental.
- **Ch 4**: riscos mitigados preservam os entregáveis-base (CSV, vídeo, relatório).
- **INA219 / cronometragem GPIO**: instrumentos centrais herdados do protocolo (Ch 2).
