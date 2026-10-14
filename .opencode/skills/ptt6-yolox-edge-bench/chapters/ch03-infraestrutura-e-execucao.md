# Capítulo 3: Infraestrutura, Cronograma e Riscos

## Ideia Central
O projeto exige **hardware específico** (ESP32 (LX6) + Raspberry Pi Zero 2 W + fonte de alimentação), executa em **11 semanas letivas** dentro de um cronograma de **15 semanas**, com **riscos mapeados** e **execução em fases**. A energia é **estimada por software** — **sem medidor físico inline** (decisão v1.2).

## Frameworks Introduzidos
- **Infraestrutura necessária**:
  - Marcado: **ESP32 (LX6)** (plataforma adotada; a proposta citava ESP32-S3 — ver `docs/protocolo.md` §13)
  - Especificação adicional: **Raspberry Pi Zero 2 W**, **fonte de alimentação regulada**, **câmera** (apenas na fase 2)
  - Não marcados (fora): Kit Raspberry Pi Zero W, FPGA + Quartus Prime Lite, estação com GPU RTX 3090, sensores extra
  - **Dispensado na v1.2:** **medidor de potência inline (INA219)** — a energia passou a ser **estimada por software** (D24)
- **Execução em fases (D25)** — a câmera **não** entra desde o início:
  | Fase | O quê | Critério de saída |
  |------|-------|-------------------|
  | F1 | Modelo int8 + validação com **imagens estáticas** | mAP determinada; equivalência entre plataformas |
  | F2 | **Câmera ao vivo** e tempo real | pipeline estável nas duas pontas |
  | F3 | Medição (piloto→N, warm-up por dado) | CSV bruto completo |
  | F4 | Análise e entrega | entregáveis-base |
- **Cronograma (15 semanas)**:
  | Fase | Semanas | Atividades |
  |------|---------|------------|
  | Revisão e protocolo | 1–2 | Revisão bibliográfica; definição final do protocolo |
  | Implementação | 3–5 | F1 (estático) e início da F2 (câmera) nas duas plataformas |
  | Execução dos experimentos | 6–9 | Coleta das repetições por plataforma (N definido no piloto) |
  | Análise dos dados | 10–11 | Estatística descritiva e comparação entre plataformas |
  | Redação do relatório | 12–13 | Relatório técnico conclusivo |
  | Vídeo e entrega final | 14–15 | Gravação do vídeo de demonstração e ajustes finais |
- **Riscos e resposta**:
  | Risco | Prob. | Impacto | Resposta |
  |-------|-------|---------|----------|
  | Energia por software não refletir o consumo real | média | alto | mesmo modelo nas duas plataformas; registrar premissas; verificação cruzada; declarar limitação |
  | Modelo não performar em algum dispositivo | média | alto | Usar outras versões quantizadas do YOLO |
  | Câmera não integrar na fase 2 | baixa | médio | manter a validação estática como evidência; declarar a fase não concluída |

## Conceitos-Chave
- **ESP32 (LX6)**: MCU alvo (representante do extremo de baixo consumo do espectro Edge AI).
- **Raspberry Pi Zero 2 W**: SBC alvo (representante do extremo de maior throughput/recursos).
- **Energia estimada por software**: `E = Σ (P_modo × t_fase)`, com potência típica de datasheet e tempo medido por fase.
- **Fases (estático → tempo real)**: ordem de trabalho que valida o modelo antes de integrar a câmera.
- **Fonte de alimentação regulada**: garante tensão estável (e valor de `V` no modelo de energia).
- **Plano B / proxy documentado**: caminho alternativo pré-aprovado; toda substituição traz limitação explicitada.
- **Medidor físico (INA219)**: **contexto histórico** — descartado; se disponível, só como **verificação cruzada**.

## Modelos Mentais
- Use **"Plano B antecipado"** como seguro: para cada risco de alto impacto, o caminho já está decidido.
- Pense na **energia por software como um modelo, não uma medição**: o que importa é registrar as **premissas** (`P_modo`, tensão) e usar o **mesmo modelo** nas duas pontas.
- **Valide o estático antes do tempo real**: a câmera é a maior fonte de variabilidade; só entra depois do modelo validado.
- Trate o **cronograma** como dependências: implementação (3–5) bloqueia experimentos (6–9), que bloqueiam análise (10–11).

## Anti-padrões
- **Ir para tempo real antes de validar o modelo estaticamente**: soma a variabilidade da câmera a um pipeline ainda não confiável.
- **Tratar a energia por software como medição exata**: não declare conclusões absolutas sem as premissas e a limitação.
- **Rodar experimentos sem protocolo finalizado (semanas 1–2)**: gera retrabalho e invalida dados.
- **Improvisar a resposta a um risco na hora do problema**: perde tempo e rigor.

## Exemplo Trabalhado
Dois cenários:
1. **Surgiu um INA219 emprestado na semana 8.** Use-o como **verificação cruzada** de algumas execuções e reporte o desvio — **sem** trocar o método principal (software) nem reprocessar todo o experimento. Mantenha o mesmo modelo e declare a comparação.
2. **A câmera não integra na fase 2.** Mantenha a **validação estática (F1)** como evidência, declare a fase não concluída como limitação e **não** altere o protocolo de medição.

## Principais Lições
1. O hardware mínimo é **ESP32 (LX6) + RPi Zero 2 W + fonte** (+ câmera na fase 2); o medidor físico foi dispensado.
2. O trabalho é **em fases**: estático antes de tempo real.
3. A energia é **estimada por software** — registre premissas e declare a limitação.
4. Cada risco de alto impacto tem uma resposta pré-definida — use-a sem improvisar.

## Conexões
- **Ch 2**: hardware, fases e cronograma viabilizam o protocolo experimental.
- **Ch 4**: riscos mitigados preservam os entregáveis-base (CSV, vídeo, relatório).
- **Energia por software / fases**: decisões herdadas do protocolo (Ch 2).
