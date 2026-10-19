# Plano de Ação — PTT6 (registro das issues)

> Gerado a partir do GitHub. Repositório: https://github.com/GuilhermeBn198/ptt6-yolox-edge-bench
> Board (Projects v2): https://github.com/users/GuilhermeBn198/projects/7
> Gerado em: 2026-10-19T17:52:53Z

## Milestones

| Milestone | Abertas | Fechadas | Total |
|---|---:|---:|---:|
| M1 — Revisão e protocolo (semanas 1–2) | 0 | 4 | 4 |
| M2 — Implementação (semanas 3–5) | 5 | 1 | 6 |
| M3 — Experimentos (semanas 6–9) | 2 | 0 | 2 |
| M4 — Análise (semanas 10–11) | 2 | 0 | 2 |
| M5 — Relatório (semanas 12–13) | 2 | 0 | 2 |
| M6 — Vídeo e entrega (semanas 14–15) | 3 | 0 | 3 |

## Issues em sequência

| # | Título | Estado | Milestone | Labels | Criada | Fechada | Bloqueada por |
|---:|---|---|---|---|---|---|---|
| #1 | [Setup] Repositório, licença e estrutura inicial | CLOSED | M1 — Revisão e protocolo (semanas 1–2) | tipo:setup, fase:revisao | 2026-10-09T00:40:48Z | 2026-10-09T02:18:03Z | — |
| #2 | [Método] Revisão bibliográfica e definição final do protocolo experimental | CLOSED | M1 — Revisão e protocolo (semanas 1–2) | tipo:metodo, fase:revisao, prioridade:alta | 2026-10-09T00:40:50Z | 2026-10-13T17:19:58Z | — |
| #3 | [Hardware] Preparar ESP32 (LX6) e Raspberry Pi Zero 2 W | CLOSED | M1 — Revisão e protocolo (semanas 1–2) | tipo:setup, fase:revisao, hardware | 2026-10-09T00:40:53Z | 2026-10-18T23:03:41Z | — |
| #4 | [Dados] Dataset Construction-PPE e classes (1 classe → 11 classes) | CLOSED | M1 — Revisão e protocolo (semanas 1–2) | tipo:setup, fase:revisao | 2026-10-09T00:40:56Z | 2026-10-18T23:03:42Z | 2 |
| #5 | [Pipeline] Inferência YOLOX-Nano int8 no ESP32 (LX6) com ingestão de imagens pela serial (Arduino + TFLite Micro) | OPEN | M2 — Implementação (semanas 3–5) | tipo:metodo, fase:implementacao, prioridade:alta | 2026-10-09T00:40:58Z | — | 2, 3, 4 |
| #6 | [Pipeline] Implementar inferência YOLOX-Nano int8 no Raspberry Pi Zero 2 W | OPEN | M2 — Implementação (semanas 3–5) | tipo:metodo, fase:implementacao, prioridade:alta | 2026-10-09T00:41:01Z | — | 2, 3, 4 |
| #7 | [Validação] Garantir pipeline funcionalmente idêntico nas duas plataformas | OPEN | M2 — Implementação (semanas 3–5) | tipo:metodo, fase:implementacao | 2026-10-09T00:41:03Z | — | 5, 6 |
| #8 | [Instrumentação] Cronometragem por fase (timestamps em software) e telemetria | OPEN | M2 — Implementação (semanas 3–5) | tipo:metodo, fase:implementacao, hardware | 2026-10-09T00:41:06Z | — | 3 |
| #9 | [Energia] Modelo de potência por software (método principal, D24/D29) | OPEN | M2 — Implementação (semanas 3–5) | tipo:metodo, fase:implementacao | 2026-10-09T00:41:09Z | — | 2 |
| #10 | [Experimentos] Piloto (E2) e coleta principal em blocos (E3/E4) | OPEN | M3 — Experimentos (semanas 6–9) | tipo:metodo, fase:experimentos, prioridade:alta | 2026-10-09T00:41:12Z | — | 7, 8 |
| #11 | [QA] Controle de validade: aquecimento, variação térmica e outliers | OPEN | M3 — Experimentos (semanas 6–9) | tipo:metodo, fase:experimentos | 2026-10-09T00:41:15Z | — | 10 |
| #12 | [Análise] Estatística descritiva (média, DP, IQR, CV), inferências/joule e teste de H1 | OPEN | M4 — Análise (semanas 10–11) | tipo:metodo, fase:analise, prioridade:alta | 2026-10-09T00:41:17Z | — | 10 |
| #13 | [Entregável] Gerar tabelas e gráficos comparativos | OPEN | M4 — Análise (semanas 10–11) | tipo:entregavel, fase:analise | 2026-10-09T00:41:20Z | — | 12 |
| #14 | [Entregável] Relatório técnico conclusivo | OPEN | M5 — Relatório (semanas 12–13) | tipo:entregavel, fase:relatorio, prioridade:alta | 2026-10-09T00:41:22Z | — | 13 |
| #15 | [Entregável] README de reprodutibilidade | OPEN | M5 — Relatório (semanas 12–13) | tipo:entregavel, fase:relatorio | 2026-10-09T00:41:25Z | — | 6, 10 |
| #16 | [Entregável] Dataset CSV bruto + publicação Zenodo (DOI) | OPEN | M6 — Vídeo e entrega (semanas 14–15) | tipo:entregavel, fase:entrega | 2026-10-09T00:41:28Z | — | 10 |
| #17 | [Entregável] Vídeo de demonstração (3–5 min) | OPEN | M6 — Vídeo e entrega (semanas 14–15) | tipo:entregavel, fase:entrega | 2026-10-09T00:41:30Z | — | 10 |
| #18 | [Pós-disciplina] Submissão a evento regional de sistemas computacionais | OPEN | M6 — Vídeo e entrega (semanas 14–15) | tipo:entregavel, fase:entrega | 2026-10-09T00:41:33Z | — | 14, 16 |
| #20 | [Dados] Expandir de 1 classe (vest) para as 11 classes do Construction-PPE | OPEN | — | tipo:metodo | 2026-10-13T16:55:23Z | — | 10 |
| #21 | [Método] Detalhar experimentação, variáveis e coleta de métricas (protocolo v1.3) | CLOSED | M2 — Implementação (semanas 3–5) | tipo:metodo, fase:implementacao, prioridade:alta | 2026-10-18T23:03:38Z | 2026-10-19T03:19:49Z | — |
| #23 | [Extra] Integração de câmera ao vivo (opcional — fora de H1) | OPEN | — | tipo:metodo | 2026-10-19T17:52:48Z | — | — |

## Dependências (blocked by)

```
#4 <- 2
#5 <- 2, 3, 4
#6 <- 2, 3, 4
#7 <- 5, 6
#8 <- 3
#9 <- 2
#10 <- 7, 8
#11 <- 10
#12 <- 10
#13 <- 12
#14 <- 13
#15 <- 6, 10
#16 <- 10
#17 <- 10
#18 <- 14, 16
#20 <- 10
```
