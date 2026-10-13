# Plano de Ação — PTT6 (registro das issues)

> Gerado a partir do GitHub. Repositório: https://github.com/GuilhermeBn198/ptt6-yolox-edge-bench
> Gerado em: 2026-10-13T17:18:35Z

## Milestones

| Milestone | Abertas | Fechadas | Total |
|---|---:|---:|---:|
| M1 — Revisão e protocolo (semanas 1–2) | 3 | 1 | 4 |
| M2 — Implementação (semanas 3–5) | 5 | 0 | 5 |
| M3 — Experimentos (semanas 6–9) | 2 | 0 | 2 |
| M4 — Análise (semanas 10–11) | 2 | 0 | 2 |
| M5 — Relatório (semanas 12–13) | 2 | 0 | 2 |
| M6 — Vídeo e entrega (semanas 14–15) | 3 | 0 | 3 |

## Issues em sequência

| # | Título | Estado | Milestone | Labels | Criada | Fechada | Bloqueada por |
|---:|---|---|---|---|---|---|---|
| #1 | [Setup] Repositório, licença e estrutura inicial | CLOSED | M1 — Revisão e protocolo (semanas 1–2) | tipo:setup, fase:revisao | 2026-10-09T00:40:48Z | 2026-10-09T02:18:03Z | — |
| #2 | [Método] Revisão bibliográfica e definição final do protocolo experimental | OPEN | M1 — Revisão e protocolo (semanas 1–2) | tipo:metodo, fase:revisao, prioridade:alta | 2026-10-09T00:40:50Z | — | — |
| #3 | [Hardware] Adquirir e preparar ESP32-S3, RPi Zero 2 W, INA219 e fonte regulada | OPEN | M1 — Revisão e protocolo (semanas 1–2) | tipo:setup, fase:revisao, hardware | 2026-10-09T00:40:53Z | — | — |
| #4 | [Dados] Selecionar dataset de EPIs e definir classes de classificação | OPEN | M1 — Revisão e protocolo (semanas 1–2) | tipo:setup, fase:revisao | 2026-10-09T00:40:56Z | — | 2 |
| #5 | [Pipeline] Implementar inferência YOLOX-Nano int8 no ESP32-S3 (TFLite Micro) | OPEN | M2 — Implementação (semanas 3–5) | tipo:metodo, fase:implementacao, prioridade:alta | 2026-10-09T00:40:58Z | — | 2, 3, 4 |
| #6 | [Pipeline] Implementar inferência YOLOX-Nano int8 no Raspberry Pi Zero 2 W | OPEN | M2 — Implementação (semanas 3–5) | tipo:metodo, fase:implementacao, prioridade:alta | 2026-10-09T00:41:01Z | — | 2, 3, 4 |
| #7 | [Validação] Garantir pipeline funcionalmente idêntico nas duas plataformas | OPEN | M2 — Implementação (semanas 3–5) | tipo:metodo, fase:implementacao | 2026-10-09T00:41:03Z | — | 5, 6 |
| #8 | [Instrumentação] Medição de energia INA219 a 10 Hz + cronometragem por GPIO | OPEN | M2 — Implementação (semanas 3–5) | tipo:metodo, fase:implementacao, hardware | 2026-10-09T00:41:06Z | — | 3 |
| #9 | [Plano B] Implementar fallback de energia por datasheet | OPEN | M2 — Implementação (semanas 3–5) | tipo:metodo, fase:implementacao, plano-b | 2026-10-09T00:41:09Z | — | 2 |
| #10 | [Experimentos] Coletar 30 repetições por plataforma (descartar 5 primeiras) | OPEN | M3 — Experimentos (semanas 6–9) | tipo:metodo, fase:experimentos, prioridade:alta | 2026-10-09T00:41:12Z | — | 7, 8 |
| #11 | [QA] Controle de validade: aquecimento, variação térmica e outliers | OPEN | M3 — Experimentos (semanas 6–9) | tipo:metodo, fase:experimentos | 2026-10-09T00:41:15Z | — | 10 |
| #12 | [Análise] Estatística descritiva (média, DP, IQR, CV), inferências/joule e teste de H1 | OPEN | M4 — Análise (semanas 10–11) | tipo:metodo, fase:analise, prioridade:alta | 2026-10-09T00:41:17Z | — | 10 |
| #13 | [Entregável] Gerar tabelas e gráficos comparativos | OPEN | M4 — Análise (semanas 10–11) | tipo:entregavel, fase:analise | 2026-10-09T00:41:20Z | — | 12 |
| #14 | [Entregável] Relatório técnico conclusivo | OPEN | M5 — Relatório (semanas 12–13) | tipo:entregavel, fase:relatorio, prioridade:alta | 2026-10-09T00:41:22Z | — | 13 |
| #15 | [Entregável] README de reprodutibilidade | OPEN | M5 — Relatório (semanas 12–13) | tipo:entregavel, fase:relatorio | 2026-10-09T00:41:25Z | — | 6, 10 |
| #16 | [Entregável] Dataset CSV bruto + publicação Zenodo (DOI) | OPEN | M6 — Vídeo e entrega (semanas 14–15) | tipo:entregavel, fase:entrega | 2026-10-09T00:41:28Z | — | 10 |
| #17 | [Entregável] Vídeo de demonstração (3–5 min) | OPEN | M6 — Vídeo e entrega (semanas 14–15) | tipo:entregavel, fase:entrega | 2026-10-09T00:41:30Z | — | 10 |
| #18 | [Pós-disciplina] Submissão a evento regional de sistemas computacionais | OPEN | M6 — Vídeo e entrega (semanas 14–15) | tipo:entregavel, fase:entrega | 2026-10-09T00:41:33Z | — | 14, 16 |
| #20 | [Dados] Expandir classes de EPI incrementalmente (começar por colete) | OPEN | — | tipo:metodo | 2026-10-13T16:55:23Z | — | 10 |

## Dependências (blocked by)

```
#4 <- 2
#5 <- 2,3,4
#6 <- 2,3,4
#7 <- 5,6
#8 <- 3
#9 <- 2
#10 <- 7,8
#11 <- 10
#12 <- 10
#13 <- 12
#14 <- 13
#15 <- 6,10
#16 <- 10
#17 <- 10
#18 <- 14,16
#20 <- 10
```
