# PTT6 — Benchmarking de Inferência do YOLO Quantizado em MCU e SBC

> Desempenho e eficiência energética de **YOLOX-Nano (int8)** para classificação de **EPIs** em cenários de **Edge AI**.

**Discentes**: Filipe Matias & Guilherme Lucas
**Orientador**: Herbert Oliveira Rocha
**Programa**: Mestrado Profissional em Computação Aplicada (PPCA/UFRR)
**Disciplina**: Arquitetura e Organização de Computadores
**Trilha**: PTT6 — Benchmarking e eficiência energética

---

## Sobre

Este repositório contém o experimento comparativo de inferência de um único modelo
(YOLOX-Nano, quantizado em **int8**) executado em duas plataformas representativas do
espectro Edge AI:

| Plataforma | Tipo | Papel no experimento |
|------------|------|----------------------|
| **ESP32-S3** | MCU | Extremo de baixo consumo |
| **Raspberry Pi Zero 2 W** | SBC | Extremo de maior throughput/recursos |

**Pergunta de pesquisa**: como se comparam latência, throughput e energia por inferência
entre a MCU e o SBC ao executar o mesmo pipeline?

**Hipótese H1**: o SBC apresenta throughput maior, mas energia por inferência igual ou
superior, dado seu consumo estático mais alto.

## Métricas

- **Latência** de inferência (ms)
- **Throughput** (inferências/s)
- **Energia por inferência** (mJ) e eficiência normalizada (**inferências/joule**)

## Protocolo experimental (resumo)

- **30 execuções por plataforma**, descartando as **5 primeiras** (aquecimento)
- Medidor de potência inline **INA219** amostrando a **10 Hz**
- Cronometragem via **toggle de GPIO**
- Dispersão: desvio-padrão, IQR e coeficiente de variação
- Ameaça principal: **variação térmica** — mitigada com pausas de resfriamento entre
  baterias de execução e ambiente refrigerado
- **Plano B de energia**: estimar por datasheet × tempo medido, declarando a limitação

## Estrutura do repositório

```
firmware/esp32s3/     # Pipeline de inferência no ESP32-S3 (TFLite Micro)
raspberry/pi_zero_2w/ # Pipeline de inferência no Raspberry Pi Zero 2 W
hardware/ina219/      # Esquema e código de medição de energia (INA219)
data/raw/             # Dados brutos das execuções (CSV)
data/processed/       # Dados agregados/derivados
analysis/notebooks/   # Análise estatística e gráficos
report/figures/       # Figuras do relatório
video/                # Link/roteiro do vídeo de demonstração (3–5 min)
docs/proposta/        # Proposta assinada do projeto (PTT6)
.opencode/skills/     # Skill do agente com a base de conhecimento do projeto
```

## Reprodutibilidade

> A ser preenchido na fase de implementação (semanas 3–5).

1. Requisitos de hardware e software por plataforma
2. Passos de build/flash e execução
3. Como reproduzir a coleta e a análise a partir de `data/raw/`

## Entregáveis

- [ ] Repositório versionado com licença
- [ ] README de reprodutibilidade
- [ ] Dados brutos (CSV das execuções)
- [ ] Vídeo de demonstração (3–5 min) nas duas plataformas
- [ ] Relatório técnico conclusivo com tabelas e gráficos
- [ ] (pós-disciplina) Dataset com DOI no Zenodo + submissão a evento regional

## Plano de ação

O plano é acompanhado por **issues e milestones** no GitHub, espelhando as fases do
cronograma (15 semanas). Veja a aba [Issues](../../issues).

| Semanas | Fase |
|---------|------|
| 1–2 | Revisão e protocolo |
| 3–5 | Implementação do pipeline |
| 6–9 | Execução dos experimentos |
| 10–11 | Análise dos dados |
| 12–13 | Redação do relatório |
| 14–15 | Vídeo e entrega final |

## Base de conhecimento (skill do agente)

A proposta foi convertida em uma skill reutilizável em
[`.opencode/skills/ptt6-yolox-edge-bench/`](.opencode/skills/ptt6-yolox-edge-bench/SKILL.md),
cobrindo pergunta de pesquisa, hipótese, protocolo, riscos/Plano B, entregáveis e
um cheatsheet de decisão.

## Referências preliminares

- Patterson, D. A.; Hennessy, J. L. *Computer Organization and Design*.
- Datasheets: ESP32; Raspberry Pi Zero 2 W.
- YOLOX — https://github.com/Megvii-BaseDetection/YOLOX

## Licença

MIT — veja [LICENSE](LICENSE). Os dados publicados seguem CC-BY-4.0.
