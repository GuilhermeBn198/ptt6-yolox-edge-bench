# PTT6 — Benchmarking de Inferência do YOLO Quantizado em MCU e SBC

> Desempenho e eficiência energética de **YOLOX-Nano (int8)** para **detecção de EPIs** em cenários de **Edge AI**.

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
| **ESP32 (LX6)** | MCU | Extremo de baixo consumo (hardware disponível; ver emenda v1.1) |
| **Raspberry Pi Zero 2 W** | SBC | Extremo de maior throughput/recursos |

**Pergunta de pesquisa**: como se comparam latência, throughput e energia por inferência
entre a MCU e o SBC ao executar o mesmo pipeline?

**Hipótese H1**: o SBC apresenta throughput maior, mas energia por inferência igual ou
superior, dado seu consumo estático mais alto.

> **Nota de plataforma:** a proposta original citava o **ESP32-S3**; adotou-se o **ESP32 (LX6)**
> disponível, com aval do orientador (emenda **v1.1**, decisão D23). Ver
> [`docs/protocolo.md`](docs/protocolo.md) §13.

## Métricas

- **Latência** de inferência (ms), de ponta a ponta (pré + inferência + pós)
- **Throughput** (inferências/s)
- **Energia por inferência** (mJ, **estimada por software**) e eficiência normalizada
  (**inferências/joule** e **EDP**)
- **Acurácia**: mAP@50 / mAP@50-95

## Protocolo experimental (resumo)

> Detalhes e justificativas em [`docs/protocolo.md`](docs/protocolo.md) (protocolo congelado + emendas v1.1/v1.2/v1.3).

- **Dataset:** Ultralytics **Construction-PPE** — começa com 1 classe (colete) e depois 11 classes ([`docs/dataset.md`](docs/dataset.md))
- **Desenho experimental e coleta de métricas (v1.3):** [`docs/experimentacao.md`](docs/experimentacao.md)
- Tarefa de **detecção** (bounding boxes), mesmo artefato **YOLOX-Nano int8 → TFLite** (mesmo
  runtime e mesmo pipeline nas duas plataformas; hash do modelo registrado)
- **Execução em fases**: validar o pipeline com **imagens estáticas** (sem câmera) e só depois
  integrar a **câmera ao vivo** (decisão D25)
- Repetições definidas por **piloto** (CV → N para IC 95%; **piso 30**); **warm-up definido pelo dado**
- **Energia estimada por software**: modelo de potência por modo (datasheet) × tempo medido por fase
  (`E = Σ P_modo · t_fase`); **sem medidor físico inline** (decisão D24) — limitação declarada
- Estatística robusta: mediana + IQR, **Mann-Whitney U** + tamanho de efeito + IC por bootstrap
- Ameaça principal: **variação térmica** — mitigada com pausas de resfriamento e ambiente registrado
- **[Arquitetura e diagramas](docs/arquitetura/README.md)**: contexto/C4, fluxo de dados (DFD),
  casos de uso, sequência de execução, fases, implantação e modelo de dados

## Estrutura do repositório

```
firmware/esp32/       # Pipeline de inferência no ESP32 (LX6) (TFLite Micro)
raspberry/pi_zero_2w/ # Pipeline de inferência no Raspberry Pi Zero 2 W
hardware/ina219/      # (histórico) medição física descartada em D24 — energia é estimada por software
scripts/              # energy_model.py + power_model.toml (energia por software)
data/raw/             # Dados brutos das execuções (CSV)
data/processed/       # Dados agregados/derivados
analysis/notebooks/   # Análise estatística e gráficos
report/figures/       # Figuras do relatório
video/                # Link/roteiro do vídeo de demonstração (3–5 min)
docs/                 # Proposta, protocolo, experimentação, dataset, plano de ação
docs/projeto/         # Projeto em LaTeX (main.tex → PDF)
docs/arquitetura/     # Diagramas de arquitetura (C4, DFD, casos de uso, sequência)
.opencode/skills/     # Skill do agente com a base de conhecimento do projeto
```

> Observação: a pasta `firmware/esp32s3/` existe no repositório (estrutura inicial); ao implementar,
> ela passa a conter o pipeline da plataforma **ESP32 (LX6)** adotada na emenda v1.1.

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
