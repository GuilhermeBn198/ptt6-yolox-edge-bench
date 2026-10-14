# 01 — Contexto e Containers (C4)

> Diagramas de **contexto (nível 1)** e **containers (nível 2)** do benchmark.

## Nível 1 — Contexto do sistema

Quem usa o benchmark e de quais sistemas externos ele depende.

```mermaid
flowchart LR
  pesquisador["Pesquisador / Operador<br/>(Filipe, Guilherme)"]:::person
  orientador["Orientador / Banca"]:::person
  comunidade["Comunidade (reuso)"]:::person

  sistema["Benchmark YOLOX-Nano int8<br/>PTT6 — Edge AI"]:::sys

  github["GitHub<br/>repositório + issues + board"]:::ext
  dataset["Dataset de EPIs<br/>Construction-PPE"]:::ext
  zenodo["Zenodo<br/>DOI do dataset"]:::ext
  video["Plataforma de vídeo<br/>demonstração"]:::ext

  pesquisador -->|"projeta, opera, mede, analisa"| sistema
  sistema -->|"relatório, figuras, dados"| orientador
  orientador -->|"avalia / orienta"| pesquisador
  dataset -->|"imagens (calibração, validação)"| sistema
  sistema -->|"código, dados, docs"| github
  sistema -->|"CSV bruto + metadados"| zenodo
  zenodo -->|"DOI citável"| comunidade
  sistema -->|"vídeo 3-5 min"| video
  github -.->|"reprodutibilidade"| comunidade

  classDef person fill:#dae8fc,stroke:#6c8ebf,color:#000;
  classDef sys fill:#d5e8d4,stroke:#82b366,color:#000;
  classDef ext fill:#f5f5f5,stroke:#999,color:#000;
```

**Leitura:** o sistema é um **artefato de medição** — não presta serviço a usuário final em tempo real;
seu produto é **evidência** (relatório, dataset, vídeo) para o orientador e para a comunidade.

## Nível 2 — Containers

Como o sistema é decomposto por responsabilidade (o que roda à noite, o que roda na borda).

```mermaid
flowchart TB
  subgraph HOST["Host de desenvolvimento"]
    direction TB
    prep["scripts/<br/>seleção + pré-processamento<br/>do dataset"]
    convert["Artefato de modelo<br/>YOLOX-ti-lite -> .tflite int8<br/>(+ hash)"]
    analyze["analysis/<br/>notebook + scripts de<br/>estatística e gráficos"]
    raw[("data/raw/<br/>CSV bruto das execucoes")]
    figs["report/figures/<br/>tabelas e graficos"]
  end

  subgraph EDGE["Plataformas de borda (mesmo artefato, mesmo pipeline)"]
    direction TB
    fw["Firmware MCU<br/>ESP32 LX6 - TFLite Micro"]
    sbc["Aplicacao SBC<br/>RPi Zero 2 W - TFLite"]
    cam["Camera ao vivo"]
  end

  subgraph INST["Instrumentacao de medição"]
    direction TB
    ina["INA219<br/>taxa maxima ~1,9 kHz<br/>E = integral P(t)dt"]
    gpio["Captura por GPIO<br/>2 bordas (pre / inf / pos)"]
  end

  prep --> convert
  convert -->|"mesmo arquivo + hash"| fw
  convert -->|"mesmo arquivo + hash"| sbc
  cam --> fw
  cam --> sbc
  fw -->|"latencia + log"| raw
  sbc -->|"latencia + log"| raw
  ina -->|"P(t)"| gpio
  gpio -->|"energia por janela"| raw
  raw --> analyze
  analyze --> figs

  classDef store fill:#f8cecc,stroke:#b85450,color:#000;
  class raw store;
```

**Regra da comparabilidade (protocolo §3):** o container `Artefato de modelo` é **único** e alimenta as
duas plataformas; o pipeline de `prep` + `fw`/`sbc` deve ser funcionalmente idêntico.
