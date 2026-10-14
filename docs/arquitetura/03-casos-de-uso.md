# 03 — Casos de Uso

> O Mermaid **não** tem um tipo nativo de diagrama de casos de uso UML; usamos `flowchart` com
> **atores** (fora do retângulo) e a **fronteira do sistema** (subgrafo). O mapeamento para issues
> aparece na tabela final.

## Diagrama de casos de uso

```mermaid
flowchart LR
  pesquisador(("Pesquisador /<br/>Operador")):::actor
  orientador(("Orientador /<br/>Banca")):::actor
  comunidade(("Comunidade")):::actor

  mcu(("MCU<br/>ESP32 LX6")):::actor2
  sbc(("SBC<br/>RPi Zero 2 W")):::actor2
  inst(("INA219 +<br/>GPIO")):::actor2
  ds(("Dataset de<br/>EPIs")):::actor2

  subgraph SYS["Fronteira do sistema — Benchmark PTT6"]
    direction TB
    uc1["UC1 Selecionar dataset<br/>e definir classes"]
    uc2["UC2 Preparar modelo<br/>YOLOX-Nano int8"]
    uc3["UC3 Garantir equivalencia<br/>funcional"]
    uc4["UC4 Executar inferencia<br/>na MCU"]
    uc5["UC5 Executar inferencia<br/>no SBC"]
    uc6["UC6 Medir latencia<br/>e energia"]
    uc7["UC7 Coletar N repeticoes<br/>+ warm-up"]
    uc8["UC8 Controlar validade<br/>(termica, outliers)"]
    uc9["UC9 Analisar e<br/>testar H1"]
    uc10["UC10 Gerar tabelas<br/>e graficos"]
    uc11["UC11 Redigir<br/>relatorio tecnico"]
    uc12["UC12 Publicar dataset<br/>(DOI Zenodo)"]
    uc13["UC13 Gravar video<br/>de demonstracao"]
    uc14["UC14 Reproduzir<br/>o experimento"]
  end

  pesquisador --- uc1
  pesquisador --- uc2
  pesquisador --- uc3
  pesquisador --- uc7
  pesquisador --- uc8
  pesquisador --- uc9
  pesquisador --- uc10
  pesquisador --- uc11
  pesquisador --- uc12
  pesquisador --- uc13

  ds --- uc1
  mcu --- uc4
  sbc --- uc5
  mcu --- uc6
  sbc --- uc6
  inst --- uc6

  uc4 --- uc7
  uc5 --- uc7
  uc6 --- uc7
  uc3 --- uc4
  uc3 --- uc5
  uc2 --- uc4
  uc2 --- uc5
  uc9 --- uc10
  uc10 --- uc11

  orientador --- uc11
  orientador --- uc9
  comunidade --- uc12
  comunidade --- uc14
  uc11 --> uc14
  uc12 --> uc14

  classDef actor fill:#dae8fc,stroke:#6c8ebf,color:#000;
  classDef actor2 fill:#ffe6cc,stroke:#d79b00,color:#000;
```

## Catálogo de casos de uso

| ID | Caso de uso | Ator primário | Atores de apoio | Issue | Entregável |
|----|-------------|---------------|-----------------|-------|------------|
| UC1 | Selecionar dataset e definir classes | Pesquisador | Dataset EPIs | #4, #20 | `docs/dataset.md` |
| UC2 | Preparar modelo YOLOX-Nano int8 | Pesquisador | — | #2, #5, #6 | Artefato `.tflite` + hash |
| UC3 | Garantir equivalência funcional | Pesquisador | MCU, SBC | #7 | Evidência em `scripts/` |
| UC4 | Executar inferência na MCU | Pesquisador | MCU (LX6) | #5 | Log + latência |
| UC5 | Executar inferência no SBC | Pesquisador | SBC | #6 | Log + latência |
| UC6 | Medir latência e energia | Pesquisador | INA219 + GPIO | #8 | `E`, `t` por execução |
| UC7 | Coletar N repetições + warm-up | Pesquisador | MCU, SBC | #10 | `data/raw/*.csv` |
| UC8 | Controlar validade (térmica, outliers) | Pesquisador | — | #11 | Notas de coleta |
| UC9 | Analisar e testar H1 | Pesquisador | — | #12 | Notebook de análise |
| UC10 | Gerar tabelas e gráficos | Pesquisador | — | #13 | `report/figures/` |
| UC11 | Redigir relatório técnico | Pesquisador | Orientador | #14 | Relatório (produto principal) |
| UC12 | Publicar dataset (DOI Zenodo) | Pesquisador | Comunidade | #16 | DOI citável |
| UC13 | Gravar vídeo de demonstração | Pesquisador | — | #17 | Vídeo 3–5 min |
| UC14 | Reproduzir o experimento | Comunidade | — | #15 | README de reprodutibilidade |

**Observação:** UC4/UC5 são o **mesmo caso de uso** em plataformas distintas — a diferença é o
hardware/runtime de destino, não a lógica. Isso é o que torna a comparação válida (regra da
comparabilidade).
