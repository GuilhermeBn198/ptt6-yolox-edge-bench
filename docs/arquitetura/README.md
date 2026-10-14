# Arquitetura do Benchmark — PTT6

> Mapeamento de arquitetura do experimento (YOLOX-Nano int8: ESP32 (LX6) vs Raspberry Pi Zero 2 W)
> em **diagramas versionados**. Ferramenta: **Mermaid** (renderiza nativamente no GitHub, sem servidor).
>
> Fonte de verdade do comportamento: [`../protocolo.md`](../protocolo.md) (protocolo **v1.0 congelado**).
> Contexto da proposta: [`../../.opencode/skills/ptt6-yolox-edge-bench/`](../../.opencode/skills/ptt6-yolox-edge-bench/SKILL.md).

## Índice de diagramas

| # | Documento | Tipo de diagrama | Responde a |
|---|-----------|------------------|------------|
| 01 | [Contexto e containers](01-contexto-e-containers.md) | C4 (níveis 1–2) | Quem usa e de que o sistema é feito? |
| 02 | [Fluxo de dados](02-fluxo-de-dados.md) | DFD (níveis 0 e 1) | Como o dado flui da câmera até a análise? |
| 03 | [Casos de uso](03-casos-de-uso.md) | Casos de uso (UML, via flowchart) | O que cada ator faz no experimento? |
| 04 | [Sequência de execução](04-sequencia-execucao.md) | Sequência (UML) | Como uma medição acontece no tempo? |
| 05 | [Implantação e hardware](05-implantacao-hardware.md) | Implantação + esquema elétrico | Como o INA219 e o GPIO se conectam? |
| 06 | [Modelo de dados](06-modelo-de-dados.md) | ER + dicionário | Qual o formato do CSV bruto? |
| 07 | [Fluxo do Plano B](07-fluxo-plano-b.md) | Fluxo de decisão | O que fazer se o medidor falhar? |

## Convenções

- **Linguagem ubíqua:** plataforma = `MCU` (ESP32 LX6) ou `SBC` (RPi Zero 2 W); *artefato* = um único
  `.tflite` int8 compartilhado; *execução/repetição* = uma inferência medida; *janela* = intervalo
  delimitado por **duas bordas de GPIO** que contém **pré + inferência + pós**.
- **Regra da comparabilidade:** o mesmo artefato e a mesma lógica de pipeline rodam nas duas pontas
  (protocolo §3). O que difere é apenas o hardware/runtime de destino.
- **Rastreabilidade:** todo número do relatório deriva de `data/raw/*.csv`; todo diagrama deriva do
  protocolo. Mudou o protocolo → abre-se nova versão e revisa-se o diagrama.

## Como renderizar

- **No GitHub:** os blocos ` ```mermaid ` são renderizados automaticamente.
- **Local (VS Code):** extensão *Markdown Preview Mermaid Support*.
- **Exportar imagem:** [mermaid.live](https://mermaid.live) (colar o bloco) para PNG/SVG em `report/figures/`.

## Manutenção

Os diagramas são **documentação viva**: qualquer decisão nova (ex.: troca de plataforma MCU, novo
instrumento de energia, nova classe de EPI) exige atualizar o protocolo e o diagrama afetado na
mesma PR.
