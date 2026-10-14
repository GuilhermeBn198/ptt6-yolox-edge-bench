# 02 — Fluxo de Dados (DFD)

> Diagrama de Fluxo de Dados em dois níveis: **nível 0** (contexto) e **nível 1**
> (decomposição dos processos). Notação: `P` = processo, `D` = depósito de dados,
> `E` = entidade externa.

## Nível 0 — Panorama

```mermaid
flowchart LR
  E1["E1 Pesquisador / Operador"]:::ext
  E2["E2 Camera ao vivo"]:::ext
  E3["E3 Fonte + INA219"]:::ext
  E4["E4 Dataset de EPIs"]:::ext
  E5["E5 Consumidor<br/>relatorio / dataset / banca"]:::ext

  P0(("P0<br/>Benchmark de<br/>inferencia int8"))

  E4 -->|"imagens"| P0
  E2 -->|"frames"| P0
  E3 -->|"V(t), I(t), GPIO"| P0
  E1 -->|"parametros de execucao"| P0
  P0 -->|"latencia, energia, mAP"| E5

  classDef ext fill:#f5f5f5,stroke:#999,color:#000;
```

## Nível 1 — Decomposição

```mermaid
flowchart TB
  E1["E1 Pesquisador"]:::ext
  E4["E4 Dataset de EPIs"]:::ext
  E2["E2 Camera"]:::ext
  E3["E3 INA219 + GPIO"]:::ext

  P1["P1 Selecionar e padronizar<br/>dataset"]
  P2["P2 Converter/quantizar int8<br/>+ calibrar (750 imgs)"]
  P3["P3 Verificar equivalencia<br/>funcional"]
  P4["P4 Executar inferencia<br/>na MCU (ESP32 LX6)"]
  P5["P5 Executar inferencia<br/>no SBC (RPi Zero 2 W)"]
  P6["P6 Medir tempo (2 bordas GPIO)<br/>e energia (integral P dt)"]
  P7["P7 Consolidar CSV bruto"]
  P8["P8 Analise estatistica<br/>+ teste de H1"]
  P9["P9 Gerar tabelas e graficos"]

  D1[("D1 dataset preparado")]
  D2[("D2 modelo .tflite int8 + hash")]
  D3[("D3 data/raw (CSV bruto)")]
  D4[("D4 report/figures")]

  E4 --> P1 --> D1 --> P2
  E1 --> P2
  P2 --> D2
  D2 --> P3
  E2 --> P3
  P3 -->|"ok"| P4
  P3 -->|"ok"| P5
  E2 --> P4
  E2 --> P5
  D2 --> P4
  D2 --> P5
  P4 --> P6
  P5 --> P6
  E3 --> P6
  P6 --> P7 --> D3 --> P8 --> P9 --> D4
  E1 --> P7
  E1 --> P8

  classDef ext fill:#f5f5f5,stroke:#999,color:#000;
  classDef store fill:#f8cecc,stroke:#b85450,color:#000;
  class D1,D2,D3,D4 store;
```

**Notas de rigor:**
- **P2** produz um **único** artefato usado nas duas plataformas (hash registrado) — isola o hardware.
- **P3** é obrigatório: sem equivalência funcional a comparação é inválida (protocolo §3, issue #7).
- **P6** é a fronteira da medição: a janela de energia é delimitada pelas **duas bordas de GPIO**, que
  também separam as três fases (pré / inferência / pós).
- **P7** preserva o **bruto** (fonte da verdade); **P8/P9** só derivam dele.
