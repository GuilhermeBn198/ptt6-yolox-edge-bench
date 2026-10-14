# 05 — Implantação e Hardware

> Como o **INA219** e as **linhas de GPIO** se conectam à plataforma sob teste (DUT) e ao host.
> O esquema é o mesmo para as duas plataformas (fonte regulada + sensor inline).

## Esquema de alimentação e medição

```mermaid
flowchart LR
  rede["Rede eletrica"]:::ext
  fonte["Fonte regulada<br/>(tensao fixa, baixo ripple)"]:::hw
  shunt["Shunt (alto lado)<br/>R conhecido, ex. 0,1 ohm"]:::hw
  ina["INA219<br/>V_bus + V_shunt -> P(t)<br/>I2C, taxa max ~1,9 kHz"]:::sensor
  dut["Plataforma sob teste (DUT)<br/>ESP32 LX6  OU  RPi Zero 2 W"]:::dut
  cam["Camera ao vivo"]:::ext
  host["Host de coleta<br/>(logger I2C + GPIO)"]:::host
  csv[("data/raw<br/>CSV bruto")]:::store

  rede --> fonte --> shunt
  shunt -->|"V_shunt"| ina
  shunt -->|"alimentacao"| dut
  ina -->|"P(t) via I2C"| host
  dut -->|"GPIO: t0..t3"| host
  dut --- cam
  host --> csv

  classDef ext fill:#f5f5f5,stroke:#999,color:#000;
  classDef hw fill:#dae8fc,stroke:#6c8ebf,color:#000;
  classDef sensor fill:#ffe6cc,stroke:#d79b00,color:#000;
  classDef dut fill:#d5e8d4,stroke:#82b366,color:#000;
  classDef host fill:#e1d5e7,stroke:#9673a6,color:#000;
  classDef store fill:#f8cecc,stroke:#b85450,color:#000;
```

## Topologia de conexões

| Bloco | Papel | Observação de rigor |
|---|---|---|
| Fonte regulada | tensão estável | registrar tensão; verificar estabilidade (issue #8) |
| Shunt + INA219 | medir `V` e `I` inline | INA219 mede **V e I em instantes diferentes** → erro declarado como limitação |
| Linha GPIO (t0..t3) | delimitar a janela e as fases | **sem overhead de software** na cronometragem |
| DUT | ESP32 LX6 **ou** RPi Zero 2 W | medir **uma plataforma por vez**, mesmo esquema |
| Host de coleta | agregar `P(t)` e latências | timestamp comum; uma linha por execução em `data/raw` |
| Câmera | entrada ao vivo | fixar resolução/formato/buffers iguais nas duas pontas |

## Diagrama de implantação (nós físicos)

```mermaid
flowchart TB
  subgraph BANCADA["Bancada de medição (ambiente registrado)"]
    subgraph ALIM["Alimentação"]
      f2["Fonte regulada"]:::hw
      sh2["Shunt + INA219"]:::sensor
    end
    subgraph PLAT["Plataforma ativa (uma por vez)"]
      m["ESP32 LX6 (firmware TFLite Micro)"]:::dut
      s["RPi Zero 2 W (TFLite, governor performance)"]:::dut
    end
    host2["Host de coleta / orquestracao"]:::host
  end
  f2 --> sh2
  sh2 --> m
  sh2 --> s
  m -->|"GPIO + I2C"| host2
  s -->|"GPIO + I2C"| host2

  classDef hw fill:#dae8fc,stroke:#6c8ebf,color:#000;
  classDef sensor fill:#ffe6cc,stroke:#d79b00,color:#000;
  classDef dut fill:#d5e8d4,stroke:#82b366,color:#000;
  classDef host fill:#e1d5e7,stroke:#9673a6,color:#000;
```

> **Plano B desta visão:** se o INA219 não estiver disponível/confiável, o bloco `Shunt + INA219` é
> substituído por **estimativa por datasheet × tempo medido** (ver [07 — Fluxo do Plano B](07-fluxo-plano-b.md)),
> mantendo fonte, GPIO e host.
