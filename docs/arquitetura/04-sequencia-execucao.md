# 04 — Sequência de Execução

> Como **uma repetição medida** acontece no tempo, do ponto de vista dos componentes.
> A **janela de energia** é delimitada por **bordas de GPIO** e as **três fases** (pré / inferência /
> pós) são separadas para permitir reanálise (protocolo §5 e §6).

## Sequência de uma repetição

```mermaid
sequenceDiagram
  autonumber
  actor OP as Operador
  participant CAM as Camera
  participant APP as Pipeline (pre/inf/pos)
  participant GPIO as Linha GPIO
  participant INA as INA219 (~1,9 kHz)
  participant RAW as data/raw (CSV)

  OP->>APP: iniciar execucao i
  activate APP
  APP->>GPIO: borda t0 (inicio da janela)
  GPIO->>INA: abre janela de energia

  rect rgb(238,246,255)
  Note over APP,CAM: Fase 1 - pre-processamento
  APP->>CAM: captura frame
  CAM-->>APP: frame (resize + normalizacao)
  APP->>GPIO: marca t1 (fim do pre)
  end

  rect rgb(213,232,212)
  Note over APP: Fase 2 - inferencia
  APP->>APP: invoke() do modelo int8
  APP->>GPIO: marca t2 (fim da inferencia)
  end

  rect rgb(255,242,204)
  Note over APP: Fase 3 - pos-processamento
  APP->>APP: decoding + NMS
  APP->>GPIO: borda t3 (fim da janela)
  end

  GPIO->>INA: fecha janela
  INA-->>RAW: E = integral de P(t) dt
  APP-->>RAW: latencias (pre, inf, pos, total)
  deactivate APP
  Note over RAW: 1 linha por execucao
```

## Linha do tempo da janela

```
 t0                 t1                      t2                 t3
  |----- pre --------|------ inferencia -----|----- pos --------|
  |<-------------------- janela de energia ------------------->|
```

- **Latência** reportada: `pre + inferência + pós` (D5) — e **também** as 3 fases separadas.
- **Energia**: `E = ∫ P(t)·dt` integrada em `[t0, t3]` (inclui o consumo parado — D6).
- **Duas bordas GPIO** delimitam a janela; marcas internas (`t1`, `t2`) separam as fases.

## Execução em lote (a sessão completa)

```mermaid
sequenceDiagram
  autonumber
  actor OP as Operador
  participant PLAT as Plataforma (MCU ou SBC)
  participant HOST as Host (coleta)

  loop Warm-up (definido pelo dado; piso 5)
    OP->>PLAT: repeticao de aquecimento
    PLAT-->>HOST: log (nao entra na analise)
  end
  Note over OP,HOST: checar estabilidade (variacao abaixo de 2%) e temperatura
  loop N repetições válidas (piso 30)
    OP->>PLAT: repeticao valida (janela medida)
    PLAT-->>HOST: latencia + energia -> CSV
  end
  Note over OP,HOST: pausa de resfriamento entre baterias / troca de plataforma
```

**Pontos de controle:** aquecimento antes de contar (D22), **N** definido por piloto/CV (D19),
pausas de resfriamento e registro de temperatura (ameaça térmica).
