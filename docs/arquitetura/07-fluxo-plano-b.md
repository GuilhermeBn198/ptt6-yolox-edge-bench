# 07 — Fluxo do Plano B

> Árvore de decisão das contingências (protocolo §10 e §11). Regra de ouro: **nunca bloquear a entrega**
> — quando um risco se materializa, aciona-se o caminho alternativo **declarando a limitação**.

## Decisão: instrumento de energia

```mermaid
flowchart TB
  A["Inicio: medir energia<br/>por inferencia"]:::proc
  B{"INA219 disponivel<br/>e confiavel?"}:::dec
  C["Usar INA219 na taxa maxima<br/>janela por GPIO, E = integral P dt"]:::ok
  D["Calibrar contra carga conhecida<br/>quantificar erro"]:::proc
  E{"Erro compromete<br/>a comparacao?"}:::dec
  F["Manter INA219<br/>erro = limitacao declarada"]:::ok
  G["Plano B: energia =<br/>potencia do datasheet x tempo medido"]:::planb
  H["Declarar limitacao<br/>no relatorio e README"]:::proc
  I["Registrar no CSV<br/>flag de proxy"]:::proc
  J["Upgrade opcional:<br/>PPK2 / Joulescope"]:::planb

  A --> B
  B -->|sim| C --> D --> E
  B -->|"nao (indisponivel/defeituoso)"| G
  E -->|nao| F
  E -->|sim| J
  G --> H --> I
  I --> K["Seguir protocolo estatistico<br/>piloto -> N (piso 30)"]:::proc
  F --> K
  J --> K
  K --> L["Comparacao MCU vs SBC<br/>permanece valida"]:::ok

  classDef proc fill:#dae8fc,stroke:#6c8ebf,color:#000;
  classDef dec fill:#ffe6cc,stroke:#d79b00,color:#000;
  classDef ok fill:#d5e8d4,stroke:#82b366,color:#000;
  classDef planb fill:#f8cecc,stroke:#b85450,color:#000;
```

## Decisão: modelo não performa em uma plataforma

```mermaid
flowchart TB
  A["YOLOX-Nano int8<br/>nao roda ou roda lento demais"]:::proc
  B{"Roda nas duas<br/>plataformas?"}:::dec
  C["Seguir o benchmark"]:::ok
  D["Plano B: manter o runtime TFLite int8<br/>e ajustar entrada (ex. 320x320)<br/>IGUAL nas duas plataformas"]:::planb
  E["Plano B alternativo: outra versao<br/>quantizada do YOLO"]:::planb
  F["Registrar mudanca + novo hash<br/>e a ameaca a comparabilidade"]:::proc

  A --> B
  B -->|sim| C
  B -->|nao| D
  D -->|"ainda inviavel"| E
  D --> F
  E --> F

  classDef proc fill:#dae8fc,stroke:#6c8ebf,color:#000;
  classDef dec fill:#ffe6cc,stroke:#d79b00,color:#000;
  classDef ok fill:#d5e8d4,stroke:#82b366,color:#000;
  classDef planb fill:#f8cecc,stroke:#b85450,color:#000;
```

## Matriz de riscos → resposta

| Risco | Prob. | Impacto | Resposta | Onde registrar |
|---|---|---|---|---|
| Medidor físico indisponível/defeituoso | média | alto | datasheet × tempo medido + limitação | CSV (flag), relatório, README |
| Modelo não performa em um dispositivo | média | alto | ajuste de entrada idêntico ou outra versão quantizada | relatório, artefato (hash) |
| Throttling térmico (SBC) | média | alto | governor `performance`, heatsink, pausas, registrar temp. | notas de coleta (#11) |
| PM/DFS da MCU alterando medição | média | alto | desabilitar PM, travar 240 MHz, registrar `sdkconfig` | `sdkconfig`, relatório |
| Sem baseline publicado do YOLOX-Nano | alta | médio | rigor de reprodutibilidade + exemplos análogos + dataset como contribuição | relatório (D16) |
