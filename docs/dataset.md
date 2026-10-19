# Dataset — PTT6 (decisão D26, protocolo v1.3)

- **Conjunto:** Ultralytics **Construction-PPE** — https://docs.ultralytics.com/datasets/detect/construction-ppe/
- **Licença:** AGPL-3.0 (Ultralytics). Citar a fonte no relatório; não redistribuir imagens no repositório.
- **Formato:** rótulos YOLO (`classe cx cy w h`, normalizados).
- **Decisão (Filipe, issue #4, 2026-10-14):** começar com **uma classe** e depois passar às **11 classes**.
  A classe inicial é o **colete (`vest`, id 2)**, mantendo o recorte do protocolo (D10).

## Classes (ordem oficial do YAML)

| id | classe | id | classe |
|---:|---|---:|---|
| 0 | helmet | 6 | Person |
| 1 | gloves | 7 | no_helmet |
| 2 | **vest** | 8 | no_goggle |
| 3 | boots | 9 | no_gloves |
| 4 | goggles | 10 | no_boots |
| 5 | none | | |

Não existe classe `no_vest`.

## Contagens observadas (arquivo `construction-ppe.zip`, baixado em 2026-10-18)

| split | imagens | instâncias `vest` | imagens com `vest` |
|---|---:|---:|---:|
| train | 1142 | 1283 | 837 |
| val | 143 | 171 | 109 |
| test | 141 | 178 | 111 |

> A documentação oficial informa 1.132 imagens de treino; o arquivo baixado contém 1.142 rótulos em
> `labels/train`. Registrar o hash do zip usado ao preparar os dados.

Classe mais rara: `no_boots` (88 instâncias no train, 4 no val, 23 no test) — relevante para o bloco B.

## Uso no experimento

| Uso | Split | Regra |
|---|---|---|
| Treino (D27) | train | bloco A: rótulos filtrados para `vest` (demais classes removidas dos .txt) |
| Calibração int8 (D17) | train | 750 imagens estratificadas por classe; lista versionada |
| Acurácia E0 | test (141) | mAP@50 e mAP@50-95, fp32 e int8 |
| Entradas E1–E4 | test | `M = 50` imagens (bloco A: com ao menos um `vest`), lista fixa versionada |

## Pré-processamento (idêntico nas duas plataformas — D11/D30)

1. Letterbox para `r×r` (resolução comum, D31) com preenchimento cinza 114, mantendo a proporção.
2. RGB, uint8, layout HWC; **sem normalização** (YOLOX usa 0..255).
3. Gravado como `.bin` bruto (`r·r·3` bytes) com SHA-256 registrado; o mesmo arquivo vai para o
   SBC (disco) e para o MCU (flash/SD).
4. No dispositivo, `pre` = cópia para o tensor de entrada + quantização com `scale`/`zero_point` do modelo.
