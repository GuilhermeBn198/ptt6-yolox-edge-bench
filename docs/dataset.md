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

> A documentação oficial informa 1.132 imagens de treino; o zip contém 1.132 imagens e 1.142 arquivos de
> rótulo em `labels/train`.

**Arquivo de referência (não versionar as imagens):** `construction-ppe.zip` oficial da Ultralytics
(`https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip`), 178.415.813 bytes,
SHA-256 `bef8dcb599aa4e9d9f5e602cb6fa7143d3c84d7f6a0ff40463d7f2a4c2632ccc`. A cópia local usada pelo Filipe no treino é idêntica (mesmo hash, conferido em 2026-10-22).
Basta baixar o zip e conferir o hash; o E0 e a calibração usam esse mesmo arquivo.

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
2. BGR, uint8, layout HWC; **sem normalização** (YOLOX usa 0..255).
3. Gravado como `.bin` bruto (`r·r·3` bytes) com SHA-256 registrado; o mesmo arquivo vai para o
   SBC (disco) e para o MCU (flash/SD).
4. No dispositivo, `pre` = cópia para o tensor de entrada + quantização com `scale`/`zero_point` do modelo.
