# Desenho Experimental e Coleta de Métricas — PTT6 (protocolo v1.4)

> Detalha **como** os experimentos são executados e **o que** é medido em cada um. É o documento
> operacional da emenda **v1.3** do [`protocolo.md`](protocolo.md) (decisões D26–D32). Em caso de
> conflito, vale o protocolo; este arquivo o operacionaliza.
>
> **v1.4 (2026-10-19, D33):** câmera ao vivo é **extra**; validação e medição com **ingestão direta de
> imagens** no modelo (serial no ESP32, disco no RPi) — ver §5.4.
>
> Estado em 2026-10-18: hardware em mãos (ESP32 LX6 + RPi Zero 2 W); dataset definido
> (Construction-PPE, #4); **modelo já executa no Raspberry Pi** (#6); INA219 **não será usado** (D24).

---

## 1. Visão geral

Um experimento comparativo com **um fator principal** (plataforma: MCU × SBC), executado em
**fases** (D25) e repetido para **duas configurações de classes** (D26):

| Bloco | Configuração do modelo | Quando |
|---|---|---|
| **A** | YOLOX-Nano int8, **1 classe** (`vest`) | primeiro — escopo mínimo |
| **B** | YOLOX-Nano int8, **11 classes** (Construction-PPE completo) | após A concluído |

Cada bloco percorre os mesmos experimentos E0–E6 (§4). Os blocos **não** são comparados entre si
estatisticamente como fator; B é uma replicação ampliada (efeito do nº de classes reportado de forma
descritiva).

## 2. Variáveis

### 2.1 Independente (fator principal)

| Variável | Níveis |
|---|---|
| Plataforma | `MCU` = ESP32 (LX6) @240 MHz · `SBC` = Raspberry Pi Zero 2 W @1 GHz |

### 2.2 Dependentes (medidas por inferência)

| Variável | Unidade | Como é obtida |
|---|---|---|
| `t_pre` | µs | timestamp monotônico, início → fim do pré-processamento |
| `t_inf` | µs | idem, `Invoke()` do interpretador |
| `t_post` | µs | idem, decodificação + NMS |
| `t_total` = latência | µs → ms | `t_pre + t_inf + t_post` (janela D5) |
| Throughput (single-stream) | inf/s | `1 / t_total` por inferência |
| Throughput sustentado | inf/s | nº de inferências concluídas em 60 s de laço fechado (E4) |
| Energia por inferência | mJ | **derivada** pelo modelo de potência (§6): `E = Σ P_fase · t_fase` |
| Eficiência | inf/J | `1 / E` |
| EDP | mJ·ms | `E × t_total` |
| Acurácia | mAP@50, mAP@50-95 | E0 (offline) e verificação de equivalência E1 |

### 2.3 Controladas (fixadas e registradas)

| Variável | Valor fixado | Registro |
|---|---|---|
| Artefato do modelo | um único `.tflite` int8 por bloco (A/B) | SHA-256 no CSV |
| Resolução de entrada `r` | **comum às duas plataformas** (D31) | coluna `input_res` |
| Quantização | PTQ int8 *full-integer*, calibração com 750 imagens do *train* (D17) | script + lista de imagens versionados |
| Limiares de pós-processamento | `conf ≥ 0,25`, `IoU_NMS = 0,45`, máx. 100 detecções | `meta.json` |
| Entradas | mesmas `M = 50` imagens do *test*, pré-letterboxed em `r×r` RGB uint8 (D30), ingeridas direto no modelo (D33) | hash de cada `.bin` |
| Clock / governor | MCU 240 MHz, sem DFS/light-sleep; SBC `performance` 1 GHz (D21) | `sdkconfig`, `config.txt`, `scaling_governor` |
| Threads | MCU: 1 núcleo (TFLM); SBC: 4 threads no principal (E3) | coluna `threads` |
| Rádio | Wi-Fi/BT desligados no MCU; no SBC, Wi-Fi associado e ocioso (SSH só fora dos blocos) | `meta.json` |
| Alimentação | fonte 5 V oficial do RPi / USB 5 V do MCU, mesmo cabo | `meta.json` |
| Ambiente | sala com temperatura registrada; dissipador passivo no SBC | `meta.json` |

### 2.4 Covariáveis registradas (não controladas, monitoradas)

`cpu_temp_c` (SBC; o ESP32 LX6 não tem sensor confiável → temperatura ambiente por termômetro),
`arm_freq_mhz` e `throttled` (`vcgencmd get_throttled`) no SBC, `n_det` (nº de detecções — afeta
`t_post`), `image_id`, `block_id`, `seq_idx` (ordem — para checar deriva/autocorrelação).

## 3. Decisões operacionais da v1.3–v1.4 (resumo)

| # | Decisão |
|---|---|
| **D26** | Dataset **Ultralytics Construction-PPE** (AGPL-3.0; splits oficiais). Bloco A = 1 classe (`vest`, id 2); bloco B = 11 classes. |
| **D27** | Um único *fine-tuning* do YOLOX-Nano (variante ti-lite) **por configuração de classes**, no notebook, **fora do escopo medido**. D9 passa a significar: sem re-treino por plataforma, sem QAT; só PTQ int8. (O COCO não contém classes de EPI, então algum treino é inevitável.) |
| **D28** | Cronometragem por **timestamps monotônicos de alta resolução em software** (`time.perf_counter_ns()` no SBC; `esp_timer_get_time()` no MCU). GPIO só se houver analisador lógico (verificação). **Nenhum I/O dentro da janela**: logs em buffer na RAM, gravados entre blocos. |
| **D29** | Parâmetros do modelo de potência (§6) com faixa de sensibilidade e análise de *break-even*. |
| **D30** | Na fase estática, as entradas são **tensores brutos `r×r×3` uint8 já em letterbox**, byte a byte idênticos nas duas plataformas. `pre` = cópia + quantização para o tensor de entrada; decodificação de JPEG/resize **não** entram na janela estática (entram na fase de câmera, E5). |
| **D31** | **Resolução comum** `r`: a maior em {416, 320, 256, 224, 160} que caiba na memória do ESP32 (arena TFLM + PSRAM) e conclua ≥ 30 inferências por bloco em ≤ 60 min. Definida em E2; o SBC a 416 entra só como referência exploratória (E3c). |
| **D32** | Estrutura de coleta em **blocos** (sessões independentes, com *power-cycle*): `B = 3` blocos por plataforma; N total ≥ max(30, N_piloto). |
| **D33** | Câmera ao vivo vira **extra**. Validação e medição por **ingestão direta**: no ESP32 os `.bin` chegam pela serial e ficam em RAM antes da janela; no RPi são lidos do disco. Transferência sempre **fora** da janela (§5.4). |

## 4. Experimentos

| Id | Nome | Fase | Plataforma | Saída |
|---|---|---|---|---|
| **E0** | Acurácia offline fp32 × int8 | F1 | PC/notebook | tabela mAP@50 / mAP@50-95 por `r` |
| **E1** | Equivalência funcional | F1 | MCU, SBC, PC | relatório de equivalência |
| **E2** | Piloto (viabilidade, warm-up, CV → N) | F1/F3 | MCU, SBC | `W`, `N`, `r` definidos |
| **E3** | Benchmark principal (estático) | F3 | MCU, SBC | `data/raw/*_E3_*.csv` |
| E3b | Varredura de threads {1, 2, 4} (Amdahl) | F3 | SBC | exploratório |
| E3c | SBC em 416×416 (se `r < 416`) | F3 | SBC | exploratório |
| **E4** | Throughput sustentado + deriva térmica | F3 | MCU, SBC | contagem/60 s + série de temperatura |
| E5 *(extra)* | Câmera ao vivo (tempo real) — opcional, fora de H1 | — | MCU, SBC | FPS sustentado, latência c/ captura |
| **E6** | Energia derivada + sensibilidade | F4 | — (software) | `data/processed/energy_*.csv` |

### E0 — Acurácia offline (notebook)
1. Treinar (D27) e exportar ONNX → TFLite int8 (`onnx2tf`), calibração com 750 imagens do *train*
   estratificadas por classe (sem usar *val*/*test*).
2. Avaliar **fp32** e **int8** no *test* oficial (141 imagens) para cada `r` candidato.
3. Registrar mAP@50 e mAP@50-95 (protocolo COCO), perda por quantização (Δ int8 − fp32) e o SHA-256.

### E1 — Equivalência funcional
1. Rodar o mesmo `.tflite` sobre as `M = 50` entradas `.bin` no PC (referência), no SBC e no MCU.
2. Critério de aceitação por imagem: mesmo nº de detecções após NMS; classes idênticas; IoU ≥ 0,95
   entre caixas pareadas; |Δscore| ≤ 1 passo de quantização da saída.
3. Divergências inevitáveis (ex.: arredondamento de kernels distintos) são documentadas, nunca
   "corrigidas" alterando a lógica.

### E2 — Piloto
1. **Viabilidade (MCU):** testar `r` em ordem decrescente até caber (arena + PSRAM) e cumprir D31.
2. **100 inferências consecutivas** por plataforma, percorrendo as M imagens.
3. **Warm-up `W`** (D22): janelas de 10 inferências; descartar até a variação relativa da mediana
   entre janelas consecutivas ficar < 2% por duas janelas seguidas; piso de 5.
4. **N:** com o CV de `t_total` após o warm-up, `N = ⌈(1,96 · CV / ε)²⌉`, ε = 1% (meia-largura relativa
   do IC 95%); `N_final = max(30, N)` por plataforma (N pode diferir entre plataformas). Se o tempo
   por inferência no MCU tornar N inviável, usar o maior N compatível com D31 (mínimo 30) e registrar.

### E3 — Benchmark principal (estático)
Por plataforma, `B = 3` blocos. Cada bloco:
1. *Power-cycle* → boot → 2 min ocioso (registrar temperatura inicial e `throttled`).
2. Warm-up de `W` inferências (gravadas com `is_warmup = 1`, excluídas da análise principal).
3. `K = ⌈N_final / B⌉` inferências cronometradas, percorrendo as M imagens em ordem fixa
   (*round-robin*), gravadas em buffer.
4. Gravação do CSV e `meta.json`; pausa de resfriamento até a temperatura voltar a ±2 °C da inicial
   (SBC) ou 5 min (MCU).
5. Ordem dos blocos intercalada entre plataformas quando possível (MCU, SBC, MCU, …) para diluir
   efeito de dia/ambiente.

**E3b (SBC, exploratório):** repetir 1 bloco com `threads ∈ {1, 2, 4}`; estimar a fração paralelizável
`p` por Amdahl, `S(n) = 1 / ((1 − p) + p/n)`, a partir de `t_inf` — vínculo direto com a ementa.

### E4 — Throughput sustentado e deriva térmica
Ao fim de cada bloco de E3: 60 s de laço fechado sem gravação por inferência (só contador) →
`throughput_sust = n / 60 s`. No SBC, amostrar `cpu_temp_c` e `throttled` a cada 5 s. Indício de
throttling (bit ativo ou queda de `arm_freq`) invalida o bloco, que é repetido.

### E5 — Câmera ao vivo (*extra*, opcional — D33)
Só se houver tempo depois de E3/E4; não bloqueia a entrega nem entra em H1.
Mesma resolução de captura, formato (RGB565 → RGB888) e nº de buffers nas duas pontas; 5 min por
plataforma; mede-se FPS sustentado e latência ponta a ponta **incluindo** captura e resize. Resultado
descritivo; não entra no teste de H1 (pipelines de câmera diferem — ameaça declarada).

### E6 — Energia derivada
Aplicar o modelo de potência (§6) a cada linha válida de E3 e E4 com `scripts/energy_model.py`;
gerar a análise de sensibilidade e o *break-even*.

## 5. Coleta: formato dos dados

### 5.1 CSV bruto por inferência — `data/raw/{platform}_{exp}_{bloco}_{data}.csv`

| Coluna | Tipo | Descrição |
|---|---|---|
| `run_uuid` | str | UUID do bloco |
| `experiment` | str | `E2`, `E3`, `E3b`, … |
| `platform` | str | `MCU` / `SBC` |
| `block_id` | int | 1..B |
| `seq_idx` | int | ordem dentro do bloco (inclui warm-up) |
| `is_warmup` | 0/1 | 1 = descartada da análise principal |
| `image_id` | str | nome da entrada `.bin` |
| `model_sha256` | str | hash do `.tflite` |
| `num_classes` | int | 1 ou 11 |
| `input_res` | int | `r` |
| `threads` | int | threads do interpretador |
| `t_pre_us`, `t_inf_us`, `t_post_us`, `t_total_us` | int | tempos por fase |
| `n_det` | int | detecções após NMS |
| `cpu_temp_c` | float | SBC: após a inferência, fora da janela; MCU: vazio |
| `arm_freq_mhz` | int | SBC (amostrado por bloco/5 s); MCU: 240 |
| `throttled` | str | SBC: `vcgencmd get_throttled`; MCU: vazio |

### 5.2 Metadados por bloco — `…_meta.json`
Plataforma e revisão de hardware; no ESP32, módulo (WROOM/WROVER) e PSRAM; SO/firmware, versão
do runtime (LiteRT/tflite-runtime, TFLM), compilador e flags; `sdkconfig`/`config.txt` (hash);
governor; fonte e cabo; temperatura ambiente; commit do repositório; operador; data/hora UTC.

### 5.3 Implementação de referência
- SBC (orquestrador completo e benchmark v1.3 em 3 fases):
  [`raspberry/pi_zero_2w/benchmark_yolox.py`](../raspberry/pi_zero_2w/benchmark_yolox.py) e
  [`raspberry/pi_zero_2w/run_all_v13_experiments.py`](../raspberry/pi_zero_2w/run_all_v13_experiments.py), além do coletor por `.bin`
  [`raspberry/pi_zero_2w/bench_collect.py`](../raspberry/pi_zero_2w/bench_collect.py).
- MCU: [`firmware/esp32/esp_idf/`](../firmware/esp32/esp_idf/) (projeto ESP-IDF v6.1 + `esp-tflite-micro`) e
  [`firmware/esp32/bench_serial/bench_serial.ino`](../firmware/esp32/bench_serial/bench_serial.ino)
  (esqueleto Arduino + TFLite Micro) com [`scripts/esp_serial_feeder.py`](../scripts/esp_serial_feeder.py) e
  [`telemetry/record_esp32_serial.py`](../telemetry/record_esp32_serial.py).
- Fase 4 (E6 — Energia, sensibilidade, estatística robusta e figuras):
  [`telemetry/energy_and_stats_v13.py`](../telemetry/energy_and_stats_v13.py),
  [`telemetry/plot_phase4_v13.py`](../telemetry/plot_phase4_v13.py) e [`scripts/energy_model.py`](../scripts/energy_model.py).

### 5.4 Ingestão direta de imagens no ESP32 (D33)

```
PC (esp_serial_feeder.py)                      ESP32 (bench_serial.ino)
  "IMG" | id:u16 | len:u32 | payload | crc32 ──►  recebe em buffer (PSRAM/SRAM), confere CRC
                                           ◄──  "ACK,<id>"
                                                t0 → pré → t1 → Invoke → t2 → pós → t3   (janela)
                                           ◄──  "RES,<id>,t_pre,t_inf,t_post,t_total,n_det"
                                           ◄──  "DET,<id>,x1,y1,x2,y2,score,cls"  (× n_det)
                                           ◄──  "END,<id>"
```

- Baud padrão 921 600 (registrar). A 160×160 (76,8 KB) a transferência leva ~1 s — **fora** da janela.
- Nada é impresso na serial entre `t0` e `t3`; o rádio fica desligado (`WiFi.mode(WIFI_OFF)`, `btStop()`).
- Clock fixo: `setCpuFrequencyMhz(240)` (Arduino) ou `sdkconfig` sem DFS (ESP-IDF); registrar a versão do
  core Arduino/ESP-IDF e da biblioteca TFLite Micro.
- As detecções devolvidas alimentam o E1 (equivalência) com as mesmas regras do SBC.
- Alternativa (registrar se usada): subconjunto de imagens embutido na flash (LittleFS ou array C).

## 6. Modelo de potência (energia por software, D24/D29)

`E_i = P_pre · t_pre,i + P_inf · t_inf,i + P_post · t_post,i` (potência **total da placa**, inclui o
consumo estático — D6). Valores nominais e faixa usada na sensibilidade:

| Plataforma | Modo | Nominal | Faixa (sensibilidade) | Fonte |
|---|---|---|---|---|
| SBC (5 V, placa) | ocioso | 0,60 W (120 mA) | 0,50 – 1,00 W | CNX Software (2021); Geerling (2021) |
| SBC | ativo 1 núcleo (pré/pós, `threads=1`) | 1,00 W (~200 mA) | 0,75 – 1,30 W | Geerling (2021) |
| SBC | ativo 4 núcleos (`Invoke`, `threads=4`) | 2,30 W (~460 mA) | 1,85 – 2,90 W | Geerling (2021); Tom's Hardware (2021) |
| MCU (3,3 V, chip) | ativo 240 MHz, rádio off | 0,224 W (68 mA) | 0,099 – 0,224 W (30–68 mA) | Espressif, ESP32 Datasheet v5.3, Tab. 4-2 |
| MCU | *overhead* da placa (LDO, USB-UART, LED, PSRAM) | 0 W | 0 – 0,25 W | parâmetro — declarado |

Notas:
1. Os valores do SBC são de **placa** e os do MCU de **chip**; por isso o *overhead* de placa do MCU
   entra como parâmetro e o teste de H1 é repetido no **pior caso para o MCU** (chip máx. + overhead máx.).
2. Com potência constante por modo, a energia é uma **transformação linear dos tempos**: a variância
   de `E` vem só de `t`. A incerteza do modelo é tratada pela sensibilidade, não pela estatística.
3. ***Break-even:*** `E_SBC ≥ E_MCU ⇔ P_SBC / P_MCU ≥ t_MCU / t_SBC`. Reportar a razão de potências
   necessária para inverter a conclusão e se ela cai dentro da faixa plausível. Se não cair, a
   conclusão sobre H1 é robusta ao modelo.
4. Métrica secundária para cenário de aplicação a `f` quadros/s: `E_ciclo(f) = P_ativo·t + P_ocioso·(1/f − t)`,
   válida para `t < 1/f`.

## 7. Análise (resumo; detalhes em `protocolo.md` §8)

- Unidade: inferência válida (pós warm-up) de E3; por plataforma, mediana + IQR, média ± dp, CV.
- Comparação: Mann-Whitney U (bilateral para latência; unilateral nas direções de H1), Cliff's δ e IC
  95% *bootstrap* (10 000 reamostras, BCa) da diferença e da razão das medianas.
- **H1** (teste de interseção-união, cada parte a α = 0,05, sem correção):
  `throughput(SBC) > throughput(MCU)` **e** `E(SBC) ≥ E(MCU)`. A segunda parte é avaliada no nominal e
  nas bordas da faixa de §6.
- Validade: efeito de bloco (Kruskal-Wallis entre blocos), autocorrelação (ACF lag 1 de `t_total`
  vs `seq_idx`), sensibilidade com remoção de Tukey (1,5×IQR) reportando N original/removido.
- Secundária: mesma análise só com `t_inf` (isola o núcleo de inferência das diferenças de linguagem do
  pré/pós entre plataformas).

## 8. Checklist de preparação do Raspberry Pi (antes de E2)

- [x] Fonte oficial 5 V/2,5 A; dissipador passivo; sem periféricos USB/HDMI.
- [x] `config.txt`: `arm_freq=1000`; registrar o arquivo (hash) — sem overclock.
- [x] Governor: `echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`.
- [x] Registrar `uname -a`, versão do SO, `python3 --version`, versão do runtime TFLite/LiteRT (`ai-edge-litert` 2.3.0).
- [x] `vcgencmd get_throttled` = `0x0` antes e depois de cada bloco.
- [x] Copiar o `.tflite` (`models/yolox_nano_ppe_int8.tflite` e `models/yolox_nano_ppe_float32.tflite`) e imagens de entrada.
- [x] Rodar bateria completa no SBC (`run_all_v13_experiments.py`: `E0`, `E1`, `E2`, `E3b` 1/2/4 threads, `E3c` INT8/FP32, `E4` 60 s contínuos e `E5` memória) e gerar relatórios/figuras da Fase 4 (`E6`).

### Checklist do ESP32 (antes de E2)

- [ ] Confirmar módulo (WROOM/WROVER) e PSRAM; habilitar PSRAM no core Arduino se houver.
- [ ] Gravar `bench_serial.ino` com o `.tflite` embutido (array C) e conferir o SHA-256 impresso no boot.
- [ ] `python scripts/esp_serial_feeder.py --port COMx --inputs inputs/ --experiment E2 --n 30` no PC.
- [ ] Sessão SSH ociosa durante o bloco (iniciar com `nohup`/`tmux` e desconectar).

## 9. Pendências que bloqueiam a coleta final

1. `r` comum (depende da viabilidade no ESP32 — E2).
2. Artefatos A/B treinados e quantizados (E0) com hash.
3. Confirmar o módulo do ESP32 (WROOM sem PSRAM inviabiliza `r` alto).
4. Valores nominais de §6 aprovados pela dupla (podem ser refinados até o início de E3; depois, congelados).
