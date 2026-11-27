# Vídeo de Demonstração — PTT6: Benchmarking YOLOX-Nano INT8 em MCU e SBC

- **Link do vídeo no YouTube**: https://youtu.be/qOvJVo9876Y
- **Autores**: Filipe Matias & Guilherme Lucas
- **Orientador**: Prof. Dr. Herbert Oliveira Rocha
- **Disciplina**: Arquitetura e Organização de Computadores — PPCA/UFRR (Trilha PTT6)

---

## Roteiro da Demonstração

1. **Introdução e Contextualização Experimental**:
   - Apresentação do problema de detecção de EPIs (*Construction-PPE*, 11 classes, $416\times 416$ BGR) e execução do mesmo artefato `.tflite` quantizado em `int8` ($1{,}16\text{ MiB}$, `SHA-256: 7205d9dd...`) nas duas plataformas de borda: **Raspberry Pi Zero 2 W** (SBC, quad-core ARM Cortex-A53 @ 1,0 GHz) e **ESP32-S3** (MCU, dual-core Xtensa LX7 @ 240 MHz, 8 MB PSRAM Octal).
2. **Execução ao Vivo no Raspberry Pi Zero 2 W (`SBC`)**:
   - Medição das três fases da janela ($t_{\text{pre}}, t_{\text{inf}}, t_{\text{post}}$) com as 50 entradas BGR em RAM (`ai-edge-litert` + XNNPACK NEON, 4 threads).
3. **Resultados de Predições (`E1`) e Desempenho (`E3`, `E3b`, `E4`) no Raspberry Pi**:
   - Equivalência funcional (`148` detecções nas 50 imagens BGR, $\text{IoU médio} = 0{,}996$ contra o PC), escalabilidade multi-core pela Lei de Amdahl (1, 2 e 4 threads, speedup de $2{,}47\times$, $p = 79{,}5\%$), comparação `int8` vs. `fp32` ($H_2$) e laço contínuo de 60 s (`481` inferências, $8{,}02\text{ inf/s}$, zero *throttling*).
4. **Execução ao Vivo e Resultados no ESP32-S3 (`MCU`)**:
   - Ingestão de imagens BGR pela serial USB com verificação CRC32 fora da janela cronometrada, pegada de memória (`Tensor Arena` de $2{,}12\text{ MiB}$ na PSRAM Octal), predições on-device (`E1`), benchmark em 3 blocos (`E3`, $\tilde{t}_{\text{total}} = 4.981{,}2\text{ ms}$, $0{,}20\text{ inf/s}$) e laço contínuo de 60 s (`E4`, `13` inferências).

---

## Comandos Utilizados na Demonstração

```bash
# 1. Execução rápida ao vivo no Raspberry Pi Zero 2 W
ssh filipe@10.42.0.197 \
  "cd /home/filipe/ppe_benchmark && python3 -u sbc_rpi/benchmark_yolox.py \
   --model models/yolox_nano_ppe_int8.tflite \
   --inputs-dir sbc_rpi/inputs_50_bgr_416 \
   --blocks 1 --warmup-per-block 2 --runs-per-block 5 \
   --threads 4 --out-csv /tmp/demo_rpi.csv"

# 2. Exibição dos resultados consolidados do SBC (E1, E3, E3b, E4)
python3 scripts/demo_video.py 3

# 3. Execução rápida ao vivo de 2 imagens BGR via serial no ESP32-S3
python3 telemetry/record_esp32_serial.py --port /dev/ttyACM0 --demo 2

# 4. Exibição dos resultados consolidados do ESP32-S3 (E5, E1, E3, E4)
python3 scripts/demo_video.py 4

