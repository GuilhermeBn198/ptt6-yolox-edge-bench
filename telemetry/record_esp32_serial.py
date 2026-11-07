#!/usr/bin/env python3
"""
Script de captura de telemetria serial do ESP32 / ESP32-S3 para o projeto PTT6 (Protocolo v1.3).
Lê a saída serial em /dev/ttyACM0 ou /dev/ttyUSB0 a 115200 baud, exibe em tempo real,
e grava:
  - data/raw/esp32_benchmark_raw.csv (decomposição em 3 fases: t_pre, t_inf, t_post, t_total em 3 blocos)
  - data/raw/esp32_e1_predictions.json (previsões/caixas detectadas pelo modelo INT8 no ESP32-S3)
  - data/raw/esp32_e5_memory_footprint.json (pegada de Flash, Tensor Arena TFLM, SRAM e PSRAM)
  - data/raw/esp32_benchmark.log (log serial completo)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import serial


def detect_default_port() -> str:
    for candidate in ("/dev/ttyACM0", "/dev/ttyUSB0", "/dev/ttyACM1", "/dev/ttyUSB1"):
        if os.path.exists(candidate):
            return candidate
    return "/dev/ttyACM0"


def main():
    parser = argparse.ArgumentParser(description="Coletor de telemetria serial ESP32 / ESP32-S3 (Protocolo v1.3)")
    parser.add_argument("--port", default=detect_default_port(), help="Porta serial (padrão: auto /dev/ttyACM0 ou /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (padrão: 115200)")
    parser.add_argument("--out-csv", default="data/raw/esp32_benchmark_raw.csv", help="Caminho do CSV de saída")
    parser.add_argument("--out-log", default="data/raw/esp32_benchmark.log", help="Caminho do log completo")
    parser.add_argument("--out-pred", default="data/raw/esp32_e1_predictions.json", help="Caminho do JSON de previsões (E1)")
    parser.add_argument("--out-mem", default="data/raw/esp32_e5_memory_footprint.json", help="Caminho do JSON de memória (E5)")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout em segundos (padrão: 600s)")
    parser.add_argument("--no-reset", action="store_true", help="Não enviar pulso DTR/RTS de reset ao abrir a porta")
    args = parser.parse_args()

    out_csv_path = Path(args.out_csv)
    out_log_path = Path(args.out_log)
    out_pred_path = Path(args.out_pred)
    out_mem_path = Path(args.out_mem)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Abrindo porta serial {args.port} a {args.baud} baud...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1.0)
    except Exception as e:
        print(f"ERRO ao abrir porta serial {args.port}: {e}")
        print(f"DICA: Certifique-se de executar 'sudo chmod 666 {args.port}' se houver erro de permissão.")
        sys.exit(1)

    if not args.no_reset:
        try:
            ser.setDTR(False)
            ser.setRTS(True)
            time.sleep(0.1)
            ser.setRTS(False)
            time.sleep(0.2)
        except Exception:
            pass

    csv_rows_v13 = []
    legacy_rows = []
    detections = []
    first_run_timing = {}
    mem_footprint = {}
    in_benchmark = False
    start_time = time.time()

    print("Aguardando saída do benchmark do ESP32-S3...")

    with open(out_log_path, "w", encoding="utf-8") as flog:
        while True:
            if time.time() - start_time > args.timeout:
                print(f"\n[AVISO] Timeout de {args.timeout}s atingido!")
                break

            try:
                raw_bytes = ser.readline()
            except Exception as e:
                print(f"\n[AVISO] Leitura serial interrompida: {e}")
                break

            line = raw_bytes.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            print(line, flush=True)
            flog.write(line + "\n")
            flog.flush()

            if line.startswith("MEM_FOOTPRINT,"):
                parts = line.split(",")
                if len(parts) == 5:
                    model_b = int(parts[1])
                    arena_b = int(parts[2])
                    sram_b = int(parts[3])
                    psram_b = int(parts[4])
                    mem_footprint = {
                        "device": "ESP32-S3 (Xtensa LX7 @ 240 MHz)",
                        "runtime": "esp-tflite-micro (ESP-NN SIMD)",
                        "int8_flash_bytes": model_b,
                        "int8_flash_mb": round(model_b / (1024.0 * 1024.0), 3),
                        "tflm_arena_used_bytes": arena_b,
                        "tflm_arena_used_kb": round(arena_b / 1024.0, 2),
                        "tflm_arena_used_mb": round(arena_b / (1024.0 * 1024.0), 3),
                        "free_internal_sram_bytes": sram_b,
                        "free_internal_sram_kb": round(sram_b / 1024.0, 2),
                        "free_octal_psram_bytes": psram_b,
                        "free_octal_psram_kb": round(psram_b / 1024.0, 2),
                    }

            elif line.startswith("FIRST_RUN_TIMING_US,"):
                for kv in line.split(",")[1:]:
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        first_run_timing[k] = int(v)

            elif line.startswith("DET,"):
                parts = line.split(",")
                if len(parts) == 9:
                    detections.append({
                        "det_idx": int(parts[1]),
                        "class_id": int(parts[2]),
                        "class_name": parts[3],
                        "score": float(parts[4]),
                        "bbox_xyxy": [float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])],
                    })

            elif "=== BENCHMARK_START ===" in line:
                in_benchmark = True
                csv_rows_v13 = []
                legacy_rows = []
                continue

            elif in_benchmark and line.startswith("CSV_ROW,"):
                parts = line.split(",")
                if len(parts) == 18:
                    csv_rows_v13.append(parts[1:])

            elif in_benchmark and line.startswith("RUN,"):
                parts = line.split(",")
                if len(parts) == 3:
                    legacy_rows.append((parts[1], parts[2]))

            elif "=== BENCHMARK_END ===" in line:
                in_benchmark = False
                print("\n[OK] Fim da bateria de 30 inferências detectado!")
                extra_timeout = time.time() + 3
                while time.time() < extra_timeout:
                    extra_line = ser.readline().decode("utf-8", errors="replace").strip()
                    if extra_line:
                        print(extra_line, flush=True)
                        flog.write(extra_line + "\n")
                        flog.flush()
                break

    ser.close()

    if detections or first_run_timing:
        pred_payload = {
            "device": "ESP32-S3",
            "model": "models/yolox_nano_ppe_int8.tflite",
            "image": "raspberry/pi_zero_2w/test_image.jpg (416x416x3 letterbox RGB)",
            "first_run_timing_us": first_run_timing,
            "num_detections": len(detections),
            "detections": detections,
        }
        with open(out_pred_path, "w", encoding="utf-8") as fpred:
            json.dump(pred_payload, fpred, indent=2, ensure_ascii=False)
        print(f"[SUCESSO] Previsões salvas em {out_pred_path}")

    if mem_footprint:
        with open(out_mem_path, "w", encoding="utf-8") as fmem:
            json.dump(mem_footprint, fmem, indent=2, ensure_ascii=False)
        print(f"[SUCESSO] Pegada de memória (E5) salva em {out_mem_path}")

    if csv_rows_v13:
        header = (
            "iteration,block,run_in_block,t_pre_us,t_inf_us,t_post_us,t_total_us,"
            "t_pre_ms,t_inf_ms,t_post_ms,t_total_ms,latency_us,latency_ms,"
            "num_detections,arena_kb,free_sram_kb,free_psram_kb\n"
        )
        with open(out_csv_path, "w", encoding="utf-8") as fcsv:
            fcsv.write(header)
            for row in csv_rows_v13:
                fcsv.write(",".join(row) + "\n")
        print(f"\n[SUCESSO] {len(csv_rows_v13)} medições (3 fases) salvas em {out_csv_path}")
        print(f"Log completo salvo em {out_log_path}")
    elif legacy_rows:
        with open(out_csv_path, "w", encoding="utf-8") as fcsv:
            fcsv.write("iteration,latency_us,latency_ms\n")
            for it, lat in legacy_rows:
                lat_us = int(lat)
                fcsv.write(f"{it},{lat_us},{lat_us / 1000.0:.3f}\n")
        print(f"\n[SUCESSO] {len(legacy_rows)} medições salvas em {out_csv_path}")
    else:
        print("\n[ALERTA] Nenhuma linha CSV_ROW/RUN capturada no período.")


if __name__ == "__main__":
    main()
