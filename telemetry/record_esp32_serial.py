#!/usr/bin/env python3
"""
Script de captura de telemetria serial do ESP32 / ESP32-S3 para o projeto PTT6.
Lê a saída serial em /dev/ttyUSB0 a 115200 baud, exibe em tempo real,
e grava os tempos de latência em data/raw/esp32_benchmark_raw.csv.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import serial

def main():
    parser = argparse.ArgumentParser(description="Coletor de telemetria serial ESP32 / ESP32-S3")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Porta serial (padrão: /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (padrão: 115200)")
    parser.add_argument("--out-csv", default="data/raw/esp32_benchmark_raw.csv", help="Caminho do CSV de saída")
    parser.add_argument("--out-log", default="data/raw/esp32_benchmark.log", help="Caminho do log completo")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout em segundos (padrão: 180s)")
    args = parser.parse_args()

    out_csv_path = Path(args.out_csv)
    out_log_path = Path(args.out_log)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Abrindo porta serial {args.port} a {args.baud} baud...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1.0)
    except Exception as e:
        print(f"ERRO ao abrir porta serial {args.port}: {e}")
        print("DICA: Certifique-se de executar 'sudo chmod 666 /dev/ttyUSB0' se houver erro de permissão.")
        sys.exit(1)

    # Reset suave do ESP32 via linhas DTR/RTS
    ser.setDTR(False)
    ser.setRTS(True)
    time.sleep(0.1)
    ser.setRTS(False)
    time.sleep(0.2)

    csv_rows = []
    in_benchmark = False
    start_time = time.time()

    print("ESP32 reiniciado. Aguardando saída do benchmark...")

    with open(out_log_path, "w", encoding="utf-8") as flog:
        while True:
            if time.time() - start_time > args.timeout:
                print(f"\n[AVISO] Timeout de {args.timeout}s atingido!")
                break

            line = ser.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue

            print(line)
            flog.write(line + "\n")
            flog.flush()

            if "=== BENCHMARK_START ===" in line:
                in_benchmark = True
                csv_rows = []
                continue

            if in_benchmark and line.startswith("RUN,"):
                parts = line.split(",")
                if len(parts) == 3:
                    csv_rows.append((parts[1], parts[2]))

            if "=== BENCHMARK_END ===" in line:
                in_benchmark = False
                print("\n[OK] Fim da bateria de 30 inferências detectado!")
                # Aguarda mais 3 segundos para capturar o resumo estatístico
                extra_timeout = time.time() + 3
                while time.time() < extra_timeout:
                    extra_line = ser.readline().decode("utf-8", errors="replace").strip()
                    if extra_line:
                        print(extra_line)
                        flog.write(extra_line + "\n")
                        flog.flush()
                break

    ser.close()

    if csv_rows:
        with open(out_csv_path, "w", encoding="utf-8") as fcsv:
            fcsv.write("iteration,latency_us,latency_ms\n")
            for it, lat in csv_rows:
                lat_us = int(lat)
                fcsv.write(f"{it},{lat_us},{lat_us / 1000.0:.3f}\n")
        print(f"\n[SUCESSO] {len(csv_rows)} medições salvas em {out_csv_path}")
        print(f"Log completo salvo em {out_log_path}")
    else:
        print("\n[ALERTA] Nenhuma linha RUN capturada no período.")

if __name__ == "__main__":
    main()

