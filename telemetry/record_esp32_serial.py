#!/usr/bin/env python3
"""
Coletor e Alimentador Serial Completo do ESP32-S3 — Protocolo Experimental v1.5 (docs/projeto/main.tex)
Projeto PTT6 - Arquitetura e Organização de Computadores (PPCA/UFRR)
Orientador: Prof. Dr. Herbert Oliveira Rocha
Alunos: Filipe Matias & Guilherme Lucas

Realiza a ingestão direta das M=50 imagens BGR (.bin, 416x416x3 uint8) pela serial USB-UART
(quadro "IMG" | id:u16 | len:u32 | payload | crc32:u32 fora da janela cronometrada, Seção 4.8 / D33)
e coleta todas as métricas da MCU exigidas em docs/projeto/main.tex:
  1. E1 : Equivalência funcional nas M=50 imagens BGR (salva mcu_e1_50_predictions.json e esp32_e1_predictions.json)
  2. E2 : Piloto de 100 inferências na MCU (cálculo empírico de W e N_final via CV de t_total -> mcu_e2_pilot_summary.json)
  3. E3 : Benchmark principal em 3 blocos independentes (W=5 warmups + 10 execuções válidas por bloco = 30 válidas,
          percorrendo as entradas BGR em ordem fixa -> esp32_benchmark_raw.csv e MCU_E3_int8.csv)
  4. E4 : Throughput sustentado em 60 segundos de laço fechado sem I/O serial durante a janela ("E4S" -> esp32_e4_continuous_60s.csv)
  5. E5 : Pegada de memória Flash, TFLM Tensor Arena, SRAM interna e PSRAM Octal (esp32_e5_memory_footprint.json)
"""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import time
import uuid
import zlib
from datetime import datetime, timezone
import numpy as np

try:
    import serial
except ImportError:
    print("ERRO: Biblioteca 'pyserial' não encontrada! Instale com: pip install pyserial")
    sys.exit(1)

CLASS_NAMES = [
    "helmet", "gloves", "vest", "boots", "goggles",
    "none", "Person", "no_helmet", "no_goggle", "no_gloves", "no_boots"
]

STD_FIELDS = [
    "run_uuid", "experiment", "platform", "block_id", "seq_idx", "is_warmup", "image_id",
    "model_sha256", "num_classes", "input_res", "threads", "t_pre_us", "t_inf_us",
    "t_post_us", "t_total_us", "n_det", "cpu_temp_c", "arm_freq_mhz", "throttled"
]


def build_img_frame(img_id: int, payload: bytes) -> bytes:
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return b"IMG" + struct.pack("<HI", img_id & 0xFFFF, len(payload)) + payload + struct.pack("<I", crc)


def build_run_frame(img_id: int) -> bytes:
    return b"RUN" + struct.pack("<H", img_id & 0xFFFF)


def build_e4_frame(duration_s: int) -> bytes:
    return b"E4S" + struct.pack("<H", duration_s & 0xFFFF)


def read_serial_line(ser, log_lines: list[str], timeout_s: float = 30.0) -> str:
    t_end = time.monotonic() + timeout_s
    while time.monotonic() < t_end:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").strip()
        if line:
            log_lines.append(line)
            return line
    raise TimeoutError(f"Timeout ({timeout_s}s) aguardando linha serial do ESP32-S3")


def send_and_collect_single(ser, frame_bytes: bytes, expected_id: int, log_lines: list[str], is_img: bool = True):
    """Envia quadro IMG ou RUN (fora da janela), aguarda ACK e coleta RES + DET + END."""
    if is_img:
        # Envia cabeçalho + payload em blocos de 16 KB para evitar overflow do buffer USB-UART
        mv = memoryview(frame_bytes)
        chunk_sz = 16384
        for offset in range(0, len(mv), chunk_sz):
            ser.write(mv[offset:offset + chunk_sz])
        ser.flush()
    else:
        ser.write(frame_bytes)
        ser.flush()

    while True:
        line = read_serial_line(ser, log_lines, timeout_s=30.0)
        if line.startswith("ACK,"):
            break
        if line.startswith("NAK,"):
            raise RuntimeError(f"ESP32-S3 retornou {line}")

    res_line = read_serial_line(ser, log_lines, timeout_s=30.0)
    while not res_line.startswith("RES,"):
        res_line = read_serial_line(ser, log_lines, timeout_s=30.0)
    parts = res_line.split(",")
    t_pre_us, t_inf_us, t_post_us, t_total_us, n_det = map(int, parts[2:7])

    dets = []
    while True:
        line = read_serial_line(ser, log_lines, timeout_s=30.0)
        if line.startswith("END,"):
            break
        if line.startswith("DET,"):
            dp = line.split(",")
            x1, y1, x2, y2 = float(dp[2]), float(dp[3]), float(dp[4]), float(dp[5])
            score = float(dp[6])
            cls_id = int(dp[7])
            dets.append({
                "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "score": round(score, 4),
                "cls": cls_id,
                "class_name": CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else "unknown",
            })

    return {
        "t_pre_us": t_pre_us,
        "t_inf_us": t_inf_us,
        "t_post_us": t_post_us,
        "t_total_us": t_total_us,
        "n_det": n_det,
        "detections": dets,
    }


def compute_e2_pilot_summary(pilot_runs: list[dict]) -> dict:
    t_tot = np.array([r["t_total_us"] / 1000.0 for r in pilot_runs], dtype=np.float64)
    t_inf = np.array([r["t_inf_us"] / 1000.0 for r in pilot_runs], dtype=np.float64)

    win_size = 10
    n_wins = len(t_tot) // win_size
    win_meds = [float(np.median(t_tot[i * win_size:(i + 1) * win_size])) for i in range(n_wins)]
    rel_diffs = [abs(win_meds[i] - win_meds[i - 1]) / win_meds[i - 1] for i in range(1, len(win_meds))]

    w_emp = 5
    for idx in range(len(rel_diffs) - 1):
        if rel_diffs[idx] < 0.02 and rel_diffs[idx + 1] < 0.02:
            w_emp = max(5, idx * win_size)
            break

    post_warmup = t_tot[w_emp:]
    mean_t = float(np.mean(post_warmup))
    std_t = float(np.std(post_warmup, ddof=1)) if len(post_warmup) > 1 else 0.0
    cv = std_t / mean_t if mean_t > 0 else 0.0
    n_calc = int(math.ceil(((1.96 * cv) / 0.01) ** 2))
    n_final = max(30, n_calc)

    return {
        "platform": "MCU (ESP32-S3 @ 240 MHz, 8 MB Octal PSRAM)",
        "pilot_total_inferences": len(t_tot),
        "window_size": win_size,
        "window_medians_t_total_ms": [round(x, 3) for x in win_meds],
        "window_rel_diffs_pct": [round(x * 100.0, 4) for x in rel_diffs],
        "warmup_W_empirical": w_emp,
        "warmup_W_adopted_per_block": w_emp,
        "post_warmup_t_total_median_ms": round(float(np.median(post_warmup)), 3),
        "post_warmup_t_total_mean_ms": round(mean_t, 3),
        "post_warmup_t_total_std_ms": round(std_t, 3),
        "post_warmup_t_inf_median_ms": round(float(np.median(t_inf[w_emp:])), 3),
        "cv_t_total": round(cv, 6),
        "cv_t_total_percent": round(cv * 100.0, 4),
        "n_formula_eq1_eps1pct": n_calc,
        "n_final_adopted": n_final,
    }


def main():
    parser = argparse.ArgumentParser(description="Coletor e Alimentador Serial ESP32-S3 (Protocolo v1.5 / main.tex)")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Porta serial do ESP32-S3")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (padrão ESP-IDF console: 115200)")
    parser.add_argument("--inputs-dir", default="data/inputs_50_bgr_416", help="Diretório com os M=50 arquivos .bin BGR")
    parser.add_argument("--out-csv", default="data/raw/esp32_benchmark_raw.csv")
    parser.add_argument("--out-e1", default="data/raw/esp32_e1_predictions.json")
    parser.add_argument("--out-e1-50", default="data/raw/mcu_e1_50_predictions.json")
    parser.add_argument("--out-e2-csv", default="data/raw/esp32_e2_pilot_100.csv")
    parser.add_argument("--out-e2-json", default="data/raw/mcu_e2_pilot_summary.json")
    parser.add_argument("--out-e4-csv", default="data/raw/esp32_e4_continuous_60s.csv")
    parser.add_argument("--out-e5", default="data/raw/esp32_e5_memory_footprint.json")
    parser.add_argument("--out-log", default="data/raw/esp32_benchmark.log")
    args = parser.parse_args()

    bin_files = sorted(glob.glob(os.path.join(args.inputs_dir, "*.bin")))
    if len(bin_files) < 50:
        print(f"ERRO: Esperados 50 arquivos .bin em '{args.inputs_dir}', encontrados {len(bin_files)}")
        sys.exit(1)
    inputs_50 = [(os.path.splitext(os.path.basename(f))[0], open(f, "rb").read()) for f in bin_files[:50]]

    print("=======================================================================")
    print("  Coletor e Alimentador Serial YOLOX-Nano INT8 — ESP32-S3 (v1.5)       ")
    print("=======================================================================")
    print(f"Porta Serial  : {args.port} @ {args.baud} bps")
    print(f"Entradas BGR  : {len(inputs_50)} arquivos .bin (416x416x3 uint8 BGR)")

    ser = serial.Serial(args.port, args.baud, timeout=1.0)
    ser.dtr = False
    ser.rts = True
    time.sleep(0.15)
    ser.rts = False
    ser.reset_input_buffer()

    log_lines = []
    mem_footprint = {}
    boot_info = {}

    print("[BOOT] Aguardando inicialização do ESP32-S3...")
    while True:
        line = read_serial_line(ser, log_lines, timeout_s=20.0)
        print(f"  {line}")
        if line.startswith("MEM_FOOTPRINT,"):
            parts = line.split(",")
            mem_footprint = {
                "device": "ESP32-S3 (Xtensa LX7 @ 240 MHz)",
                "runtime": "esp-tflite-micro (ESP-NN SIMD)",
                "int8_flash_bytes": int(parts[1]),
                "int8_flash_mb": round(int(parts[1]) / (1024.0 * 1024.0), 3),
                "tflm_arena_used_bytes": int(parts[2]),
                "tflm_arena_used_kb": round(int(parts[2]) / 1024.0, 2),
                "tflm_arena_used_mb": round(int(parts[2]) / (1024.0 * 1024.0), 3),
                "free_internal_sram_bytes": int(parts[3]),
                "free_internal_sram_kb": round(int(parts[3]) / 1024.0, 2),
                "free_octal_psram_bytes": int(parts[4]),
                "free_octal_psram_kb": round(int(parts[4]) / 1024.0, 2),
            }
        elif line.startswith("BOOT,"):
            boot_info = dict(kv.split("=", 1) for kv in line.split(",")[1:] if "=" in kv)
            break

    model_sha = boot_info.get("sha256", "")
    arena_kb = int(boot_info.get("arena", 2224564)) // 1024
    sram_kb = int(boot_info.get("sram_free", 362075)) // 1024
    psram_kb = int(boot_info.get("psram_free", 3672680)) // 1024

    all_pilot_runs = []  # Coleta 100 inferências para E2
    e1_50_entries = []   # Coleta as 50 imagens BGR para E1
    e3_csv_rows = []     # Coleta 3 blocos x 10 execuções válidas para E3
    e3_std_rows = []     # Esquema de 19 colunas (experimentacao.md §5.1)

    # 1. Experimento E3 (3 blocos independentes: W=5 warmups + 10 execuções válidas/bloco = 30 válidas)
    #    percorrendo as primeiras 30 imagens BGR (01..30) enviadas via quadro serial "IMG" | id | len | payload | crc32
    global_valid_iter = 0
    for b in range(1, 4):
        block_uuid = str(uuid.uuid4())
        seq_idx = 0
        print(f"\n--- [E3 Bloco {b}/3] Executando W=5 inferências de warmup (is_warmup=1) ---")
        # Envia a primeira imagem do bloco via IMG no primeiro warmup e usa RUN nos 4 seguintes
        w_img_id, w_payload = inputs_50[(b - 1) * 10]
        for w in range(5):
            if w == 0:
                res = send_and_collect_single(ser, build_img_frame(seq_idx, w_payload), seq_idx, log_lines, is_img=True)
            else:
                res = send_and_collect_single(ser, build_run_frame(seq_idx), seq_idx, log_lines, is_img=False)
            all_pilot_runs.append({"image_id": w_img_id, "is_warmup": 1, **res})
            e3_std_rows.append([
                block_uuid, "E3", "MCU", b, seq_idx, 1, w_img_id,
                model_sha, 11, 416, 1,
                res["t_pre_us"], res["t_inf_us"], res["t_post_us"], res["t_total_us"],
                res["n_det"], "", 240, ""
            ])
            seq_idx += 1
            print(f"  [Warmup {w+1}/5] {w_img_id}: t_total={res['t_total_us']/1000.0:.2f} ms (t_inf={res['t_inf_us']/1000.0:.2f} ms)")

        print(f"--- [E3 Bloco {b}/3 + E1] Ingerindo 10 imagens BGR via serial c/ CRC32 e coletando métricas ---")
        for r in range(1, 11):
            img_idx = (b - 1) * 10 + (r - 1)
            img_id, payload = inputs_50[img_idx]
            global_valid_iter += 1
            res = send_and_collect_single(ser, build_img_frame(img_idx + 1, payload), img_idx + 1, log_lines, is_img=True)

            all_pilot_runs.append({"image_id": img_id, "is_warmup": 0, **res})
            e1_50_entries.append({
                "image_id": img_id,
                "timing_us": {
                    "t_pre_us": res["t_pre_us"],
                    "t_inf_us": res["t_inf_us"],
                    "t_post_us": res["t_post_us"],
                    "t_total_us": res["t_total_us"],
                    "n_det": res["n_det"],
                },
                "num_detections": res["n_det"],
                "detections": res["detections"],
            })

            t_pre_ms = res["t_pre_us"] / 1000.0
            t_inf_ms = res["t_inf_us"] / 1000.0
            t_post_ms = res["t_post_us"] / 1000.0
            t_tot_ms = res["t_total_us"] / 1000.0
            e3_csv_rows.append(
                f"{global_valid_iter},{b},{r},{img_id},{res['t_pre_us']},{res['t_inf_us']},{res['t_post_us']},{res['t_total_us']},"
                f"{t_pre_ms:.3f},{t_inf_ms:.3f},{t_post_ms:.3f},{t_tot_ms:.3f},"
                f"{res['t_inf_us']},{t_inf_ms:.3f},{res['n_det']},{arena_kb},{sram_kb},{psram_kb}\n"
            )
            e3_std_rows.append([
                block_uuid, "E3", "MCU", b, seq_idx, 0, img_id,
                model_sha, 11, 416, 1,
                res["t_pre_us"], res["t_inf_us"], res["t_post_us"], res["t_total_us"],
                res["n_det"], "", 240, ""
            ])
            seq_idx += 1
            print(f"  [Bloco {b} | {r}/10 | Global {global_valid_iter}/30] {img_id}: t_pre={t_pre_ms:.2f}ms, t_inf={t_inf_ms:.2f}ms, t_post={t_post_ms:.2f}ms, dets={res['n_det']}")

        if b < 3:
            print("  [Pausa entre blocos] Aguardando 2,0s...")
            time.sleep(2.0)

    # 2. Completar as 20 imagens restantes (31..50) do conjunto M=50 via serial c/ CRC32 para o Experimento E1
    print("\n--- [E1] Ingerindo as 20 imagens BGR restantes (31..50) via serial c/ CRC32 ---")
    for img_idx in range(30, 50):
        img_id, payload = inputs_50[img_idx]
        res = send_and_collect_single(ser, build_img_frame(img_idx + 1, payload), img_idx + 1, log_lines, is_img=True)
        all_pilot_runs.append({"image_id": img_id, "is_warmup": 0, **res})
        e1_50_entries.append({
            "image_id": img_id,
            "timing_us": {
                "t_pre_us": res["t_pre_us"],
                "t_inf_us": res["t_inf_us"],
                "t_post_us": res["t_post_us"],
                "t_total_us": res["t_total_us"],
                "n_det": res["n_det"],
            },
            "num_detections": res["n_det"],
            "detections": res["detections"],
        })
        print(f"  [E1 {img_idx+1}/50] {img_id}: t_total={res['t_total_us']/1000.0:.2f} ms, dets={res['n_det']}")

    # 3. Experimento E4: 60 segundos de laço fechado contínuo no ESP32-S3 sem I/O serial durante a janela
    print("\n--- [E4] Executando 60 segundos de inferência contínua em laço fechado no ESP32-S3 ---")
    ser.write(build_e4_frame(60))
    ser.flush()
    while True:
        line = read_serial_line(ser, log_lines, timeout_s=15.0)
        if line.startswith("ACK,E4,"):
            break

    e4_rows = []
    while True:
        line = read_serial_line(ser, log_lines, timeout_s=90.0)
        if line.startswith("E4_ROW,"):
            parts = line.split(",")
            it_idx = int(parts[1])
            el_s = float(parts[2])
            t_pre_us, t_inf_us, t_post_us, t_tot_us, n_det = map(int, parts[3:8])
            e4_rows.append({
                "iteration": it_idx,
                "elapsed_s": el_s,
                "t_pre_us": t_pre_us,
                "t_inf_us": t_inf_us,
                "t_post_us": t_post_us,
                "t_total_us": t_tot_us,
                "n_det": n_det,
            })
            all_pilot_runs.append({
                "image_id": inputs_50[-1][0],
                "is_warmup": 0,
                "t_pre_us": t_pre_us,
                "t_inf_us": t_inf_us,
                "t_post_us": t_post_us,
                "t_total_us": t_tot_us,
                "n_det": n_det,
                "detections": [],
            })
        elif line.startswith("E4_DONE,"):
            parts = line.split(",")
            e4_n_done = int(parts[1])
            e4_total_us = int(parts[2])
            break
    print(f"[OK] E4 ESP32-S3 concluído: {e4_n_done} inferências em {e4_total_us/1e6:.2f}s ({e4_n_done / (e4_total_us/1e6):.3f} inf/s)")

    # 4. Completar 100 inferências totais para o Piloto E2 (usando RUN na imagem em RAM)
    rem_pilot = max(0, 100 - len(all_pilot_runs))
    if rem_pilot > 0:
        print(f"\n--- [E2] Executando {rem_pilot} inferências adicionais em RAM para completar 100 amostras do Piloto E2 ---")
        for k in range(rem_pilot):
            idx_num = len(all_pilot_runs) + 1
            res = send_and_collect_single(ser, build_run_frame(idx_num), idx_num, log_lines, is_img=False)
            all_pilot_runs.append({"image_id": inputs_50[k % 50][0], "is_warmup": 0, **res})
            if (k + 1) % 5 == 0 or (k + 1) == rem_pilot:
                print(f"  [E2 Piloto {len(all_pilot_runs)}/100] t_total={res['t_total_us']/1000.0:.2f} ms")

    ser.close()

    # Salvar arquivos brutos e relatórios
    os.makedirs("data/raw", exist_ok=True)

    # a) E3 CSV principal + padrão 19 colunas
    e3_header = (
        "iteration,block,run_in_block,image_id,t_pre_us,t_inf_us,t_post_us,t_total_us,"
        "t_pre_ms,t_inf_ms,t_post_ms,t_total_ms,latency_us,latency_ms,"
        "num_detections,arena_kb,free_sram_kb,free_psram_kb\n"
    )
    with open(args.out_csv, "w", encoding="utf-8") as f:
        f.write(e3_header)
        f.writelines(e3_csv_rows)

    with open("data/raw/MCU_E3_int8.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(STD_FIELDS)
        w.writerows(e3_std_rows)

    # b) E1 (50 imagens BGR + resumo primeira imagem)
    total_mcu_dets = sum(e["num_detections"] for e in e1_50_entries)
    with open(args.out_e1_50, "w", encoding="utf-8") as f:
        json.dump({
            "platform": "MCU ESP32-S3 (esp-tflite-micro + ESP-NN SIMD)",
            "model": "models/yolox_nano_ppe_int8.tflite",
            "model_sha256": model_sha,
            "channel_order": "BGR",
            "input_res": 416,
            "out_scale": float(boot_info.get("out_scale", 0.027536)),
            "out_zp": int(boot_info.get("out_zp", -34)),
            "conf_thr": 0.25,
            "nms_thr": 0.45,
            "num_images": len(e1_50_entries),
            "total_detections": total_mcu_dets,
            "images": e1_50_entries,
        }, f, indent=2, ensure_ascii=False)

    first_e1 = e1_50_entries[0]
    with open(args.out_e1, "w", encoding="utf-8") as f:
        json.dump({
            "device": "ESP32-S3",
            "model": "models/yolox_nano_ppe_int8.tflite",
            "model_sha256": model_sha,
            "image": f"{first_e1['image_id']}.bin (416x416x3 letterbox BGR)",
            "first_run_timing_us": {
                "t_pre": first_e1["timing_us"]["t_pre_us"],
                "t_inf": first_e1["timing_us"]["t_inf_us"],
                "t_post": first_e1["timing_us"]["t_post_us"],
                "t_total": first_e1["timing_us"]["t_total_us"],
                "num_dets": first_e1["num_detections"],
            },
            "num_detections": first_e1["num_detections"],
            "detections": [
                {
                    "det_idx": idx,
                    "class_id": d["cls"],
                    "class_name": d["class_name"],
                    "score": d["score"],
                    "bbox_xyxy": d["bbox_xyxy"],
                }
                for idx, d in enumerate(first_e1["detections"])
            ],
            "all_50_images_summary": {
                "num_images": len(e1_50_entries),
                "total_detections": total_mcu_dets,
                "file": args.out_e1_50,
            },
        }, f, indent=2, ensure_ascii=False)

    # c) E2 Piloto de 100 inferências
    with open(args.out_e2_csv, "w", encoding="utf-8") as f:
        f.write("iteration,image_id,is_warmup,t_pre_us,t_inf_us,t_post_us,t_total_us,t_pre_ms,t_inf_ms,t_post_ms,t_total_ms,num_detections\n")
        for idx, r in enumerate(all_pilot_runs[:100], start=1):
            f.write(
                f"{idx},{r['image_id']},{r['is_warmup']},{r['t_pre_us']},{r['t_inf_us']},{r['t_post_us']},{r['t_total_us']},"
                f"{r['t_pre_us']/1000.0:.3f},{r['t_inf_us']/1000.0:.3f},{r['t_post_us']/1000.0:.3f},{r['t_total_us']/1000.0:.3f},{r['n_det']}\n"
            )
    e2_summary = compute_e2_pilot_summary(all_pilot_runs[:100])
    with open(args.out_e2_json, "w", encoding="utf-8") as f:
        json.dump(e2_summary, f, indent=2, ensure_ascii=False)

    # d) E4 Contínuo 60s
    with open(args.out_e4_csv, "w", encoding="utf-8") as f:
        f.write("iteration,elapsed_s,t_pre_us,t_inf_us,t_post_us,t_total_us,t_pre_ms,t_inf_ms,t_post_ms,t_total_ms,latency_us,latency_ms,num_detections\n")
        for r in e4_rows:
            f.write(
                f"{r['iteration']},{r['elapsed_s']:.3f},{r['t_pre_us']},{r['t_inf_us']},{r['t_post_us']},{r['t_total_us']},"
                f"{r['t_pre_us']/1000.0:.3f},{r['t_inf_us']/1000.0:.3f},{r['t_post_us']/1000.0:.3f},{r['t_total_us']/1000.0:.3f},"
                f"{r['t_inf_us']},{r['t_inf_us']/1000.0:.3f},{r['n_det']}\n"
            )

    # e) E5 Memória e Log
    if mem_footprint:
        mem_footprint["model_sha256"] = model_sha
        with open(args.out_e5, "w", encoding="utf-8") as f:
            json.dump(mem_footprint, f, indent=2, ensure_ascii=False)

    with open(args.out_log, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    if os.path.isdir("results"):
        for p in [args.out_csv, args.out_e1, args.out_e1_50, args.out_e2_csv, args.out_e2_json, args.out_e4_csv, args.out_e5, args.out_log]:
            if os.path.exists(p):
                shutil.copyfile(p, os.path.join("results", os.path.basename(p)))

    print("\n[OK] Coleta completa no ESP32-S3 concluída com sucesso:")
    print(f"  - E1 (50 imagens BGR): {args.out_e1_50} ({total_mcu_dets} detecções)")
    print(f"  - E2 (Piloto 100 inf): {args.out_e2_json} (W={e2_summary['warmup_W_empirical']}, N_final={e2_summary['n_final_adopted']})")
    print(f"  - E3 (3 blocos x 10) : {args.out_csv}")
    print(f"  - E4 (60s contínuo)  : {args.out_e4_csv} ({len(e4_rows)} inferências)")
    print(f"  - E5 (Memória)       : {args.out_e5}")


if __name__ == "__main__":
    main()
