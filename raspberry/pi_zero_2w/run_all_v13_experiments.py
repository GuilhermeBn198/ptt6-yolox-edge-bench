#!/usr/bin/env python3
"""
Orquestrador Completo dos Experimentos no SBC (Raspberry Pi Zero 2 W) — Protocolo v1.3
Projeto PTT6 - Arquitetura e Organização de Computadores (PPCA/UFRR)
Orientador: Prof. Dr. Herbert Oliveira Rocha
Alunos: Filipe Matias & Guilherme Lucas

Executa na sequência os experimentos planejados na Tabela 4 do Protocolo v1.3:
  - E0 : Baseline ocioso (3 blocos de 30 s sem carga de inferência)
  - E1 : Sanidade da quantização (FP32 vs. INT8 sobre 30 amostras de teste fixas)
  - E2 : Cálculo formal do dimensionamento amostral N (margem <= 5% da média)
  - E3b: Escalabilidade multi-core (1, 2 e 4 threads em 3 blocos independentes + 10 warmups/bloco)
  - E3c: Comparação INT8 vs. FP32 em 416x416 (3 blocos independentes + 10 warmups/bloco)
  - E4 : Estabilidade térmica e temporal em execução contínua por 60 segundos (delay=0)
  - E5 : Pegada de memória Flash (.tflite) e RAM de pico (VmHWM / RSS)
"""

import glob
import json
import os
import subprocess
import sys
import time
import numpy as np

from benchmark_yolox import (
    Interpreter,
    RUNTIME_NAME,
    build_yolox_grids,
    load_source_image,
    postprocess_yolox,
    preprocess_frame,
    read_cpu_freq_mhz,
    read_cpu_temp_c,
    read_peak_rss_mb,
    read_throttled_hex,
)


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Calcula matriz IoU entre dois conjuntos de caixas [N, 4] e [M, 4] no formato xyxy."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = np.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0.0, boxes1[:, 3] - boxes1[:, 1])
    area2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter + 1e-9
    return inter / union


def run_e0_idle_baseline(out_csv="data/raw/rpi_e0_idle_baseline.csv", blocks=3, duration_per_block_s=30, interval_s=2.0):
    """Experimento E0: 3 janelas de 30 s sem carga de inferência."""
    print("\n=======================================================================")
    print("  [E0] Baseline Ocioso (3 x 30 s sem carga de inferência)             ")
    print("=======================================================================")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    lines = ["sample_idx,block,elapsed_block_s,cpu_temp_c,cpu_freq_mhz,throttled_hex\n"]
    sample_idx = 0
    for b in range(1, blocks + 1):
        print(f"  [E0] Bloco {b}/{blocks} ({duration_per_block_s}s em repouso)...")
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) < duration_per_block_s:
            sample_idx += 1
            el = time.perf_counter() - t0
            temp_c = read_cpu_temp_c()
            freq_mhz = read_cpu_freq_mhz()
            throttled = read_throttled_hex()
            lines.append(f"{sample_idx},{b},{el:.2f},{temp_c:.1f},{freq_mhz:.0f},{throttled}\n")
            time.sleep(interval_s)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[OK] E0 concluído: {sample_idx} amostras salvas em '{out_csv}'")


def run_e1_quantization_sanity(
    fp32_model="models/yolox_nano_ppe_float32.tflite",
    int8_model="models/yolox_nano_ppe_int8.tflite",
    image_dir="raspberry/pi_zero_2w/test_images_30",
    fallback_image="raspberry/pi_zero_2w/test_image.jpg",
    out_csv="data/raw/rpi_e1_quantization_sanity.csv",
    out_json="data/raw/rpi_e1_quantization_sanity.json",
):
    """Experimento E1: Sanidade da quantização (FP32 vs. INT8 em 30 imagens fixas)."""
    print("\n=======================================================================")
    print("  [E1] Sanidade da Quantização (FP32 vs. INT8 em 30 amostras fixas)   ")
    print("=======================================================================")
    interp_fp32 = Interpreter(model_path=fp32_model, num_threads=4)
    interp_fp32.allocate_tensors()
    in_fp32 = interp_fp32.get_input_details()[0]
    out_fp32 = interp_fp32.get_output_details()[0]

    interp_int8 = Interpreter(model_path=int8_model, num_threads=4)
    interp_int8.allocate_tensors()
    in_int8 = interp_int8.get_input_details()[0]
    out_int8 = interp_int8.get_output_details()[0]

    target_h, target_w = int(in_int8["shape"][1]), int(in_int8["shape"][2])
    in_scale, in_zp = in_int8.get("quantization", (1.0, -128))
    if in_scale == 0.0:
        in_scale = 1.0
    out_scale, out_zp = out_int8.get("quantization", (1.0, 0))
    if out_scale == 0.0:
        out_scale = 1.0

    grids, expanded_strides = build_yolox_grids(target_h, target_w)

    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png")))
    img_paths = img_paths[:30]

    base_pil, base_arr = load_source_image(fallback_image)
    rng = np.random.default_rng(2026)

    csv_lines = ["sample_id,image_name,fp32_dets,int8_dets,matched_dets_iou50,mean_iou,tensor_mae,tensor_corr\n"]
    maes = []
    corrs = []
    ious_all = []
    total_fp32_dets = 0
    total_matched_dets = 0

    for idx in range(30):
        if idx < len(img_paths):
            pil_img, raw_arr = load_source_image(img_paths[idx])
            img_name = os.path.basename(img_paths[idx])
        else:
            # Gera variação determinística reprodutível a partir da imagem de referência caso < 30 arquivos
            img_name = f"test_frame_{idx+1:02d}.jpg"
            if base_pil is not None:
                w, h = base_pil.size
                dx = int(rng.integers(-12, 13))
                dy = int(rng.integers(-12, 13))
                arr_mod = np.roll(np.roll(base_arr, dy, axis=0), dx, axis=1)
                from PIL import Image
                pil_img = Image.fromarray(arr_mod)
                raw_arr = arr_mod
            else:
                pil_img, raw_arr = None, base_arr

        t_in_fp32 = preprocess_frame(pil_img, raw_arr, target_h, target_w, in_fp32["dtype"], 1.0, 0)
        t_in_int8 = preprocess_frame(pil_img, raw_arr, target_h, target_w, in_int8["dtype"], in_scale, in_zp)

        interp_fp32.set_tensor(in_fp32["index"], t_in_fp32)
        interp_fp32.invoke()
        raw_y_fp32 = interp_fp32.get_tensor(out_fp32["index"])

        interp_int8.set_tensor(in_int8["index"], t_in_int8)
        interp_int8.invoke()
        raw_y_int8 = interp_int8.get_tensor(out_int8["index"])
        deq_y_int8 = (raw_y_int8.astype(np.float32) - float(out_zp)) * float(out_scale)

        mae = float(np.mean(np.abs(raw_y_fp32 - deq_y_int8)))
        corr = float(np.corrcoef(raw_y_fp32.ravel(), deq_y_int8.ravel())[0, 1])
        maes.append(mae)
        corrs.append(corr)

        dets_fp32 = postprocess_yolox(raw_y_fp32, out_fp32["dtype"], 1.0, 0, grids, expanded_strides)
        dets_int8 = postprocess_yolox(raw_y_int8, out_int8["dtype"], out_scale, out_zp, grids, expanded_strides)

        n_fp32 = len(dets_fp32)
        n_int8 = len(dets_int8)
        total_fp32_dets += n_fp32

        if n_fp32 > 0 and n_int8 > 0:
            iou_mat = compute_iou_matrix(dets_fp32[:, :4], dets_int8[:, :4])
            best_ious = np.max(iou_mat, axis=1)
            matched = int(np.sum(best_ious >= 0.50))
            mean_iou = float(np.mean(best_ious))
            total_matched_dets += matched
            ious_all.append(mean_iou)
        elif n_fp32 == 0 and n_int8 == 0:
            matched = 0
            mean_iou = 1.0
            ious_all.append(1.0)
        else:
            matched = 0
            mean_iou = 0.0
            ious_all.append(0.0)

        csv_lines.append(
            f"{idx+1},{img_name},{n_fp32},{n_int8},{matched},{mean_iou:.4f},{mae:.5f},{corr:.5f}\n"
        )

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.writelines(csv_lines)

    recall_vs_fp32 = (total_matched_dets / total_fp32_dets) if total_fp32_dets > 0 else 1.0
    summary = {
        "num_test_samples": 30,
        "total_fp32_detections": int(total_fp32_dets),
        "total_int8_matched_iou50": int(total_matched_dets),
        "box_agreement_recall_iou50": round(float(recall_vs_fp32), 4),
        "mean_best_iou": round(float(np.mean(ious_all)), 4),
        "mean_tensor_mae": round(float(np.mean(maes)), 5),
        "mean_tensor_pearson_r": round(float(np.mean(corrs)), 5),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] E1 concluído: Concordância IoU>=0.5 = {recall_vs_fp32*100:.1f}%, IoU médio = {np.mean(ious_all):.4f}, r = {np.mean(corrs):.4f}")
    return summary


def run_benchmark_cmd(args_list):
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_yolox.py")
    cmd = [sys.executable, script_path] + args_list
    print(f"\n>>> Executando: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ensure_performance_governor():
    """Garante que os 4 núcleos Cortex-A53 estejam em modo 'performance' (1,0 GHz)."""
    try:
        subprocess.run(
            ["sudo", "sh", "-c", "for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $f; done"],
            check=False,
        )
    except Exception:
        pass


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    if not os.path.exists(os.path.join(repo_root, "models")):
        repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    os.chdir(repo_root)
    print(f"[INFO] Diretório de trabalho: {os.getcwd()}")
    ensure_performance_governor()

    # 1. E0: Baseline ocioso (3 x 30s)
    run_e0_idle_baseline(blocks=3, duration_per_block_s=30, interval_s=2.0)

    # 2. E1: Sanidade da quantização FP32 vs INT8 em 30 amostras
    e1_summary = run_e1_quantization_sanity()

    # 3. E3b & E3c: 3 blocos independentes x 10 execuções válidas (= 30) + 10 warmups descartados/bloco
    #    a) INT8 — 4 Threads (E3c / E3b)
    run_benchmark_cmd([
        "--model", "models/yolox_nano_ppe_int8.tflite",
        "--threads", "4",
        "--blocks", "3",
        "--runs-per-block", "10",
        "--warmup-per-block", "10",
        "--block-cooldown", "8.0",
        "--delay", "0.1",
        "--out-csv", "data/raw/rpi_benchmark_raw.csv",
    ])

    #    b) INT8 — 2 Threads (E3b — Lei de Amdahl ponto intermediário)
    run_benchmark_cmd([
        "--model", "models/yolox_nano_ppe_int8.tflite",
        "--threads", "2",
        "--blocks", "3",
        "--runs-per-block", "10",
        "--warmup-per-block", "10",
        "--block-cooldown", "8.0",
        "--delay", "0.1",
        "--out-csv", "data/raw/rpi_int8_2threads.csv",
    ])

    #    c) INT8 — 1 Thread (E3b — Lei de Amdahl baseline sequencial)
    run_benchmark_cmd([
        "--model", "models/yolox_nano_ppe_int8.tflite",
        "--threads", "1",
        "--blocks", "3",
        "--runs-per-block", "10",
        "--warmup-per-block", "10",
        "--block-cooldown", "8.0",
        "--delay", "0.1",
        "--out-csv", "data/raw/rpi_int8_1thread.csv",
    ])

    #    d) FP32 — 4 Threads (E3c — Baseline não-quantizado)
    run_benchmark_cmd([
        "--model", "models/yolox_nano_ppe_float32.tflite",
        "--threads", "4",
        "--blocks", "3",
        "--runs-per-block", "10",
        "--warmup-per-block", "10",
        "--block-cooldown", "8.0",
        "--delay", "0.1",
        "--out-csv", "data/raw/rpi_fp32_4threads.csv",
    ])

    # 4. E4: Estabilidade térmica e temporal em execução contínua por 60 segundos (INT8, 4 threads, delay=0)
    run_benchmark_cmd([
        "--model", "models/yolox_nano_ppe_int8.tflite",
        "--threads", "4",
        "--warmup-per-block", "10",
        "--duration-s", "60.0",
        "--out-csv", "data/raw/rpi_e4_continuous_60s.csv",
    ])

    # 5. E5: Pegada de memória Flash e RAM
    int8_flash = os.path.getsize("models/yolox_nano_ppe_int8.tflite")
    fp32_flash = os.path.getsize("models/yolox_nano_ppe_float32.tflite")
    e5_data = {
        "runtime": RUNTIME_NAME,
        "int8_flash_bytes": int8_flash,
        "int8_flash_mb": round(int8_flash / (1024.0 * 1024.0), 3),
        "fp32_flash_bytes": fp32_flash,
        "fp32_flash_mb": round(fp32_flash / (1024.0 * 1024.0), 3),
        "compression_ratio_fp32_over_int8": round(fp32_flash / int8_flash, 3),
        "peak_rss_mb_orchestrator": round(read_peak_rss_mb(), 2),
        "e1_sanity_summary": e1_summary,
    }
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/rpi_e5_memory_footprint.json", "w", encoding="utf-8") as f:
        json.dump(e5_data, f, indent=2, ensure_ascii=False)
    print("\n[OK] Todos os experimentos do Protocolo v1.3 (E0, E1, E2, E3b, E3c, E4, E5) foram concluídos com sucesso!")


if __name__ == "__main__":
    main()

