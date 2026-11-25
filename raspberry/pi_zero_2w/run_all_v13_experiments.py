#!/usr/bin/env python3
"""
Orquestrador Completo dos Experimentos no SBC (Raspberry Pi Zero 2 W) — Protocolo v1.5 (docs/projeto/main.tex)
Projeto PTT6 - Arquitetura e Organização de Computadores (PPCA/UFRR)
Orientador: Prof. Dr. Herbert Oliveira Rocha
Alunos: Filipe Matias & Guilherme Lucas

Executa todos os experimentos do SBC exigidos em docs/projeto/main.tex (Tabela 4):
  - E0_idle : Baseline ocioso (3 blocos de 30 s sem carga de inferência)
  - E1      : Equivalência funcional sobre as mesmas M=50 entradas .bin BGR (salva todas as detecções por imagem
              em sbc_e1_50_predictions.json + comparação FP32 vs INT8 nas 50 entradas em rpi_e1_quantization_sanity.csv)
  - E2      : Piloto de 100 inferências consecutivas percorrendo as 50 entradas BGR; cálculo empírico
              do aquecimento W (janelas de 10, <2% por 2 janelas seguidas, piso 5) e do tamanho amostral
              N = ceil((1.96 * CV / 0.01)^2), N_final = max(30, N) a partir do CV de t_total (Equação 1)
  - E3      : Benchmark principal em 3 blocos independentes percorrendo as 50 entradas BGR em ordem fixa
              (INT8 4 threads para H1/H2 e FP32 4 threads para H2), gerando CSVs com e sem warmup + metadados
  - E3b     : Escalabilidade multi-core (1, 2 e 4 threads — Lei de Amdahl) sobre as 50 entradas BGR em RAM
  - E4      : Throughput sustentado em 60 segundos de laço fechado (delay=0) e deriva térmica
  - E5_mem  : Pegada de memória Flash (.tflite) e RAM de pico (VmHWM / RSS)
"""

import json
import math
import os
import subprocess
import sys
import time
import numpy as np

from benchmark_yolox import (
    Interpreter,
    RUNTIME_NAME,
    build_yolox_grids,
    compute_file_sha256,
    load_bgr_bin_inputs,
    postprocess_yolox,
    preprocess_in_memory,
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
    """Baseline ocioso: 3 janelas de 30 s sem carga de inferência."""
    print("\n=======================================================================")
    print("  [Baseline Ocioso] 3 x 30 s sem carga de inferência                  ")
    print("=======================================================================")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    lines = ["sample_idx,block,elapsed_block_s,cpu_temp_c,cpu_freq_mhz,throttled_hex\n"]
    sample_idx = 0
    for b in range(1, blocks + 1):
        print(f"  [Ocioso] Bloco {b}/{blocks} ({duration_per_block_s}s em repouso)...")
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
    print(f"[OK] Baseline ocioso concluído: {sample_idx} amostras salvas em '{out_csv}'")


def run_e1_50_bgr_equivalence_and_sanity(
    fp32_model="models/yolox_nano_ppe_float32.tflite",
    int8_model="models/yolox_nano_ppe_int8.tflite",
    inputs_dir="inputs_50_bgr_416",
    fallback_image="raspberry/pi_zero_2w/test_image.jpg",
    out_sbc_e1_json="data/raw/sbc_e1_50_predictions.json",
    out_csv="data/raw/rpi_e1_quantization_sanity.csv",
    out_json="data/raw/rpi_e1_quantization_sanity.json",
):
    """
    Experimento E1 (Seção 4.10 de main.tex):
    Executa o modelo INT8 (e o FP32 de referência) sobre as mesmas M=50 entradas .bin BGR em memória,
    salvando todas as detecções por imagem do SBC para verificação de equivalência funcional
    (PC vs. SBC vs. MCU: mesmas classes, nº de detecções, IoU >= 0.95 e |Delta score| <= 1 passo).
    """
    print("\n=======================================================================")
    print("  [E1] Equivalência Funcional nas M=50 Entradas BGR (SBC)             ")
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
    bgr_inputs = load_bgr_bin_inputs(inputs_dir, fallback_image, target_h, target_w)
    int8_sha = compute_file_sha256(int8_model)

    csv_lines = ["sample_id,image_name,fp32_dets,int8_dets,matched_dets_iou50,mean_iou,tensor_mae,tensor_corr\n"]
    maes, corrs, ious_all = [], [], []
    total_fp32_dets = 0
    total_int8_dets = 0
    total_matched_dets = 0
    sbc_e1_entries = []

    for idx, (img_id, bgr_arr) in enumerate(bgr_inputs):
        # INT8 cronometrado em 3 fases
        t0 = time.perf_counter_ns()
        t_in_int8 = preprocess_in_memory(bgr_arr, in_int8["dtype"], in_scale, in_zp)
        interp_int8.set_tensor(in_int8["index"], t_in_int8)
        t1 = time.perf_counter_ns()
        interp_int8.invoke()
        t2 = time.perf_counter_ns()
        raw_y_int8 = interp_int8.get_tensor(out_int8["index"])
        dets_int8 = postprocess_yolox(raw_y_int8, out_int8["dtype"], out_scale, out_zp, grids, expanded_strides)
        t3 = time.perf_counter_ns()

        d_list = []
        for d in dets_int8:
            d_list.append({
                "bbox_xyxy": [round(float(d[0]), 2), round(float(d[1]), 2), round(float(d[2]), 2), round(float(d[3]), 2)],
                "score": round(float(d[4]), 4),
                "cls": int(d[5]),
            })
        sbc_e1_entries.append({
            "image_id": img_id,
            "timing_us": {
                "t_pre_us": int((t1 - t0) // 1000),
                "t_inf_us": int((t2 - t1) // 1000),
                "t_post_us": int((t3 - t2) // 1000),
                "t_total_us": int((t3 - t0) // 1000),
                "n_det": len(d_list),
            },
            "num_detections": len(d_list),
            "detections": d_list,
        })

        # FP32 para comparação de sanidade INT8 vs FP32 nas mesmas 50 imagens BGR
        t_in_fp32 = preprocess_in_memory(bgr_arr, in_fp32["dtype"], 1.0, 0)
        interp_fp32.set_tensor(in_fp32["index"], t_in_fp32)
        interp_fp32.invoke()
        raw_y_fp32 = interp_fp32.get_tensor(out_fp32["index"])
        dets_fp32 = postprocess_yolox(raw_y_fp32, out_fp32["dtype"], 1.0, 0, grids, expanded_strides)

        deq_y_int8 = (raw_y_int8.astype(np.float32) - float(out_zp)) * float(out_scale)
        mae = float(np.mean(np.abs(raw_y_fp32 - deq_y_int8)))
        corr = float(np.corrcoef(raw_y_fp32.ravel(), deq_y_int8.ravel())[0, 1])
        maes.append(mae)
        corrs.append(corr)

        n_fp32 = len(dets_fp32)
        n_int8 = len(dets_int8)
        total_fp32_dets += n_fp32
        total_int8_dets += n_int8

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
            f"{idx+1},{img_id},{n_fp32},{n_int8},{matched},{mean_iou:.4f},{mae:.5f},{corr:.5f}\n"
        )

    os.makedirs(os.path.dirname(out_sbc_e1_json) or ".", exist_ok=True)
    with open(out_sbc_e1_json, "w", encoding="utf-8") as f:
        json.dump({
            "platform": f"SBC Raspberry Pi Zero 2 W ({RUNTIME_NAME})",
            "model": int8_model,
            "model_sha256": int8_sha,
            "channel_order": "BGR",
            "input_res": target_w,
            "out_scale": round(float(out_scale), 6),
            "out_zp": int(out_zp),
            "conf_thr": 0.25,
            "nms_thr": 0.45,
            "num_images": len(sbc_e1_entries),
            "total_detections": total_int8_dets,
            "images": sbc_e1_entries,
        }, f, indent=2, ensure_ascii=False)

    with open(out_csv, "w", encoding="utf-8") as f:
        f.writelines(csv_lines)

    recall_vs_fp32 = (total_matched_dets / total_fp32_dets) if total_fp32_dets > 0 else 1.0
    summary = {
        "num_test_samples": len(sbc_e1_entries),
        "channel_order": "BGR",
        "total_fp32_detections": int(total_fp32_dets),
        "total_int8_detections": int(total_int8_dets),
        "total_int8_matched_iou50": int(total_matched_dets),
        "box_agreement_recall_iou50": round(float(recall_vs_fp32), 4),
        "mean_best_iou": round(float(np.mean(ious_all)), 4),
        "mean_tensor_mae": round(float(np.mean(maes)), 5),
        "mean_tensor_pearson_r": round(float(np.mean(corrs)), 5),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] E1 SBC concluído: {total_int8_dets} detecções INT8 em {len(sbc_e1_entries)} imagens BGR salvas em '{out_sbc_e1_json}'")
    print(f"     Correlação tensorial FP32 vs INT8 r = {np.mean(corrs):.4f}, MAE = {np.mean(maes):.5f}")
    return summary


def evaluate_e2_pilot_csv(pilot_csv="data/raw/rpi_e2_pilot_100.csv", out_json="data/raw/sbc_e2_pilot_summary.json") -> dict:
    """
    Analisa as 100 inferências consecutivas do Piloto E2 (Seção 4.10 e Equação 1 de main.tex):
      - Janelas de 10 inferências: descarta iniciais até variação relativa da mediana entre
        janelas consecutivas ficar < 2% por duas janelas seguidas (piso de 5).
      - Com o CV de t_total após o warm-up: N = ceil((1.96 * CV / 0.01)^2), N_final = max(30, N).
    """
    with open(pilot_csv, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    hdr = lines[0].split(",")
    col = {n: i for i, n in enumerate(hdr)}
    t_tot = np.array([float(r.split(",")[col["t_total_ms"]]) for r in lines[1:]], dtype=np.float64)
    t_inf = np.array([float(r.split(",")[col["t_inf_ms"]]) for r in lines[1:]], dtype=np.float64)

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
    std_t = float(np.std(post_warmup, ddof=1))
    cv = std_t / mean_t if mean_t > 0 else 0.0
    n_calc = int(math.ceil(((1.96 * cv) / 0.01) ** 2))
    n_final = max(30, n_calc)

    res = {
        "platform": "SBC (Raspberry Pi Zero 2 W)",
        "pilot_total_inferences": len(t_tot),
        "window_size": win_size,
        "window_medians_t_total_ms": [round(x, 3) for x in win_meds],
        "window_rel_diffs_pct": [round(x * 100.0, 3) for x in rel_diffs],
        "warmup_W_empirical": w_emp,
        "warmup_W_adopted_per_block": max(10, w_emp),
        "post_warmup_t_total_median_ms": round(float(np.median(post_warmup)), 3),
        "post_warmup_t_total_mean_ms": round(mean_t, 3),
        "post_warmup_t_total_std_ms": round(std_t, 3),
        "post_warmup_t_inf_median_ms": round(float(np.median(t_inf[w_emp:])), 3),
        "cv_t_total": round(cv, 6),
        "cv_t_total_percent": round(cv * 100.0, 3),
        "n_formula_eq1_eps1pct": n_calc,
        "n_final_adopted": n_final,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"[OK] Piloto E2 SBC: W_emp={w_emp}, CV(t_total)={cv*100:.3f}% -> N_eq1={n_calc} -> N_final={n_final}")
    return res


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

    # 1. Baseline ocioso (3 x 30s)
    run_e0_idle_baseline(blocks=3, duration_per_block_s=30, interval_s=2.0)

    # 2. E1: Equivalência funcional nas M=50 entradas BGR (.bin)
    e1_summary = run_e1_50_bgr_equivalence_and_sanity()

    # 3. E2: Piloto formal de 100 inferências consecutivas sobre as 50 entradas BGR em RAM
    run_benchmark_cmd([
        "--model", "models/yolox_nano_ppe_int8.tflite",
        "--experiment", "E2",
        "--threads", "4",
        "--blocks", "1",
        "--runs-per-block", "100",
        "--warmup-per-block", "0",
        "--delay", "0.02",
        "--out-csv", "data/raw/rpi_e2_pilot_100.csv",
        "--out-std-csv", "data/raw/SBC_E2_b1.csv",
    ])
    e2_summary = evaluate_e2_pilot_csv("data/raw/rpi_e2_pilot_100.csv", "data/raw/sbc_e2_pilot_summary.json")

    # 4. E3 & E3b: 3 blocos independentes x 10 execuções válidas (= 30) + 10 warmups/bloco percorrendo as 50 entradas BGR
    #    a) INT8 — 4 Threads (E3 principal para H1/H2 e E3b)
    run_benchmark_cmd([
        "--model", "models/yolox_nano_ppe_int8.tflite",
        "--experiment", "E3",
        "--threads", "4",
        "--blocks", "3",
        "--runs-per-block", "10",
        "--warmup-per-block", "10",
        "--block-cooldown", "8.0",
        "--delay", "0.05",
        "--out-csv", "data/raw/rpi_benchmark_raw.csv",
        "--out-std-csv", "data/raw/SBC_E3_int8_4t.csv",
    ])

    #    b) INT8 — 2 Threads (E3b — Lei de Amdahl ponto intermediário)
    run_benchmark_cmd([
        "--model", "models/yolox_nano_ppe_int8.tflite",
        "--experiment", "E3b",
        "--threads", "2",
        "--blocks", "3",
        "--runs-per-block", "10",
        "--warmup-per-block", "10",
        "--block-cooldown", "8.0",
        "--delay", "0.05",
        "--out-csv", "data/raw/rpi_int8_2threads.csv",
        "--out-std-csv", "data/raw/SBC_E3b_int8_2t.csv",
    ])

    #    c) INT8 — 1 Thread (E3b — Lei de Amdahl baseline sequencial)
    run_benchmark_cmd([
        "--model", "models/yolox_nano_ppe_int8.tflite",
        "--experiment", "E3b",
        "--threads", "1",
        "--blocks", "3",
        "--runs-per-block", "10",
        "--warmup-per-block", "10",
        "--block-cooldown", "8.0",
        "--delay", "0.05",
        "--out-csv", "data/raw/rpi_int8_1thread.csv",
        "--out-std-csv", "data/raw/SBC_E3b_int8_1t.csv",
    ])

    #    d) FP32 — 4 Threads (E3 para H2 — trade-off int8 vs fp32 no SBC)
    run_benchmark_cmd([
        "--model", "models/yolox_nano_ppe_float32.tflite",
        "--experiment", "E3",
        "--threads", "4",
        "--blocks", "3",
        "--runs-per-block", "10",
        "--warmup-per-block", "10",
        "--block-cooldown", "8.0",
        "--delay", "0.05",
        "--out-csv", "data/raw/rpi_fp32_4threads.csv",
        "--out-std-csv", "data/raw/SBC_E3_fp32_4t.csv",
    ])

    # 5. E4: Estabilidade térmica e throughput sustentado em execução contínua por 60 segundos (INT8, 4 threads, delay=0)
    run_benchmark_cmd([
        "--model", "models/yolox_nano_ppe_int8.tflite",
        "--experiment", "E4",
        "--threads", "4",
        "--warmup-per-block", "10",
        "--duration-s", "60.0",
        "--out-csv", "data/raw/rpi_e4_continuous_60s.csv",
    ])

    # 6. Pegada de memória Flash e RAM
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
        "e2_pilot_summary": e2_summary,
    }
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/rpi_e5_memory_footprint.json", "w", encoding="utf-8") as f:
        json.dump(e5_data, f, indent=2, ensure_ascii=False)
    print("\n[OK] Todos os experimentos do SBC (Ocioso, E1 50 BGR, E2 Piloto 100, E3, E3b, E4 60s, Memória) foram concluídos com sucesso!")


if __name__ == "__main__":
    main()
