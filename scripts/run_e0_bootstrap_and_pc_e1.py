#!/usr/bin/env python3
"""Executa no PC (usando /tmp/tflm_host_eval em paralelo nos núcleos da CPU):
1. Experimento E1 (Referência PC): inferência do modelo INT8 nas M=50 imagens BGR (.bin)
   com limiares de operação (conf >= 0.25, IoU_NMS = 0.45, max 100 dets).
   Salva: data/raw/pc_e1_50_predictions.json
2. Experimento E0 (Acurácia COCO no split de teste: 141 imagens, 1.251 anotações, BGR):
   Avalia FP32 e INT8 com conf = 0.001, IoU_NMS = 0.65, max 100 dets.
   Calcula mAP@50, mAP@50:95, AP@50 por classe (destaque para 'vest') e a retenção
   rho = mAP_int8 / mAP_fp32 com Intervalo de Confiança de 95% via Bootstrap de imagens
   (B = 1.000 reamostras com reposição sobre as 141 imagens de teste, main.tex §4.3 e §4.6).
   Salva: data/processed/e0_map_test.json
"""
from __future__ import annotations

import concurrent.futures
import glob
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

NAMES = [
    "helmet", "gloves", "vest", "boots", "goggles", "none", "Person",
    "no_helmet", "no_goggle", "no_gloves", "no_boots"
]
IOU_THRS = np.linspace(0.50, 0.95, 10)  # 0.50:0.05:0.95
REC_THRS = np.linspace(0.0, 1.0, 101)   # 101-point COCO PR interpolation


def prepare_all_141_test_bins(dataset_dir: str, out_dir: str, r: int = 416) -> list[dict]:
    """Converte as 141 imagens do split de teste para .bin BGR 416x416 (pad 114) e extrai GTs."""
    os.makedirs(out_dir, exist_ok=True)
    imgs = sorted(glob.glob(os.path.join(dataset_dir, "images", "test", "*")))
    items = []
    for img_id, p in enumerate(imgs):
        stem = os.path.splitext(os.path.basename(p))[0]
        bin_path = os.path.join(out_dir, f"{img_id:03d}_{stem}.bin")
        with Image.open(p) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        rr = min(r / h, r / w)
        new_w, new_h = int(w * rr), int(h * rr)
        if not os.path.exists(bin_path):
            resized_rgb = np.asarray(
                Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
            pad = np.full((r, r, 3), 114, dtype=np.uint8)
            pad[:new_h, :new_w, :] = resized_rgb[:, :, ::-1]  # BGR
            with open(bin_path, "wb") as f:
                f.write(pad.tobytes())

        lab_path = os.path.join(dataset_dir, "labels", "test", f"{stem}.txt")
        gts = []
        if os.path.exists(lab_path):
            with open(lab_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    c, cx, cy, bw, bh = map(float, line.split()[:5])
                    x1 = (cx - bw / 2.0) * w
                    y1 = (cy - bh / 2.0) * h
                    x2 = (cx + bw / 2.0) * w
                    y2 = (cy + bh / 2.0) * h
                    gts.append({"cls": int(c), "bbox": [x1, y1, x2, y2]})

        items.append({
            "img_id": img_id,
            "file_name": os.path.basename(p),
            "stem": stem,
            "bin_path": bin_path,
            "width": w,
            "height": h,
            "scale": rr,
            "gts": gts,
        })
    return items


def _run_chunk(args):
    runner, model_path, conf_thr, nms_thr, bin_paths = args
    cmd = [runner, model_path, str(conf_thr), str(nms_thr), *bin_paths]
    out = subprocess.check_output(cmd, text=True)
    res_map = {}
    det_map = {bp: [] for bp in bin_paths}
    for line in out.splitlines():
        parts = line.strip().split(",")
        if not parts:
            continue
        if parts[0] == "RES":
            bp = parts[1]
            res_map[bp] = {
                "t_pre_us": int(parts[2]),
                "t_inf_us": int(parts[3]),
                "t_post_us": int(parts[4]),
                "t_total_us": int(parts[5]),
                "n_det": int(parts[6]),
            }
        elif parts[0] == "DET":
            bp = parts[1]
            det_map[bp].append({
                "bbox_xyxy": [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])],
                "score": float(parts[6]),
                "cls": int(parts[7]),
            })
    return res_map, det_map


def run_tflm_parallel(runner: str, model_path: str, bin_paths: list[str], conf_thr: float, nms_thr: float):
    n_workers = min(os.cpu_count() or 4, len(bin_paths), 12)
    chunks = [list(c) for c in np.array_split(bin_paths, n_workers) if len(c) > 0]
    tasks = [(runner, model_path, conf_thr, nms_thr, c) for c in chunks]
    all_res = {}
    all_dets = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
        for rm, dm in ex.map(_run_chunk, tasks):
            all_res.update(rm)
            all_dets.update(dm)
    return all_res, all_dets


def box_iou_matrix(dt_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Calcula IoU [D, G] entre detecções [D, 4] e ground-truths [G, 4] em formato xyxy."""
    if len(dt_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(dt_boxes), len(gt_boxes)), dtype=np.float64)
    x1 = np.maximum(dt_boxes[:, None, 0], gt_boxes[None, :, 0])
    y1 = np.maximum(dt_boxes[:, None, 1], gt_boxes[None, :, 1])
    x2 = np.minimum(dt_boxes[:, None, 2], gt_boxes[None, :, 2])
    y2 = np.minimum(dt_boxes[:, None, 3], gt_boxes[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_d = np.maximum(0.0, dt_boxes[:, 2] - dt_boxes[:, 0]) * np.maximum(0.0, dt_boxes[:, 3] - dt_boxes[:, 1])
    area_g = np.maximum(0.0, gt_boxes[:, 2] - gt_boxes[:, 0]) * np.maximum(0.0, gt_boxes[:, 3] - gt_boxes[:, 1])
    union = area_d[:, None] + area_g[None, :] - inter + 1e-9
    return inter / union


def precompute_coco_matches(items: list[dict], det_map: dict, num_classes: int = 11):
    """
    Pré-calcula o pareamento COCO por (img_idx, cls_id):
      - n_gt[img_idx, cls_id]: número de anotações GT
      - dt_scores[img_idx][cls_id]: array 1D de scores ordenados das detecções
      - dt_tp[img_idx][cls_id]: array booleano [T=10, D] indicando True Positive em cada limiar IoU
    """
    n_imgs = len(items)
    T = len(IOU_THRS)
    n_gt = np.zeros((n_imgs, num_classes), dtype=np.int32)
    dt_scores = [[np.empty(0, dtype=np.float64) for _ in range(num_classes)] for _ in range(n_imgs)]
    dt_tp = [[np.empty((T, 0), dtype=bool) for _ in range(num_classes)] for _ in range(n_imgs)]

    for i, item in enumerate(items):
        scale = item["scale"]
        raw_dets = det_map.get(item["bin_path"], [])
        for c in range(num_classes):
            gts_c = [g["bbox"] for g in item["gts"] if g["cls"] == c]
            G = len(gts_c)
            n_gt[i, c] = G

            dets_c = [d for d in raw_dets if d["cls"] == c]
            if not dets_c:
                continue
            # Ordenar detecções por score decrescente
            dets_c.sort(key=lambda x: x["score"], reverse=True)
            scores = np.array([d["score"] for d in dets_c], dtype=np.float64)
            # Desfazer escala letterbox (igual a eval_map_tflite.py linha 116: b / rr)
            d_boxes = np.array([d["bbox_xyxy"] for d in dets_c], dtype=np.float64) / scale
            D = len(dets_c)
            tp = np.zeros((T, D), dtype=bool)

            if G > 0:
                g_boxes = np.array(gts_c, dtype=np.float64)
                ious = box_iou_matrix(d_boxes, g_boxes)
                for t_idx, thr in enumerate(IOU_THRS):
                    gt_matched = np.zeros(G, dtype=bool)
                    for d_idx in range(D):
                        best_g = -1
                        best_iou = thr
                        for g_idx in range(G):
                            if not gt_matched[g_idx] and ious[d_idx, g_idx] >= best_iou:
                                best_iou = ious[d_idx, g_idx]
                                best_g = g_idx
                        if best_g >= 0:
                            gt_matched[best_g] = True
                            tp[t_idx, d_idx] = True

            dt_scores[i][c] = scores
            dt_tp[i][c] = tp

    return n_gt, dt_scores, dt_tp


def accumulate_coco_metrics(img_indices: np.ndarray, n_gt: np.ndarray, dt_scores: list, dt_tp: list, num_classes: int = 11):
    """Calcula mAP@50, mAP@50:95 e AP@50 por classe para um conjunto (ou amostra bootstrap) de imagens."""
    T = len(IOU_THRS)
    R = len(REC_THRS)
    precisions = np.full((T, R, num_classes), -1.0, dtype=np.float64)

    for c in range(num_classes):
        npig = int(np.sum(n_gt[img_indices, c]))
        if npig == 0:
            continue
        scores_list = [dt_scores[i][c] for i in img_indices if len(dt_scores[i][c]) > 0]
        if not scores_list:
            precisions[:, :, c] = 0.0
            continue
        tp_list = [dt_tp[i][c] for i in img_indices if len(dt_scores[i][c]) > 0]
        scores_all = np.concatenate(scores_list, axis=0)
        tp_all = np.concatenate(tp_list, axis=1)  # [T, D_total]

        order = np.argsort(-scores_all, kind="mergesort")
        tp_sorted = tp_all[:, order]
        fp_sorted = ~tp_sorted

        tp_cum = np.cumsum(tp_sorted, axis=1).astype(np.float64)
        fp_cum = np.cumsum(fp_sorted, axis=1).astype(np.float64)

        rc = tp_cum / float(npig)
        pr = tp_cum / (tp_cum + fp_cum + np.spacing(1))

        # Envelope monotônico decrescente da precisão da direita para a esquerda
        for t_idx in range(T):
            p_row = pr[t_idx].copy()
            for k in range(len(p_row) - 1, 0, -1):
                if p_row[k] > p_row[k - 1]:
                    p_row[k - 1] = p_row[k]
            inds = np.searchsorted(rc[t_idx], REC_THRS, side="left")
            q = np.zeros(R, dtype=np.float64)
            valid = inds < len(p_row)
            q[valid] = p_row[inds[valid]]
            precisions[t_idx, :, c] = q

    valid_50 = precisions[0, :, :] > -1
    map50 = float(np.mean(precisions[0, :, :][valid_50])) if np.any(valid_50) else 0.0
    valid_all = precisions > -1
    map50_95 = float(np.mean(precisions[valid_all])) if np.any(valid_all) else 0.0

    ap50_per_class = {}
    for c, name in enumerate(NAMES):
        pr_c = precisions[0, :, c]
        ap50_per_class[name] = float(np.mean(pr_c[pr_c > -1])) if np.any(pr_c > -1) else 0.0

    return map50, map50_95, ap50_per_class


def main() -> int:
    runner = "/tmp/tflm_host_eval"
    if not os.path.exists(runner):
        raise FileNotFoundError(f"Executável {runner} não encontrado!")

    # 1. Experimento E1 (Referência PC nas M=50 imagens BGR com conf=0.25, NMS=0.45)
    bins_50 = sorted(glob.glob("/tmp/inputs_50_bgr_416/*.bin"))
    print(f"[E1-PC] Executando modelo INT8 nas {len(bins_50)} imagens BGR de referência (conf=0.25, NMS=0.45)...")
    res_50, dets_50 = run_tflm_parallel(runner, "models/yolox_nano_ppe_int8.tflite", bins_50, 0.25, 0.45)

    with open("models/yolox_nano_ppe_int8.tflite", "rb") as f:
        int8_sha = hashlib.sha256(f.read()).hexdigest()

    pc_e1_entries = []
    total_pc_dets = 0
    for bp in bins_50:
        img_id = os.path.splitext(os.path.basename(bp))[0]
        d_list = dets_50[bp]
        total_pc_dets += len(d_list)
        pc_e1_entries.append({
            "image_id": img_id,
            "timing_us": res_50[bp],
            "num_detections": len(d_list),
            "detections": d_list,
        })

    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/pc_e1_50_predictions.json", "w", encoding="utf-8") as f:
        json.dump({
            "platform": "PC (TFLite Micro Reference C++ x86_64)",
            "model": "models/yolox_nano_ppe_int8.tflite",
            "model_sha256": int8_sha,
            "channel_order": "BGR",
            "input_res": 416,
            "conf_thr": 0.25,
            "nms_thr": 0.45,
            "num_images": len(bins_50),
            "total_detections": total_pc_dets,
            "images": pc_e1_entries,
        }, f, indent=2, ensure_ascii=False)
    print(f"[OK] E1-PC salvo em 'data/raw/pc_e1_50_predictions.json' ({total_pc_dets} detecções em {len(bins_50)} imagens).")

    # 2. Experimento E0 (Acurácia nas 141 imagens do split de teste + Bootstrap B=1000)
    print("[E0] Preparando as 141 imagens BGR do split de teste...")
    items_141 = prepare_all_141_test_bins("/tmp/construction-ppe", "/tmp/test_141_bgr_416", r=416)
    bins_141 = [it["bin_path"] for it in items_141]

    print("[E0] Avaliando yolox_nano_ppe_int8.tflite nas 141 imagens de teste (conf=0.001, NMS=0.65)...")
    _, dets_int8_141 = run_tflm_parallel(runner, "models/yolox_nano_ppe_int8.tflite", bins_141, 0.001, 0.65)

    print("[E0] Avaliando yolox_nano_ppe_float32.tflite nas 141 imagens de teste (conf=0.001, NMS=0.65)...")
    _, dets_fp32_141 = run_tflm_parallel(runner, "models/yolox_nano_ppe_float32.tflite", bins_141, 0.001, 0.65)

    n_gt_fp32, sc_fp32, tp_fp32 = precompute_coco_matches(items_141, dets_fp32_141)
    n_gt_int8, sc_int8, tp_int8 = precompute_coco_matches(items_141, dets_int8_141)

    all_idx = np.arange(len(items_141))
    fp32_m50, fp32_m50_95, fp32_ap_cls = accumulate_coco_metrics(all_idx, n_gt_fp32, sc_fp32, tp_fp32)
    int8_m50, int8_m50_95, int8_ap_cls = accumulate_coco_metrics(all_idx, n_gt_int8, sc_int8, tp_int8)

    # Bootstrap de imagens (B = 1.000 reamostras pareadas entre FP32 e INT8, main.tex §4.3 e §4.6)
    n_boot = 1000
    rng = np.random.default_rng(2026)
    b_fp32_m50 = np.empty(n_boot)
    b_fp32_m50_95 = np.empty(n_boot)
    b_fp32_vest = np.empty(n_boot)
    b_int8_m50 = np.empty(n_boot)
    b_int8_m50_95 = np.empty(n_boot)
    b_int8_vest = np.empty(n_boot)
    b_rho_m50 = np.empty(n_boot)
    b_rho_m50_95 = np.empty(n_boot)
    b_rho_vest = np.empty(n_boot)

    print(f"[E0] Calculando Intervalos de Confiança de 95% via Bootstrap ({n_boot} reamostras de imagens)...")
    for b in range(n_boot):
        sample_idx = rng.integers(0, len(items_141), size=len(items_141))
        m50_f, m95_f, cls_f = accumulate_coco_metrics(sample_idx, n_gt_fp32, sc_fp32, tp_fp32)
        m50_i, m95_i, cls_i = accumulate_coco_metrics(sample_idx, n_gt_int8, sc_int8, tp_int8)
        b_fp32_m50[b] = m50_f
        b_fp32_m50_95[b] = m95_f
        b_fp32_vest[b] = cls_f["vest"]
        b_int8_m50[b] = m50_i
        b_int8_m50_95[b] = m95_i
        b_int8_vest[b] = cls_i["vest"]
        b_rho_m50[b] = (m50_i / m50_f) if m50_f > 0 else 0.0
        b_rho_m50_95[b] = (m95_i / m95_f) if m95_f > 0 else 0.0
        b_rho_vest[b] = (cls_i["vest"] / cls_f["vest"]) if cls_f["vest"] > 0 else 0.0

    def ci95(arr):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]

    # Atualizar data/processed/e0_map_test.json preservando os campos existentes e adicionando Bootstrap + rho
    e0_path = Path("data/processed/e0_map_test.json")
    existing = json.loads(e0_path.read_text(encoding="utf-8")) if e0_path.exists() else {}

    rho_50 = round(int8_m50 / fp32_m50, 4) if fp32_m50 > 0 else 0.0
    rho_50_95 = round(int8_m50_95 / fp32_m50_95, 4) if fp32_m50_95 > 0 else 0.0
    rho_vest = round(int8_ap_cls["vest"] / fp32_ap_cls["vest"], 4) if fp32_ap_cls["vest"] > 0 else 0.0

    existing.update({
        "split": "test",
        "n_images": len(items_141),
        "n_gt": int(np.sum(n_gt_fp32)),
        "bootstrap_resamples_B": n_boot,
        "bootstrap_ci_95_summary": {
            "fp32_BGR": {
                "mAP50": round(fp32_m50, 4),
                "mAP50_ci95": ci95(b_fp32_m50),
                "mAP50_95": round(fp32_m50_95, 4),
                "mAP50_95_ci95": ci95(b_fp32_m50_95),
                "AP50_vest": round(fp32_ap_cls["vest"], 4),
                "AP50_vest_ci95": ci95(b_fp32_vest),
                "AP50_per_class": {k: round(v, 4) for k, v in fp32_ap_cls.items()},
            },
            "int8_BGR": {
                "mAP50": round(int8_m50, 4),
                "mAP50_ci95": ci95(b_int8_m50),
                "mAP50_95": round(int8_m50_95, 4),
                "mAP50_95_ci95": ci95(b_int8_m50_95),
                "AP50_vest": round(int8_ap_cls["vest"], 4),
                "AP50_vest_ci95": ci95(b_int8_vest),
                "AP50_per_class": {k: round(v, 4) for k, v in int8_ap_cls.items()},
            },
            "retention_rho_int8_over_fp32_BGR": {
                "target_mlperf_min": 0.99,
                "rho_mAP50": rho_50,
                "rho_mAP50_ci95": ci95(b_rho_m50),
                "rho_mAP50_95": rho_50_95,
                "rho_mAP50_95_ci95": ci95(b_rho_m50_95),
                "rho_AP50_vest": rho_vest,
                "rho_AP50_vest_ci95": ci95(b_rho_vest),
                "meets_mlperf_99pct_target": bool(rho_50 >= 0.99),
            },
        },
    })

    e0_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] E0 com Bootstrap (1000 reamostras) salvo em '{e0_path}':")
    print(f"  * FP32 (BGR): mAP@50 = {fp32_m50:.4f} IC95%={ci95(b_fp32_m50)} | mAP@50:95 = {fp32_m50_95:.4f} IC95%={ci95(b_fp32_m50_95)} | AP@50(vest) = {fp32_ap_cls['vest']:.4f} IC95%={ci95(b_fp32_vest)}")
    print(f"  * INT8 (BGR): mAP@50 = {int8_m50:.4f} IC95%={ci95(b_int8_m50)} | mAP@50:95 = {int8_m50_95:.4f} IC95%={ci95(b_int8_m50_95)} | AP@50(vest) = {int8_ap_cls['vest']:.4f} IC95%={ci95(b_int8_vest)}")
    print(f"  * Retenção rho (INT8/FP32): rho_mAP@50 = {rho_50:.4f} IC95%={ci95(b_rho_m50)} | rho_mAP@50:95 = {rho_50_95:.4f} IC95%={ci95(b_rho_m50_95)} | rho_vest = {rho_vest:.4f} IC95%={ci95(b_rho_vest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
