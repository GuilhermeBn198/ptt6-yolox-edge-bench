#!/usr/bin/env python3
"""
Avaliação Formal do Experimento E1 — Equivalência Funcional (PC vs. SBC vs. MCU)
Conforme docs/projeto/main.tex (Seção 4.10 e Tabela 4):
  - Mesmas M=50 entradas .bin (416x416x3 uint8 BGR) executadas com o mesmo modelo INT8
    no PC, no SBC (Raspberry Pi Zero 2 W) e na MCU (ESP32-S3).
  - Critérios verificados por imagem e globalmente entre cada par de plataformas:
      1. Mesmo número de detecções após NMS
      2. Mesmas classes detectadas
      3. IoU >= 0,95 entre caixas pareadas da mesma classe
      4. |Delta pontuação| <= 1 passo de quantização na probabilidade/objetividade (out_scale = 0,027536)
  - Salva relatório detalhado em data/processed/e1_equivalence_report.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np


def box_iou(a: list[float], b: list[float]) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def compare_two_platforms(data_a: dict, data_b: dict, label_a: str, label_b: str, out_scale: float = 0.027536) -> dict:
    imgs_a = {it["image_id"]: it["detections"] for it in data_a.get("images", [])}
    imgs_b = {it["image_id"]: it["detections"] for it in data_b.get("images", [])}
    common_ids = sorted(set(imgs_a.keys()) & set(imgs_b.keys()))

    exact_count_matches = 0
    exact_class_set_matches = 0
    total_boxes_a = 0
    total_boxes_b = 0
    paired_boxes = 0
    paired_iou_ge_095 = 0
    paired_score_within_1step = 0
    all_ious = []
    all_score_diffs = []
    per_image = []

    for img_id in common_ids:
        da = imgs_a[img_id]
        db = imgs_b[img_id]
        total_boxes_a += len(da)
        total_boxes_b += len(db)

        same_count = (len(da) == len(db))
        if same_count:
            exact_count_matches += 1

        cls_a = sorted(d["cls"] for d in da)
        cls_b = sorted(d["cls"] for d in db)
        same_classes = (cls_a == cls_b)
        if same_classes:
            exact_class_set_matches += 1

        used_b = set()
        img_ious = []
        img_sdiffs = []
        for box_a in da:
            best_j = -1
            best_iou = -1.0
            for j, box_b in enumerate(db):
                if j in used_b or box_b["cls"] != box_a["cls"]:
                    continue
                iou_val = box_iou(box_a["bbox_xyxy"], box_b["bbox_xyxy"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j
            if best_j >= 0:
                used_b.add(best_j)
                paired_boxes += 1
                sdiff = abs(box_a["score"] - db[best_j]["score"])
                all_ious.append(best_iou)
                all_score_diffs.append(sdiff)
                img_ious.append(best_iou)
                img_sdiffs.append(sdiff)
                if best_iou >= 0.95:
                    paired_iou_ge_095 += 1
                if sdiff <= out_scale + 1e-4:
                    paired_score_within_1step += 1

        per_image.append({
            "image_id": img_id,
            "n_det_a": len(da),
            "n_det_b": len(db),
            "same_count": same_count,
            "same_classes": same_classes,
            "min_iou": round(float(min(img_ious)), 4) if img_ious else 1.0,
            "max_score_diff": round(float(max(img_sdiffs)), 5) if img_sdiffs else 0.0,
        })

    n_imgs = len(common_ids)
    return {
        "comparison": f"{label_a} vs {label_b}",
        "num_common_images": n_imgs,
        "total_detections_a": total_boxes_a,
        "total_detections_b": total_boxes_b,
        "paired_same_class_detections": paired_boxes,
        "images_same_detection_count": exact_count_matches,
        "images_same_detection_count_pct": round(100.0 * exact_count_matches / n_imgs, 2) if n_imgs else 0.0,
        "images_same_classes": exact_class_set_matches,
        "images_same_classes_pct": round(100.0 * exact_class_set_matches / n_imgs, 2) if n_imgs else 0.0,
        "paired_boxes_iou_ge_095": paired_iou_ge_095,
        "paired_boxes_iou_ge_095_pct": round(100.0 * paired_iou_ge_095 / paired_boxes, 2) if paired_boxes else 100.0,
        "paired_boxes_score_diff_le_1step": paired_score_within_1step,
        "paired_boxes_score_diff_le_1step_pct": round(100.0 * paired_score_within_1step / paired_boxes, 2) if paired_boxes else 100.0,
        "mean_paired_iou": round(float(np.mean(all_ious)), 4) if all_ious else 1.0,
        "min_paired_iou": round(float(np.min(all_ious)), 4) if all_ious else 1.0,
        "mean_abs_score_diff": round(float(np.mean(all_score_diffs)), 5) if all_score_diffs else 0.0,
        "max_abs_score_diff": round(float(np.max(all_score_diffs)), 5) if all_score_diffs else 0.0,
        "quant_step_out_scale": out_scale,
        "per_image": per_image,
    }


def main() -> int:
    pc_path = Path("data/raw/pc_e1_50_predictions.json")
    pc_tflm_path = Path("data/raw/pc_tflm_e1_50_predictions.json")
    sbc_path = Path("data/raw/sbc_e1_50_predictions.json")
    mcu_path = Path("data/raw/mcu_e1_50_predictions.json")

    loaded = {}
    for label, p in [("PC", pc_path), ("PC_TFLM", pc_tflm_path), ("SBC", sbc_path), ("MCU", mcu_path)]:
        if p.exists():
            loaded[label] = json.loads(p.read_text(encoding="utf-8"))

    comparisons = []
    for la, lb in [("PC", "SBC"), ("PC", "MCU"), ("SBC", "MCU"), ("PC_TFLM", "MCU"), ("PC_TFLM", "SBC")]:
        if la in loaded and lb in loaded:
            comparisons.append(compare_two_platforms(loaded[la], loaded[lb], la, lb))

    report = {
        "experiment": "E1 — Equivalência Funcional (main.tex §4.10)",
        "platforms_evaluated": list(loaded.keys()),
        "criteria": {
            "num_inputs": 50,
            "channel_order": "BGR",
            "input_res": 416,
            "min_iou_threshold": 0.95,
            "max_score_diff_quant_steps": 1,
            "out_scale": 0.027536,
        },
        "comparisons": comparisons,
    }

    out_path = Path("data/processed/e1_equivalence_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=========================================================================")
    print("  RELATÓRIO E1 — EQUIVALÊNCIA FUNCIONAL NAS M=50 ENTRADAS BGR")
    print("=========================================================================")
    for c in comparisons:
        print(f"  * {c['comparison']} ({c['num_common_images']} imagens):")
        print(f"    - Detecções totais: {c['total_detections_a']} vs {c['total_detections_b']} (pareadas: {c['paired_same_class_detections']})")
        print(f"    - Mesmo nº detecções/img: {c['images_same_detection_count']}/{c['num_common_images']} ({c['images_same_detection_count_pct']}%)")
        print(f"    - Mesmas classes/img    : {c['images_same_classes']}/{c['num_common_images']} ({c['images_same_classes_pct']}%)")
        print(f"    - Caixas com IoU >= 0.95: {c['paired_boxes_iou_ge_095']}/{c['paired_same_class_detections']} ({c['paired_boxes_iou_ge_095_pct']}%) | IoU médio={c['mean_paired_iou']}")
        print(f"    - |Delta score| <= 1 step: {c['paired_boxes_score_diff_le_1step']}/{c['paired_same_class_detections']} ({c['paired_boxes_score_diff_le_1step_pct']}%) | máx={c['max_abs_score_diff']}")
    print(f"[OK] Salvo em '{out_path}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
