#!/usr/bin/env python3
"""E0 — acurácia (mAP COCO) de artefatos TFLite do YOLOX-Nano no Construction-PPE (protocolo v1.5).

Avalia fp32 e int8 no MESMO split, com o MESMO pré/pós-processamento usado nos dispositivos:
letterbox 416x416 (pad 114, canto superior esquerdo), sem normalização, decodificação da cabeça
YOLOX (strides 8/16/32), score = obj * cls, NMS por classe. Para mAP usa conf=0.001 e NMS=0.65
(valores de avaliação do YOLOX); limiares de implantação (0.25/0.45) não se aplicam ao mAP.

Uso:
  python scripts/eval_map_tflite.py --data construction-ppe --split test \
      --model models/yolox_nano_ppe_float32.tflite --model models/yolox_nano_ppe_int8.tflite \
      --order BGR --order RGB --out data/processed/e0_map.json

Requisitos: tensorflow (ou ai-edge-litert), opencv-python-headless, pycocotools, numpy.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # pragma: no cover
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

NAMES = ["helmet", "gloves", "vest", "boots", "goggles", "none", "Person",
         "no_helmet", "no_goggle", "no_gloves", "no_boots"]


def build_gt(data: str, split: str, path: str) -> list[str]:
    imgs = sorted(glob.glob(os.path.join(data, "images", split, "*")))
    images, anns, aid = [], [], 1
    for k, p in enumerate(imgs):
        h, w = cv2.imread(p).shape[:2]
        images.append({"id": k, "file_name": os.path.basename(p), "width": w, "height": h})
        lab = os.path.join(data, "labels", split, os.path.splitext(os.path.basename(p))[0] + ".txt")
        if os.path.exists(lab):
            for line in open(lab):
                if not line.strip():
                    continue
                c, cx, cy, bw, bh = map(float, line.split()[:5])
                anns.append({"id": aid, "image_id": k, "category_id": int(c),
                             "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
                             "area": bw * w * bh * h, "iscrowd": 0})
                aid += 1
    json.dump({"images": images, "annotations": anns,
               "categories": [{"id": i, "name": n} for i, n in enumerate(NAMES)]}, open(path, "w"))
    return imgs


def grids_for(r: int):
    g_all, s_all = [], []
    for s in (8, 16, 32):
        g = r // s
        yv, xv = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        g_all.append(np.stack((xv, yv), -1).reshape(-1, 2))
        s_all.append(np.full((g * g, 1), s))
    return np.concatenate(g_all).astype(np.float32), np.concatenate(s_all).astype(np.float32)


def nms(b, s, thr):
    x1, y1, x2, y2 = b.T
    ar = (x2 - x1) * (y2 - y1)
    o = s.argsort()[::-1]
    keep = []
    while o.size:
        i = o[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[o[1:]]), np.maximum(y1[i], y1[o[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[o[1:]]), np.minimum(y2[i], y2[o[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        o = o[1:][inter / (ar[i] + ar[o[1:]] - inter + 1e-9) <= thr]
    return keep


def evaluate(model: str, order: str, imgs, coco_gt, r=416, conf=0.001, nms_thr=0.65):
    it = Interpreter(model_path=model, num_threads=min(8, os.cpu_count() or 1))
    it.allocate_tensors()
    di, do = it.get_input_details()[0], it.get_output_details()[0]
    grids, strides = grids_for(r)
    dets = []
    for k, p in enumerate(imgs):
        im = cv2.imread(p)  # BGR
        h, w = im.shape[:2]
        rr = min(r / h, r / w)
        pad = np.full((r, r, 3), 114, np.uint8)
        pad[:int(h * rr), :int(w * rr)] = cv2.resize(im, (int(w * rr), int(h * rr)),
                                                     interpolation=cv2.INTER_LINEAR)
        x = (pad if order == "BGR" else pad[:, :, ::-1]).astype(np.float32)[None]
        if di["dtype"] == np.int8:
            sc, zp = di["quantization"]
            x = np.clip(np.round(x / sc + zp), -128, 127).astype(np.int8)
        it.set_tensor(di["index"], x)
        it.invoke()
        y = it.get_tensor(do["index"])[0]
        if do["dtype"] == np.int8:
            sc, zp = do["quantization"]
            y = (y.astype(np.float32) - zp) * sc
        xy = (y[:, :2] + grids) * strides
        wh = np.exp(np.clip(y[:, 2:4], -10, 10)) * strides
        cls = y[:, 5:].argmax(1)
        score = y[:, 4] * y[np.arange(len(y)), 5 + cls]
        m = score >= conf
        if not m.any():
            continue
        b = np.concatenate([xy[m] - wh[m] / 2, xy[m] + wh[m] / 2], 1) / rr
        s, c = score[m], cls[m]
        for i in nms(b + c[:, None] * 4096, s, nms_thr)[:100]:
            x1, y1, x2, y2 = b[i]
            dets.append({"image_id": k, "category_id": int(c[i]),
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                         "score": float(s[i])})
    if not dets:
        return {"mAP50": 0.0, "mAP50_95": 0.0, "n_dets": 0}
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(coco_gt, coco_gt.loadRes(dets), "bbox")
        E.evaluate(); E.accumulate(); E.summarize()
    per_class = {}
    for ci, n in enumerate(NAMES):
        pr = E.eval["precision"][0, :, ci, 0, 2]
        per_class[n] = round(float(np.mean(pr[pr > -1])), 4) if (pr > -1).any() else None
    return {"mAP50": round(float(E.stats[1]), 4), "mAP50_95": round(float(E.stats[0]), 4),
            "n_dets": len(dets), "AP50_per_class": per_class}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--order", action="append", default=None, help="BGR (treino YOLOX) e/ou RGB")
    ap.add_argument("--out", default="data/processed/e0_map.json")
    a = ap.parse_args()
    gt_path = f"gt_{a.split}.json"
    imgs = build_gt(a.data, a.split, gt_path)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(gt_path)
    out = {"split": a.split, "n_images": len(imgs), "n_gt": len(coco_gt.getAnnIds())}
    for model in a.model:
        for order in a.order or ["BGR"]:
            key = f"{os.path.basename(model)}|{order}"
            out[key] = evaluate(model, order, imgs, coco_gt)
            print(key, out[key]["mAP50"], out[key]["mAP50_95"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
