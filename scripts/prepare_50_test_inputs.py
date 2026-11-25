#!/usr/bin/env python3
"""Gera as M=50 entradas .bin (416x416x3 uint8 BGR em letterbox, pad 114) do split de teste (D30/D33).

Seleciona de forma determinística as 50 primeiras imagens ordenadas do split 'test' do
Construction-PPE que possuem ao menos uma instância da classe 'vest' (id 2), conforme
docs/dataset.md e docs/projeto/main.tex §4.7.

Salva:
  - /tmp/inputs_50_bgr_416/<stem>.bin (519.168 bytes cada, 416x416x3 uint8 BGR)
  - data/processed/inputs_50_manifest.json (lista versionada com SHA-256 e metadados de letterbox)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def letterbox_bgr_uint8(img_path: str, r: int = 416) -> tuple[np.ndarray, int, int, float]:
    """Lê imagem, aplica letterbox r x r (pad 114 no canto superior esquerdo) e converte para BGR uint8."""
    with Image.open(img_path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    rr = min(r / h, r / w)
    new_w, new_h = int(w * rr), int(h * rr)
    resized_rgb = np.asarray(
        Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    resized_bgr = resized_rgb[:, :, ::-1]  # RGB -> BGR
    pad = np.full((r, r, 3), 114, dtype=np.uint8)
    pad[:new_h, :new_w, :] = resized_bgr
    return np.ascontiguousarray(pad), w, h, rr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/tmp/construction-ppe")
    ap.add_argument("--out-dir", default="data/inputs_50_bgr_416")
    ap.add_argument("--manifest", default="data/processed/inputs_50_manifest.json")
    ap.add_argument("--r", type=int, default=416)
    ap.add_argument("--m", type=int, default=50)
    a = ap.parse_args()

    imgs = sorted(glob.glob(os.path.join(a.dataset, "images", "test", "*")))
    if not imgs:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {a.dataset}/images/test")

    selected = []
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        lab = os.path.join(a.dataset, "labels", "test", f"{stem}.txt")
        has_vest = False
        if os.path.exists(lab):
            with open(lab, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip() and int(line.split()[0]) == 2:
                        has_vest = True
                        break
        if has_vest:
            selected.append(p)
        if len(selected) == a.m:
            break

    os.makedirs(a.out_dir, exist_ok=True)
    entries = []
    for idx, p in enumerate(selected):
        stem = os.path.splitext(os.path.basename(p))[0]
        arr, orig_w, orig_h, scale_r = letterbox_bgr_uint8(p, r=a.r)
        raw_bytes = arr.tobytes()
        sha = hashlib.sha256(raw_bytes).hexdigest()
        bin_name = f"{idx+1:02d}_{stem}.bin"
        bin_path = os.path.join(a.out_dir, bin_name)
        with open(bin_path, "wb") as f:
            f.write(raw_bytes)
        entries.append({
            "index": idx + 1,
            "image_id": f"{idx+1:02d}_{stem}",
            "orig_file": os.path.basename(p),
            "bin_file": bin_name,
            "orig_width": orig_w,
            "orig_height": orig_h,
            "letterbox_scale": round(scale_r, 6),
            "input_res": a.r,
            "channel_order": "BGR",
            "bytes": len(raw_bytes),
            "sha256": sha,
        })

    manifest_data = {
        "protocol": "v1.5 / main.tex §4.7 (D30)",
        "split": "test",
        "selection_rule": "First 50 sorted test images containing class 'vest' (id=2)",
        "count": len(entries),
        "input_res": a.r,
        "channel_order": "BGR",
        "pad_value": 114,
        "bytes_per_image": a.r * a.r * 3,
        "inputs": entries,
    }
    os.makedirs(os.path.dirname(a.manifest) or ".", exist_ok=True)
    with open(a.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] {len(entries)} arquivos .bin ({a.r}x{a.r}x3 BGR) gerados em '{a.out_dir}'")
    print(f"[OK] Manifesto com hashes SHA-256 salvo em '{a.manifest}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
