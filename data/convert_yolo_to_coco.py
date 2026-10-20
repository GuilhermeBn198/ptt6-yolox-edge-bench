#!/usr/bin/env python3
"""
Script para conversão do dataset Construction-PPE (formato YOLO)
para o formato padrão COCO (JSON) utilizado nativamente pelo YOLOX.

Uso:
    python data/convert_yolo_to_coco.py --dataset_dir /mnt/ssd_dados/construction-ppe-dataset
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image

PPE_CLASSES = [
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles",
    "none",
    "Person",
    "no_helmet",
    "no_goggle",
    "no_gloves",
    "no_boots",
]


def convert_split(dataset_dir: Path, split: str, output_dir: Path):
  images_dir = dataset_dir / "images" / split
  labels_dir = dataset_dir / "labels" / split

  if not images_dir.exists():
    print(f"Diretório de imagens não encontrado: {images_dir}")
    return

  categories = [
      {"id": i, "name": name, "supercategory": "none"}
      for i, name in enumerate(PPE_CLASSES)
  ]

  coco_data = {
      "info": {
          "description": f"Construction PPE Dataset - {split}",
          "year": 2026,
      },
      "licenses": [],
      "categories": categories,
      "images": [],
      "annotations": [],
  }

  valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
  image_files = sorted(
      [f for f in images_dir.iterdir() if f.suffix.lower() in valid_extensions]
  )

  ann_id = 1
  for img_id, img_path in enumerate(image_files, start=1):
    with Image.open(img_path) as img:
      width, height = img.size

    coco_data["images"].append({
        "id": img_id,
        "file_name": img_path.name,
        "width": width,
        "height": height,
    })

    label_path = labels_dir / f"{img_path.stem}.txt"
    if not label_path.exists():
      continue

    with open(label_path, "r", encoding="utf-8") as f:
      for line in f:
        parts = line.strip().split()
        if len(parts) != 5:
          continue

        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])

        # Converter de YOLO normalizado para coordenadas COCO [x_min, y_min, w, h]
        abs_w = w * width
        abs_h = h * height
        x_min = (cx - w / 2) * width
        y_min = (cy - h / 2) * height

        coco_data["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": cls_id,
            "bbox": [round(x_min, 2), round(y_min, 2), round(abs_w, 2), round(abs_h, 2)],
            "area": round(abs_w * abs_h, 2),
            "iscrowd": 0,
        })
        ann_id += 1

  output_dir.mkdir(parents=True, exist_ok=True)
  output_file = output_dir / f"instances_{split}.json"
  with open(output_file, "w", encoding="utf-8") as f:
    json.dump(coco_data, f)

  print(
      f"Split '{split}': {len(coco_data['images'])} imagens e"
      f" {len(coco_data['annotations'])} anotações salvas em {output_file}"
  )


def main():
  parser = argparse.ArgumentParser(
      description="Converte anotações YOLO para COCO JSON"
  )
  parser.add_argument(
      "--dataset_dir",
      type=str,
      default="construction-ppe-dataset",
      help="Caminho raiz do dataset",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default=None,
      help="Diretório de saída para os arquivos JSON (padrão: <dataset_dir>/annotations)",
  )
  args = parser.parse_args()

  dataset_dir = Path(args.dataset_dir).resolve()
  output_dir = (
      Path(args.output_dir).resolve()
      if args.output_dir
      else dataset_dir / "annotations"
  )

  for split in ["train", "val", "test"]:
    convert_split(dataset_dir, split, output_dir)


if __name__ == "__main__":
  main()

