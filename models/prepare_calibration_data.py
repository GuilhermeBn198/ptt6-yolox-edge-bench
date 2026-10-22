#!/usr/bin/env python3
"""
Script para gerar conjunto de calibração representativo (750 imagens de EPIs, BGR — protocolo v1.5)
para a quantização INT8 do modelo YOLOX-Nano.

Gera o arquivo 'models/calibration_data.npy' no formato NCHW [750, 3, 416, 416] (float32).
"""

import os
from pathlib import Path
import cv2
import numpy as np


def preprocess_image(img_path: str, input_size=(416, 416)) -> np.ndarray:
  """Pré-processamento padrão do YOLOX com letterbox / padding mantendo proporção"""
  img = cv2.imread(img_path)
  if img is None:
    return None

  target_h, target_w = input_size
  h, w = img.shape[:2]
  r = min(target_h / h, target_w / w)
  new_h, new_w = int(h * r), int(w * r)

  resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

  # Canvas preenchido com valor 114 (cinza padrão YOLO)
  padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
  padded[:new_h, :new_w] = resized

  # v1.5 (D38): manter BGR — o YOLOX treina com cv2 (BGR); calibrar em RGB descasa as ativações

  # Transpor de HWC para CHW e converter para float32 [0..255] (YOLOX recebe float32 sem dividir por 255)
  chw = padded.transpose(2, 0, 1).astype(np.float32)
  return chw


def main():
  dataset_dir = Path("construction-ppe-dataset/images/train")
  if not dataset_dir.exists():
    dataset_dir = Path("/mnt/ssd_dados/construction-ppe-dataset/images/train")

  image_files = sorted(
      [
          f
          for f in dataset_dir.iterdir()
          if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
      ]
  )

  print(f"Total de imagens encontradas: {len(image_files)}")

  # v1.5 (D17/D38): 750 imagens do train distribuídas pelo dataset (estratificação simples por ordem)
  n_calib = 750
  step = max(1, len(image_files) // n_calib)
  selected_files = image_files[::step][:n_calib]

  calib_data = []
  for f in selected_files:
    data = preprocess_image(str(f))
    if data is not None:
      calib_data.append(data)

  calib_array = np.stack(calib_data, axis=0)  # [N, 3, 416, 416]
  print(f"Dataset de calibração gerado com forma: {calib_array.shape}")

  out_dir = Path("models")
  out_dir.mkdir(parents=True, exist_ok=True)
  np.save(out_dir / "calibration_data.npy", calib_array)
  print(f"Salvo em: {out_dir / 'calibration_data.npy'}")


if __name__ == "__main__":
  main()

