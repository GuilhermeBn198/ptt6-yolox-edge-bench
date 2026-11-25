#!/usr/bin/env python3
"""
Gera as predições de referência no PC (x86_64) usando o runtime TFLite / LiteRT do venv
(/mnt/ssd_dados/venvs/yolox_edge) sobre as mesmas M=50 entradas .bin BGR (416x416x3 uint8),
com exatamente a mesma função postprocess_yolox do SBC e da MCU.

Salva:
  - data/raw/pc_e1_50_predictions.json (Referência PC LiteRT/TFLite)
  - data/raw/pc_tflm_e1_50_predictions.json (Referência PC TFLite Micro C++, preservada para comparar TFLM PC vs TFLM MCU)
"""
import hashlib
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raspberry.pi_zero_2w.benchmark_yolox import (
    Interpreter,
    RUNTIME_NAME,
    build_yolox_grids,
    load_bgr_bin_inputs,
    postprocess_yolox,
    preprocess_in_memory,
)


def main() -> int:
    # Preservar a saída anterior do TFLite Micro C++ no PC se ainda não copiada
    if os.path.exists("data/raw/pc_e1_50_predictions.json") and not os.path.exists("data/raw/pc_tflm_e1_50_predictions.json"):
        with open("data/raw/pc_e1_50_predictions.json", "r", encoding="utf-8") as f:
            prev = json.load(f)
        if "TFLite Micro" in prev.get("platform", ""):
            shutil.copyfile("data/raw/pc_e1_50_predictions.json", "data/raw/pc_tflm_e1_50_predictions.json")

    model_path = "models/yolox_nano_ppe_int8.tflite"
    with open(model_path, "rb") as f:
        int8_sha = hashlib.sha256(f.read()).hexdigest()

    interp = Interpreter(model_path=model_path, num_threads=4)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    target_h, target_w = int(in_det["shape"][1]), int(in_det["shape"][2])
    in_scale, in_zp = in_det.get("quantization", (1.0, -128))
    if in_scale == 0.0:
        in_scale = 1.0
    out_scale, out_zp = out_det.get("quantization", (1.0, 0))
    if out_scale == 0.0:
        out_scale = 1.0

    grids, expanded_strides = build_yolox_grids(target_h, target_w)
    bgr_inputs = load_bgr_bin_inputs("data/inputs_50_bgr_416", "raspberry/pi_zero_2w/test_image.jpg", target_h, target_w)

    entries = []
    total_dets = 0
    for img_id, bgr_arr in bgr_inputs:
        t0 = time.perf_counter_ns()
        t_in = preprocess_in_memory(bgr_arr, in_det["dtype"], in_scale, in_zp)
        interp.set_tensor(in_det["index"], t_in)
        t1 = time.perf_counter_ns()
        interp.invoke()
        t2 = time.perf_counter_ns()
        raw_y = interp.get_tensor(out_det["index"])
        dets = postprocess_yolox(raw_y, out_det["dtype"], out_scale, out_zp, grids, expanded_strides)
        t3 = time.perf_counter_ns()

        d_list = []
        for d in dets:
            d_list.append({
                "bbox_xyxy": [round(float(d[0]), 2), round(float(d[1]), 2), round(float(d[2]), 2), round(float(d[3]), 2)],
                "score": round(float(d[4]), 4),
                "cls": int(d[5]),
            })
        total_dets += len(d_list)
        entries.append({
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

    out_path = "data/raw/pc_e1_50_predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "platform": f"PC x86_64 ({RUNTIME_NAME})",
            "model": model_path,
            "model_sha256": int8_sha,
            "channel_order": "BGR",
            "input_res": target_w,
            "out_scale": round(float(out_scale), 6),
            "out_zp": int(out_zp),
            "conf_thr": 0.25,
            "nms_thr": 0.45,
            "num_images": len(entries),
            "total_detections": total_dets,
            "images": entries,
        }, f, indent=2, ensure_ascii=False)

    print(f"[OK] E1-PC ({RUNTIME_NAME}) salvo em '{out_path}' ({total_dets} detecções em {len(entries)} imagens).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
