#!/usr/bin/env python3
"""Coletor de métricas do PTT6 no Raspberry Pi Zero 2 W (protocolo v1.3, docs/experimentacao.md).

Mede, por inferência, t_pre / t_inf / t_post com time.perf_counter_ns() (D28), sem I/O dentro da
janela: as linhas ficam em memória e o CSV + meta.json são gravados ao fim do bloco.

Entradas (D30): arquivos .bin com r*r*3 bytes (RGB uint8, já em letterbox), idênticos aos do ESP32.

Uso:
  python3 bench_collect.py --model yolox_nano_int8.tflite --inputs inputs/ --experiment E3 \
      --block 1 --warmup 10 --n 300 --threads 4 --num-classes 1 --out ../../data/raw
  python3 bench_collect.py --dry-run --inputs inputs/      # testa o fluxo sem modelo

O pós-processamento assume a saída YOLOX exportada com decodificação ([1, A, 5 + C]: cx, cy, w, h,
obj, classes já em sigmoide). Se o seu export não decodifica, use --raw-head (decodifica grades
nos strides 8/16/32). Ajuste `postprocess` se o seu modelo diferir — mantendo a MESMA lógica no ESP32.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CONF_THR = 0.25
NMS_IOU = 0.45
MAX_DET = 100
STRIDES = (8, 16, 32)

FIELDS = ["run_uuid", "experiment", "platform", "block_id", "seq_idx", "is_warmup", "image_id",
          "model_sha256", "num_classes", "input_res", "threads", "t_pre_us", "t_inf_us",
          "t_post_us", "t_total_us", "n_det", "cpu_temp_c", "arm_freq_mhz", "throttled"]


# ---------------------------------------------------------------- telemetria (fora da janela)
def read_temp_c() -> float | None:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
    except OSError:
        return None


def read_freq_mhz() -> int | None:
    try:
        return int(Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read_text()) // 1000
    except OSError:
        return None


def vcgencmd(*args: str) -> str:
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True, text=True,
                              timeout=2).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def throttled() -> str:
    out = vcgencmd("get_throttled")  # "throttled=0x0"
    return out.split("=")[-1] if out else ""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- interpretador
class DryRunInterpreter:
    """Simula um modelo para validar o fluxo de coleta sem TFLite."""

    def __init__(self, r: int, num_classes: int):
        self.r, self.c = r, num_classes
        self.anchors = sum((r // s) ** 2 for s in STRIDES)

    def input_details(self):
        return {"shape": (1, self.r, self.r, 3), "dtype": np.int8, "quant": (1.0, -128)}

    def set_input(self, x):
        self._x = x

    def invoke(self):
        time.sleep(0.002)

    def get_output(self):
        rng = np.random.default_rng(0)
        out = rng.random((1, self.anchors, 5 + self.c), dtype=np.float32) * 0.2
        out[0, :, :4] *= self.r * 5
        return out


class TFLiteInterpreter:
    def __init__(self, model: Path, threads: int):
        try:
            from ai_edge_litert.interpreter import Interpreter  # LiteRT (nome novo)
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                from tensorflow.lite import Interpreter  # type: ignore
        self.it = Interpreter(model_path=str(model), num_threads=threads)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]

    def input_details(self):
        return {"shape": tuple(self.inp["shape"]), "dtype": self.inp["dtype"],
                "quant": self.inp["quantization"]}

    def set_input(self, x):
        self.it.set_tensor(self.inp["index"], x)

    def invoke(self):
        self.it.invoke()

    def get_output(self):
        y = self.it.get_tensor(self.out["index"])
        scale, zp = self.out["quantization"]
        if y.dtype != np.float32 and scale:
            y = (y.astype(np.float32) - zp) * scale
        return y


# ---------------------------------------------------------------- pipeline (MESMA lógica no ESP32)
def preprocess(raw: np.ndarray, det: dict) -> np.ndarray:
    """raw: uint8 (r, r, 3) em letterbox. YOLOX não normaliza (entrada 0..255)."""
    x = raw.astype(np.float32)[None]
    dtype = det["dtype"]
    if dtype == np.float32:
        return x
    scale, zp = det["quant"]
    q = np.round(x / scale + zp)
    info = np.iinfo(dtype)
    return np.clip(q, info.min, info.max).astype(dtype)


def decode_raw_head(y: np.ndarray, r: int) -> np.ndarray:
    grids, strides = [], []
    for s in STRIDES:
        g = r // s
        yv, xv = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        grids.append(np.stack((xv, yv), -1).reshape(-1, 2))
        strides.append(np.full((g * g, 1), s))
    grids, strides = np.concatenate(grids), np.concatenate(strides)
    y = y.copy()
    y[..., :2] = (y[..., :2] + grids) * strides
    y[..., 2:4] = np.exp(y[..., 2:4]) * strides
    y[..., 4:] = 1.0 / (1.0 + np.exp(-y[..., 4:]))
    return y


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size and len(keep) < MAX_DET:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def postprocess(y: np.ndarray, r: int, raw_head: bool) -> np.ndarray:
    y = y[0]
    if raw_head:
        y = decode_raw_head(y, r)
    cls = y[:, 5:].argmax(1)
    score = y[:, 4] * y[np.arange(len(y)), 5 + cls]
    m = score >= CONF_THR
    if not m.any():
        return np.zeros((0, 6), np.float32)
    b, s, c = y[m, :4], score[m], cls[m]
    xyxy = np.concatenate([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], 1)
    # NMS por classe (offset de coordenadas)
    keep = nms(xyxy + c[:, None] * (r + 1), s, NMS_IOU)
    return np.concatenate([xyxy[keep], s[keep, None], c[keep, None]], 1)


# ---------------------------------------------------------------- coleta
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--inputs", type=Path, required=True, help="pasta com .bin (r*r*3 uint8)")
    ap.add_argument("--experiment", default="E2")
    ap.add_argument("--block", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=10, help="inferências de warm-up (is_warmup=1)")
    ap.add_argument("--n", type=int, default=100, help="inferências cronometradas após o warm-up")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--num-classes", type=int, default=1)
    ap.add_argument("--raw-head", action="store_true")
    ap.add_argument("--sustained-s", type=float, default=0.0, help="E4: segundos de laço fechado")
    ap.add_argument("--temp-every", type=int, default=1, help="ler temperatura a cada k inferências")
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--ambient-c", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    files = sorted(a.inputs.glob("*.bin"))
    if not files:
        sys.exit(f"nenhum .bin em {a.inputs}")
    first = np.fromfile(files[0], np.uint8)
    r = int(round((first.size / 3) ** 0.5))
    if r * r * 3 != first.size:
        sys.exit(f"{files[0].name}: tamanho {first.size} não é r*r*3")
    raws = [(f.stem, np.fromfile(f, np.uint8).reshape(r, r, 3)) for f in files]

    if a.dry_run:
        interp, model_hash = DryRunInterpreter(r, a.num_classes), "dry-run"
    else:
        if not a.model:
            sys.exit("--model é obrigatório (ou use --dry-run)")
        interp, model_hash = TFLiteInterpreter(a.model, a.threads), sha256(a.model)
    det = interp.input_details()
    if det["shape"][1] != r:
        sys.exit(f"entrada do modelo {det['shape']} difere das .bin (r={r})")

    run_id = str(uuid.uuid4())
    started = datetime.now(timezone.utc)
    meta = {
        "run_uuid": run_id, "experiment": a.experiment, "platform": "SBC", "block_id": a.block,
        "started_utc": started.isoformat(), "hostname_masked": True,
        "uname": " ".join(platform.uname()), "python": sys.version.split()[0],
        "numpy": np.__version__, "model": str(a.model) if a.model else None,
        "model_sha256": model_hash, "input_res": r, "num_classes": a.num_classes,
        "threads": a.threads, "conf_thr": CONF_THR, "nms_iou": NMS_IOU, "max_det": MAX_DET,
        "inputs": {s: hashlib.sha256(x.tobytes()).hexdigest() for s, x in raws},
        "governor": Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
        if Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").exists() else None,
        "throttled_start": throttled(), "temp_start_c": read_temp_c(),
        "arm_clock_start": vcgencmd("measure_clock", "arm"), "ambient_c": a.ambient_c,
        "warmup": a.warmup, "n": a.n,
    }
    for mod in ("ai_edge_litert", "tflite_runtime"):
        try:
            meta[f"{mod}_version"] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass

    freq, thr = read_freq_mhz(), meta["throttled_start"]
    rows, pc = [], time.perf_counter_ns
    total = a.warmup + a.n
    for k in range(total):
        img_id, raw = raws[k % len(raws)]
        t0 = pc()
        x = preprocess(raw, det)
        interp.set_input(x)
        t1 = pc()
        interp.invoke()
        t2 = pc()
        dets = postprocess(interp.get_output(), r, a.raw_head)
        t3 = pc()
        # ---- fora da janela
        temp = read_temp_c() if k % a.temp_every == 0 else None
        rows.append([run_id, a.experiment, "SBC", a.block, k, int(k < a.warmup), img_id, model_hash,
                     a.num_classes, r, a.threads, (t1 - t0) // 1000, (t2 - t1) // 1000,
                     (t3 - t2) // 1000, (t3 - t0) // 1000, len(dets), temp, freq, thr])
        if k % 50 == 0:  # amostragem esparsa: subprocess fora da janela
            freq, thr = read_freq_mhz(), throttled()

    sustained = None
    if a.sustained_s > 0:  # E4: laço fechado só com contador
        temps, n_done, t_end = [], 0, time.perf_counter() + a.sustained_s
        next_tel = time.perf_counter()
        while time.perf_counter() < t_end:
            _, raw = raws[n_done % len(raws)]
            interp.set_input(preprocess(raw, det))
            interp.invoke()
            postprocess(interp.get_output(), r, a.raw_head)
            n_done += 1
            if time.perf_counter() >= next_tel:
                temps.append((round(time.perf_counter() - (t_end - a.sustained_s), 1),
                              read_temp_c(), throttled()))
                next_tel += 5.0
        sustained = {"seconds": a.sustained_s, "inferences": n_done,
                     "throughput_inf_s": n_done / a.sustained_s, "telemetry_5s": temps}

    meta.update({"ended_utc": datetime.now(timezone.utc).isoformat(),
                 "throttled_end": throttled(), "temp_end_c": read_temp_c(),
                 "sustained": sustained})
    a.out.mkdir(parents=True, exist_ok=True)
    stem = f"SBC_{a.experiment}_b{a.block}_{started.strftime('%Y%m%dT%H%M%SZ')}"
    with open(a.out / f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(rows)
    (a.out / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    valid = np.array([row[14] for row in rows if not row[5]], dtype=float) / 1000.0
    print(f"{stem}: {len(valid)} válidas | t_total mediana {np.median(valid):.2f} ms | "
          f"IQR {np.subtract(*np.percentile(valid, [75, 25])):.2f} ms | "
          f"CV {valid.std(ddof=1) / valid.mean() * 100:.2f}%"
          + (f" | sustentado {sustained['throughput_inf_s']:.2f} inf/s" if sustained else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
