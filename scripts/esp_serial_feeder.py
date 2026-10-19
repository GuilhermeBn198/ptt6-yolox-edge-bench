#!/usr/bin/env python3
"""Alimenta o ESP32 com imagens pela serial e coleta as métricas (protocolo v1.4, D33).

Roda no PC. Envia cada entrada .bin (r*r*3 uint8, letterbox — D30) para o firmware
firmware/esp32/bench_serial/bench_serial.ino, espera o ACK e recebe tempos por fase e detecções.
A transferência acontece FORA da janela cronometrada; a janela é medida no ESP32 (esp_timer).

Saída: data/raw/MCU_<exp>_b<bloco>_<ts>.csv (mesmo esquema do SBC, docs/experimentacao.md §5.1),
       ..._meta.json e ..._dets.csv (detecções, para o E1).

Uso:
  pip install pyserial
  python scripts/esp_serial_feeder.py --port COM5 --inputs inputs/ --experiment E2 --warmup 5 --n 30
  python scripts/esp_serial_feeder.py --simulate --inputs inputs/ --n 5     # testa sem hardware

Quadro PC -> ESP32:  b"IMG" | id:u16 LE | len:u32 LE | payload | crc32:u32 LE  (zlib.crc32)
Linhas ESP32 -> PC:  BOOT,sha256=<hex>,r=<int>,classes=<int>,arena=<bytes>,core=<versao>
                     ACK,<id> | NAK,<id>,<motivo>
                     RES,<id>,t_pre_us,t_inf_us,t_post_us,t_total_us,n_det
                     DET,<id>,x1,y1,x2,y2,score,cls
                     END,<id>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import sys
import time
import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ["run_uuid", "experiment", "platform", "block_id", "seq_idx", "is_warmup", "image_id",
          "model_sha256", "num_classes", "input_res", "threads", "t_pre_us", "t_inf_us",
          "t_post_us", "t_total_us", "n_det", "cpu_temp_c", "arm_freq_mhz", "throttled"]


def frame(img_id: int, payload: bytes) -> bytes:
    return (b"IMG" + struct.pack("<HI", img_id, len(payload)) + payload
            + struct.pack("<I", zlib.crc32(payload) & 0xFFFFFFFF))


class FakeESP32:
    """Simula o firmware para validar o fluxo do PC sem hardware."""

    def __init__(self, r: int, num_classes: int):
        self.r, self.c, self.out, self.k = r, num_classes, [], 0
        self.out.append(f"BOOT,sha256=simulated,r={r},classes={num_classes},arena=0,core=sim\n")

    def write(self, data: bytes):
        assert data[:3] == b"IMG"
        img_id, n = struct.unpack("<HI", data[3:9])
        payload, crc = data[9:9 + n], struct.unpack("<I", data[9 + n:13 + n])[0]
        if zlib.crc32(payload) & 0xFFFFFFFF != crc or n != self.r * self.r * 3:
            self.out.append(f"NAK,{img_id},crc\n")
            return
        self.k += 1
        t_pre, t_inf, t_post = 900, 4_000_000 + 1000 * (self.k % 7), 1500
        self.out += [f"ACK,{img_id}\n",
                     f"RES,{img_id},{t_pre},{t_inf},{t_post},{t_pre + t_inf + t_post},1\n",
                     f"DET,{img_id},10.0,12.0,80.0,140.0,0.81,0\n", f"END,{img_id}\n"]

    def readline(self) -> bytes:
        return self.out.pop(0).encode() if self.out else b""

    def reset_input_buffer(self):
        pass

    def close(self):
        pass


def read_line(ser, timeout_s: float) -> str:
    t_end = time.monotonic() + timeout_s
    while time.monotonic() < t_end:
        line = ser.readline().decode(errors="replace").strip()
        if line:
            return line
    raise TimeoutError(f"sem resposta do ESP32 em {timeout_s:.0f} s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--experiment", default="E2")
    ap.add_argument("--block", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--num-classes", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600.0, help="s por inferência (MCU é lento)")
    ap.add_argument("--ambient-c", type=float, default=None)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--simulate", action="store_true")
    a = ap.parse_args()

    files = sorted(a.inputs.glob("*.bin"))
    if not files:
        sys.exit(f"nenhum .bin em {a.inputs}")
    imgs = [(f.stem, f.read_bytes()) for f in files]
    r = int(round((len(imgs[0][1]) / 3) ** 0.5))
    if any(len(b) != r * r * 3 for _, b in imgs):
        sys.exit("todas as entradas devem ter r*r*3 bytes")

    if a.simulate:
        ser = FakeESP32(r, a.num_classes)
    else:
        import serial  # pyserial
        if not a.port:
            sys.exit("--port é obrigatório (ou use --simulate)")
        ser = serial.Serial(a.port, a.baud, timeout=1)
        time.sleep(2.0)  # reset por DTR ao abrir a porta

    boot = {}
    try:
        line = read_line(ser, 15)
        while not line.startswith("BOOT"):
            line = read_line(ser, 15)
        boot = dict(kv.split("=", 1) for kv in line.split(",")[1:])
    except TimeoutError:
        print("aviso: linha BOOT não recebida (reinicie o ESP32 para registrar o hash do modelo)")
    if boot.get("r") and int(boot["r"]) != r:
        sys.exit(f"firmware espera r={boot['r']}, entradas têm r={r}")

    run_id, started = str(uuid.uuid4()), datetime.now(timezone.utc)
    model_sha = boot.get("sha256", "")
    rows, dets = [], []
    total = a.warmup + a.n
    for k in range(total):
        img_name, payload = imgs[k % len(imgs)]
        ser.reset_input_buffer()
        ser.write(frame(k & 0xFFFF, payload))
        line = read_line(ser, 30)
        if not line.startswith("ACK"):
            sys.exit(f"[{k}] esperado ACK, recebido: {line}")
        res = read_line(ser, a.timeout).split(",")
        if res[0] != "RES":
            sys.exit(f"[{k}] esperado RES, recebido: {','.join(res)}")
        t_pre, t_inf, t_post, t_tot, n_det = map(int, res[2:7])
        while True:
            line = read_line(ser, 30)
            if line.startswith("END"):
                break
            if line.startswith("DET"):
                dets.append([run_id, k, img_name] + line.split(",")[2:])
        rows.append([run_id, a.experiment, "MCU", a.block, k, int(k < a.warmup), img_name, model_sha,
                     a.num_classes, r, 1, t_pre, t_inf, t_post, t_tot, n_det, "", 240, ""])
        print(f"[{k + 1}/{total}] {img_name}: t_total {t_tot / 1000:.1f} ms, n_det {n_det}"
              + (" (warm-up)" if k < a.warmup else ""))
    ser.close()

    a.out.mkdir(parents=True, exist_ok=True)
    stem = f"MCU_{a.experiment}_b{a.block}_{started.strftime('%Y%m%dT%H%M%SZ')}"
    with open(a.out / f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(rows)
    with open(a.out / f"{stem}_dets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_uuid", "seq_idx", "image_id", "x1", "y1", "x2", "y2", "score", "cls"])
        w.writerows(dets)
    meta = {"run_uuid": run_id, "experiment": a.experiment, "platform": "MCU", "block_id": a.block,
            "started_utc": started.isoformat(), "ended_utc": datetime.now(timezone.utc).isoformat(),
            "ingestion": "serial (D33)", "baud": a.baud, "boot": boot, "model_sha256": model_sha,
            "input_res": r, "num_classes": a.num_classes, "threads": 1, "warmup": a.warmup, "n": a.n,
            "inputs": {s: hashlib.sha256(b).hexdigest() for s, b in imgs},
            "ambient_c": a.ambient_c, "simulated": a.simulate}
    (a.out / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    valid = sorted(row[14] for row in rows if not row[5])
    if valid:
        print(f"{stem}: {len(valid)} válidas | t_total mediana {valid[len(valid) // 2] / 1000:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
