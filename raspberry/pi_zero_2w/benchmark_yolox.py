#!/usr/bin/env python3
"""
Benchmark Oficial YOLOX-Nano no Raspberry Pi Zero 2 W — Protocolo Experimental v1.3
Projeto PTT6 - Arquitetura e Organização de Computadores (PPCA/UFRR)
Orientador: Prof. Dr. Herbert Oliveira Rocha
Alunos: Filipe Matias & Guilherme Lucas

Implementa integralmente a especificação do Protocolo v1.3:
  - Equação (1): Decomposição temporal por execução em 3 fases:
      t_total = t_pre + t_inf + t_post
    onde:
      * t_pre  : redimensionamento letterbox + quantização INT8/FP32 + cópia para tensor de entrada
      * t_inf  : execução do grafo no runtime (interpreter.invoke())
      * t_post : leitura do tensor + dequantização + decodificação das 3549 âncoras YOLOX +
                 filtragem por confiança (conf >= 0,25) + NMS (IoU = 0,45)
  - Seção 4.5: Organização em B blocos independentes (padrão: 3 blocos), com 10 inferências
    de aquecimento (warmup) descartadas no início de cada bloco e pausa de resfriamento entre blocos.
  - Monitoramento de temperatura do SoC (/sys/class/thermal/thermal_zone0/temp),
    flags de throttling (vcgencmd get_throttled) e pegada de memória RAM de pico (VmHWM / RSS — E5)
    fora da janela cronometrada.
  - Suporte ao Experimento E4: execução contínua ininterrupta por --duration-s segundos (ex.: 60 s).
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time
import numpy as np

# Tentar importar ai_edge_litert (LiteRT oficial Google p/ Python 3.12+), tflite_runtime ou tensorflow.lite
RUNTIME_NAME = None
try:
    from ai_edge_litert.interpreter import Interpreter
    RUNTIME_NAME = "ai-edge-litert"
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        RUNTIME_NAME = "tflite-runtime"
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
            RUNTIME_NAME = "tensorflow.lite"
        except ImportError:
            print("ERRO: Nenhum motor LiteRT/TFLite ('ai_edge_litert', 'tflite_runtime' ou 'tensorflow') encontrado!")
            sys.exit(1)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import gpiod
    HAS_GPIOD = True
except ImportError:
    HAS_GPIOD = False


class GpioController:
    def __init__(self, pin_num=4):
        self.pin_num = pin_num
        self.line = None
        self.request_v2 = None
        if not HAS_GPIOD:
            return

        chip_path = "/dev/gpiochip4" if os.path.exists("/dev/gpiochip4") else "/dev/gpiochip0"
        try:
            if hasattr(gpiod, "request_lines") and hasattr(gpiod, "LineSettings"):
                from gpiod.line import Direction, Value
                self._val_high = Value.ACTIVE
                self._val_low = Value.INACTIVE
                self.request_v2 = gpiod.request_lines(
                    chip_path,
                    consumer="yolox_benchmark",
                    config={
                        self.pin_num: gpiod.LineSettings(
                            direction=Direction.OUTPUT,
                            output_value=Value.INACTIVE,
                        )
                    },
                )
            else:
                chip = gpiod.Chip(os.path.basename(chip_path))
                self.line = chip.get_line(self.pin_num)
                self.line.request(consumer="yolox_benchmark", type=gpiod.LINE_REQ_DIR_OUT)
                self.line.set_value(0)
        except Exception:
            self.line = None
            self.request_v2 = None

    def set_high(self):
        if self.request_v2 is not None:
            self.request_v2.set_value(self.pin_num, self._val_high)
        elif self.line is not None:
            self.line.set_value(1)

    def set_low(self):
        if self.request_v2 is not None:
            self.request_v2.set_value(self.pin_num, self._val_low)
        elif self.line is not None:
            self.line.set_value(0)

    def cleanup(self):
        try:
            self.set_low()
            if self.request_v2 is not None:
                self.request_v2.release()
            elif self.line is not None:
                self.line.release()
        except Exception:
            pass


def read_cpu_temp_c() -> float:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return -1.0


def read_cpu_freq_mhz() -> float:
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return -1.0


def read_throttled_hex() -> str:
    try:
        out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True, timeout=2.0).strip()
        if "=" in out:
            return out.split("=")[1].strip()
        return out
    except Exception:
        return "0x0"


def read_peak_rss_mb() -> float:
    """Retorna o pico de memória residente (RSS / VmHWM) do processo em MiB (Experimento E5)."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    kb = float(line.split()[1])
                    return kb / 1024.0
    except Exception:
        pass
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return float(ru.ru_maxrss) / 1024.0


def load_source_image(img_path: str):
    """Carrega a imagem original na memória antes do benchmark (sem redimensionar)."""
    if img_path and not os.path.exists(img_path):
        alt = "sbc_rpi/test_image.jpg" if "raspberry/pi_zero_2w" in img_path else "raspberry/pi_zero_2w/test_image.jpg"
        if os.path.exists(alt):
            img_path = alt
    if img_path and os.path.exists(img_path) and HAS_PIL:
        img = Image.open(img_path).convert("RGB")
        return img, np.asarray(img, dtype=np.uint8)
    raw_arr = np.full((640, 640, 3), 114, dtype=np.uint8)
    if HAS_PIL:
        return Image.fromarray(raw_arr), raw_arr
    return None, raw_arr


def preprocess_frame(pil_img, raw_arr, target_h: int, target_w: int, input_dtype, scale: float, zero_point: int):
    """
    Fase 1 (t_pre): Redimensionamento letterbox para (target_h, target_w) + quantização do tensor.
    """
    if pil_img is not None:
        orig_w, orig_h = pil_img.size
        r = min(target_h / orig_h, target_w / orig_w)
        new_w, new_h = int(orig_w * r), int(orig_h * r)
        resized = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w, :] = np.asarray(resized, dtype=np.uint8)
        rgb = padded
    else:
        rgb = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

    if input_dtype == np.int8:
        if scale == 1.0 and zero_point == -128:
            # Otimização vetorizada exata para scale=1.0, zp=-128: int8(uint8 - 128)
            tensor = (rgb.astype(np.int16) - 128).astype(np.int8)
        else:
            tensor = np.round(rgb.astype(np.float32) / scale + zero_point).astype(np.int8)
    else:
        tensor = rgb.astype(np.float32)

    return np.expand_dims(tensor, axis=0)


def build_yolox_grids(img_h: int, img_w: int, strides=(8, 16, 32)):
    """Pré-calcula as grades de âncoras e strides (constantes para uma dada resolução r)."""
    grids = []
    expanded_strides = []
    for stride in strides:
        hsize = img_h // stride
        wsize = img_w // stride
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2).astype(np.float32)
        grids.append(grid)
        expanded_strides.append(np.full((1, grid.shape[1], 1), stride, dtype=np.float32))
    return np.concatenate(grids, axis=1), np.concatenate(expanded_strides, axis=1)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, nms_thr: float = 0.45):
    """Non-Maximum Suppression (NMS) em NumPy conforme Seção 4.4 do Protocolo v1.3."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def postprocess_yolox(
    raw_output: np.ndarray,
    output_dtype,
    out_scale: float,
    out_zp: int,
    grids: np.ndarray,
    expanded_strides: np.ndarray,
    conf_thr: float = 0.25,
    nms_thr: float = 0.45,
):
    """
    Fase 3 (t_post): Dequantização + Decodificação das grades multiescala YOLOX +
    Filtragem por confiança (conf >= 0,25) + NMS class-aware (IoU = 0,45).
    """
    if output_dtype == np.int8:
        preds = (raw_output.astype(np.float32) - float(out_zp)) * float(out_scale)
    else:
        preds = raw_output.astype(np.float32, copy=True)

    # Decodificar centro (cx, cy) e dimensões (w, h) nas 3549 âncoras
    xy = (preds[..., :2] + grids) * expanded_strides
    wh = np.exp(np.clip(preds[..., 2:4], -10.0, 10.0)) * expanded_strides

    pred = preds[0]  # (3549, 16)
    xy0 = xy[0]
    wh0 = wh[0]

    obj_conf = pred[:, 4:5]
    cls_scores = pred[:, 5:]
    scores = obj_conf * cls_scores  # (3549, 11)

    max_cls_inds = np.argmax(scores, axis=1)
    max_scores = scores[np.arange(scores.shape[0]), max_cls_inds]

    valid_mask = max_scores >= conf_thr
    if not np.any(valid_mask):
        return np.empty((0, 6), dtype=np.float32)

    v_xy = xy0[valid_mask]
    v_wh = wh0[valid_mask]
    v_scores = scores[valid_mask]

    boxes = np.empty((v_xy.shape[0], 4), dtype=np.float32)
    boxes[:, 0] = v_xy[:, 0] - v_wh[:, 0] * 0.5
    boxes[:, 1] = v_xy[:, 1] - v_wh[:, 1] * 0.5
    boxes[:, 2] = v_xy[:, 0] + v_wh[:, 0] * 0.5
    boxes[:, 3] = v_xy[:, 1] + v_wh[:, 1] * 0.5

    # NMS por classe (class-aware)
    final_dets = []
    num_classes = v_scores.shape[1]
    for cls_idx in range(num_classes):
        c_scores = v_scores[:, cls_idx]
        c_mask = c_scores >= conf_thr
        if not np.any(c_mask):
            continue
        c_boxes = boxes[c_mask]
        c_sc = c_scores[c_mask]
        keep = nms_numpy(c_boxes, c_sc, nms_thr=nms_thr)
        if keep:
            cls_col = np.full((len(keep), 1), cls_idx, dtype=np.float32)
            det = np.concatenate([c_boxes[keep], c_sc[keep, None], cls_col], axis=1)
            final_dets.append(det)

    if not final_dets:
        return np.empty((0, 6), dtype=np.float32)
    return np.concatenate(final_dets, axis=0)


def run_single_cycle(
    interpreter,
    input_idx: int,
    output_idx: int,
    pil_img,
    raw_arr,
    target_h: int,
    target_w: int,
    input_dtype,
    in_scale: float,
    in_zp: int,
    output_dtype,
    out_scale: float,
    out_zp: int,
    grids: np.ndarray,
    expanded_strides: np.ndarray,
    conf_thr: float,
    nms_thr: float,
    gpio: GpioController,
):
    """Executa e cronometra separadamente t_pre, t_inf e t_post em nanossegundos."""
    gpio.set_high()

    # 1. Fase de Pré-processamento (t_pre)
    t0 = time.perf_counter_ns()
    input_tensor = preprocess_frame(pil_img, raw_arr, target_h, target_w, input_dtype, in_scale, in_zp)
    interpreter.set_tensor(input_idx, input_tensor)
    t1 = time.perf_counter_ns()

    # 2. Fase de Inferência do Modelo (t_inf)
    interpreter.invoke()
    t2 = time.perf_counter_ns()

    # 3. Fase de Pós-processamento: Decode + NMS (t_post)
    raw_out = interpreter.get_tensor(output_idx)
    dets = postprocess_yolox(
        raw_out,
        output_dtype,
        out_scale,
        out_zp,
        grids,
        expanded_strides,
        conf_thr=conf_thr,
        nms_thr=nms_thr,
    )
    t3 = time.perf_counter_ns()

    gpio.set_low()

    t_pre_us = (t1 - t0) // 1000
    t_inf_us = (t2 - t1) // 1000
    t_post_us = (t3 - t2) // 1000
    t_total_us = (t3 - t0) // 1000
    return t_pre_us, t_inf_us, t_post_us, t_total_us, len(dets)


def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLOX-Nano no Raspberry Pi Zero 2 W (Protocolo v1.3)")
    parser.add_argument("--model", default="models/yolox_nano_ppe_int8.tflite", help="Caminho do modelo .tflite")
    parser.add_argument("--image", default="raspberry/pi_zero_2w/test_image.jpg", help="Caminho da imagem de entrada")
    parser.add_argument("--blocks", type=int, default=3, help="Número de blocos independentes (padrão v1.3: 3)")
    parser.add_argument("--runs-per-block", type=int, default=10, help="Inferências medidas por bloco (padrão: 10 => 30 totais)")
    parser.add_argument("--warmup-per-block", type=int, default=10, help="Inferências de warmup descartadas por bloco (padrão v1.3: 10)")
    parser.add_argument("--block-cooldown", type=float, default=15.0, help="Pausa térmica entre blocos em segundos")
    parser.add_argument("--delay", type=float, default=0.2, help="Pausa entre inferências dentro do bloco em segundos")
    parser.add_argument("--duration-s", type=float, default=0.0, help="Se > 0, executa o Experimento E4 contínuo por N segundos sem pausa")
    parser.add_argument("--conf-thr", type=float, default=0.25, help="Limiar de confiança (padrão v1.3: 0.25)")
    parser.add_argument("--nms-thr", type=float, default=0.45, help="Limiar IoU do NMS (padrão v1.3: 0.45)")
    parser.add_argument("--gpio", type=int, default=4, help="Pino BCM GPIO para sincronismo (padrão: 4)")
    parser.add_argument("--threads", type=int, default=4, help="Número de threads TFLite (1, 2 ou 4)")
    parser.add_argument("--out-csv", default="data/raw/rpi_benchmark_raw.csv", help="Caminho do CSV de saída")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERRO: Arquivo do modelo '{args.model}' não encontrado!")
        sys.exit(1)

    model_size_bytes = os.path.getsize(args.model)

    # 1. Carregar interpretador TFLite / LiteRT
    interpreter = Interpreter(model_path=args.model, num_threads=args.threads)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_shape = input_details["shape"]
    target_h, target_w = int(input_shape[1]), int(input_shape[2])
    input_dtype = input_details["dtype"]
    in_scale, in_zp = input_details.get("quantization", (1.0, 0))
    if in_scale == 0.0:
        in_scale = 1.0

    output_dtype = output_details["dtype"]
    out_scale, out_zp = output_details.get("quantization", (1.0, 0))
    if out_scale == 0.0:
        out_scale = 1.0

    grids, expanded_strides = build_yolox_grids(target_h, target_w)
    pil_img, raw_arr = load_source_image(args.image)
    gpio = GpioController(pin_num=args.gpio)

    print("=======================================================================")
    print("  Benchmark YOLOX-Nano — Raspberry Pi Zero 2 W (Protocolo v1.3)        ")
    print("=======================================================================")
    print(f"Runtime       : {RUNTIME_NAME}")
    print(f"Modelo        : {args.model} ({model_size_bytes / (1024*1024):.2f} MiB)")
    print(f"Resolução     : {target_w}x{target_h} | Threads: {args.threads}")
    print(f"Limiares v1.3 : conf >= {args.conf_thr:.2f}, IoU_NMS = {args.nms_thr:.2f}")
    print(f"Temp Inicial  : {read_cpu_temp_c():.1f} °C | Throttled: {read_throttled_hex()}")

    csv_lines = []
    header = (
        "iteration,block,run_in_block,t_pre_us,t_inf_us,t_post_us,t_total_us,"
        "t_pre_ms,t_inf_ms,t_post_ms,t_total_ms,latency_us,latency_ms,"
        "num_detections,cpu_temp_c,cpu_freq_mhz,throttled_hex,rss_mb\n"
    )

    all_t_pre_ms = []
    all_t_inf_ms = []
    all_t_post_ms = []
    all_t_total_ms = []
    all_temps = []

    if args.duration_s > 0.0:
        # Experimento E4: Execução contínua ininterrupta por args.duration_s segundos
        print(f"\n[E4] Iniciando {args.warmup_per_block} inferências de warmup antes dos {args.duration_s:.0f}s contínuos...")
        for _ in range(args.warmup_per_block):
            run_single_cycle(
                interpreter, input_details["index"], output_details["index"],
                pil_img, raw_arr, target_h, target_w, input_dtype, in_scale, in_zp,
                output_dtype, out_scale, out_zp, grids, expanded_strides,
                args.conf_thr, args.nms_thr, gpio
            )

        header = (
            "iteration,elapsed_s,t_pre_us,t_inf_us,t_post_us,t_total_us,"
            "t_pre_ms,t_inf_ms,t_post_ms,t_total_ms,latency_us,latency_ms,"
            "num_detections,cpu_temp_c,cpu_freq_mhz,throttled_hex,rss_mb\n"
        )
        print(f"[E4] Executando inferência contínua por {args.duration_s:.1f} segundos (delay=0)...")
        t_exp_start = time.perf_counter()
        global_iter = 0
        while (time.perf_counter() - t_exp_start) < args.duration_s:
            global_iter += 1
            t_pre_us, t_inf_us, t_post_us, t_total_us, n_dets = run_single_cycle(
                interpreter, input_details["index"], output_details["index"],
                pil_img, raw_arr, target_h, target_w, input_dtype, in_scale, in_zp,
                output_dtype, out_scale, out_zp, grids, expanded_strides,
                args.conf_thr, args.nms_thr, gpio
            )
            elapsed_s = time.perf_counter() - t_exp_start
            temp_c = read_cpu_temp_c()
            freq_mhz = read_cpu_freq_mhz()
            throttled = read_throttled_hex() if (global_iter % 25 == 1) else "0x0"
            rss_mb = read_peak_rss_mb()

            t_pre_ms = t_pre_us / 1000.0
            t_inf_ms = t_inf_us / 1000.0
            t_post_ms = t_post_us / 1000.0
            t_total_ms = t_total_us / 1000.0

            all_t_pre_ms.append(t_pre_ms)
            all_t_inf_ms.append(t_inf_ms)
            all_t_post_ms.append(t_post_ms)
            all_t_total_ms.append(t_total_ms)
            all_temps.append(temp_c)

            csv_lines.append(
                f"{global_iter},{elapsed_s:.3f},{t_pre_us},{t_inf_us},{t_post_us},{t_total_us},"
                f"{t_pre_ms:.3f},{t_inf_ms:.3f},{t_post_ms:.3f},{t_total_ms:.3f},"
                f"{t_inf_us},{t_inf_ms:.3f},{n_dets},{temp_c:.1f},{freq_mhz:.0f},{throttled},{rss_mb:.2f}\n"
            )
        throttled_end = read_throttled_hex()
        print(f"[E4] Concluído: {global_iter} inferências em {args.duration_s:.1f}s | Temp Final: {read_cpu_temp_c():.1f} °C | Throttled: {throttled_end}")

    else:
        # Experimentos E3 / E3b / E3c: Execução em B blocos independentes com warmup por bloco
        global_iter = 0
        for b in range(1, args.blocks + 1):
            print(f"\n--- Bloco {b}/{args.blocks}: Executando {args.warmup_per_block} inferências de warmup (descartadas) ---")
            for _ in range(args.warmup_per_block):
                run_single_cycle(
                    interpreter, input_details["index"], output_details["index"],
                    pil_img, raw_arr, target_h, target_w, input_dtype, in_scale, in_zp,
                    output_dtype, out_scale, out_zp, grids, expanded_strides,
                    args.conf_thr, args.nms_thr, gpio
                )

            throttled_pre = read_throttled_hex()
            print(f"--- Bloco {b}/{args.blocks}: Coletando {args.runs_per_block} execuções válidas (Temp: {read_cpu_temp_c():.1f} °C, Throttled: {throttled_pre}) ---")

            for r in range(1, args.runs_per_block + 1):
                global_iter += 1
                t_pre_us, t_inf_us, t_post_us, t_total_us, n_dets = run_single_cycle(
                    interpreter, input_details["index"], output_details["index"],
                    pil_img, raw_arr, target_h, target_w, input_dtype, in_scale, in_zp,
                    output_dtype, out_scale, out_zp, grids, expanded_strides,
                    args.conf_thr, args.nms_thr, gpio
                )
                # Leituras de sensores SEMPRE fora da janela cronometrada (Seção 4.5)
                temp_c = read_cpu_temp_c()
                freq_mhz = read_cpu_freq_mhz()
                throttled = read_throttled_hex() if r == args.runs_per_block else throttled_pre
                rss_mb = read_peak_rss_mb()

                t_pre_ms = t_pre_us / 1000.0
                t_inf_ms = t_inf_us / 1000.0
                t_post_ms = t_post_us / 1000.0
                t_total_ms = t_total_us / 1000.0

                all_t_pre_ms.append(t_pre_ms)
                all_t_inf_ms.append(t_inf_ms)
                all_t_post_ms.append(t_post_ms)
                all_t_total_ms.append(t_total_ms)
                all_temps.append(temp_c)

                csv_lines.append(
                    f"{global_iter},{b},{r},{t_pre_us},{t_inf_us},{t_post_us},{t_total_us},"
                    f"{t_pre_ms:.3f},{t_inf_ms:.3f},{t_post_ms:.3f},{t_total_ms:.3f},"
                    f"{t_inf_us},{t_inf_ms:.3f},{n_dets},{temp_c:.1f},{freq_mhz:.0f},{throttled},{rss_mb:.2f}\n"
                )
                if args.delay > 0:
                    time.sleep(args.delay)

            if b < args.blocks and args.block_cooldown > 0:
                print(f"  [Pausa térmica entre blocos] Aguardando {args.block_cooldown:.1f}s antes do Bloco {b + 1}...")
                time.sleep(args.block_cooldown)

    gpio.cleanup()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8") as f:
        f.write(header)
        f.writelines(csv_lines)

    print(f"\n[OK] {len(csv_lines)} execuções salvas em '{args.out_csv}'")
    print("================ RESUMO ESTATÍSTICO (PROTOCOLO v1.3) ================")
    print(f"Modelo / Threads   : {os.path.basename(args.model)} ({args.threads} threads)")
    print(f"Amostras Válidas   : {len(all_t_inf_ms)}")
    print(f"t_pre   Mediana/Méd: {np.median(all_t_pre_ms):.2f} ms / {np.mean(all_t_pre_ms):.2f} ± {np.std(all_t_pre_ms, ddof=1):.2f} ms")
    print(f"t_inf   Mediana/Méd: {np.median(all_t_inf_ms):.2f} ms / {np.mean(all_t_inf_ms):.2f} ± {np.std(all_t_inf_ms, ddof=1):.2f} ms")
    print(f"t_post  Mediana/Méd: {np.median(all_t_post_ms):.2f} ms / {np.mean(all_t_post_ms):.2f} ± {np.std(all_t_post_ms, ddof=1):.2f} ms")
    print(f"t_total Mediana/Méd: {np.median(all_t_total_ms):.2f} ms / {np.mean(all_t_total_ms):.2f} ± {np.std(all_t_total_ms, ddof=1):.2f} ms")
    print(f"Throughput (t_inf) : {1000.0 / float(np.mean(all_t_inf_ms)):.2f} FPS | (t_total): {1000.0 / float(np.mean(all_t_total_ms)):.2f} FPS")
    print(f"Temp Média / Máx   : {np.mean(all_temps):.1f} °C / {np.max(all_temps):.1f} °C | Pico RSS (E5): {read_peak_rss_mb():.2f} MiB")
    print("=====================================================================\n")


if __name__ == "__main__":
    main()

