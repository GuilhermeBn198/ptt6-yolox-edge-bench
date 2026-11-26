# -*- coding: utf-8 -*-
"""Análise final do artigo (docs/projeto/main.tex) a partir dos dados brutos.

Gera, de forma determinística (semente fixa), todos os números das seções de Resultados:
  data/processed/artigo/resultados.json   -> valores citados no texto e nas tabelas
  data/processed/artigo/descritiva_E3.csv -> estatística descritiva por configuração e fase
  data/processed/artigo/energia_E3.csv    -> energia por inferência (nominal e extremos)
  data/processed/artigo/deteccao_50.csv   -> qualidade de detecção no dispositivo (50 imagens)

Uso:
  python scripts/analise_artigo.py --dataset-zip <construction-ppe.zip>
O zip oficial do Construction-PPE (SHA-256 em docs/dataset.md) só é necessário para a avaliação de
detecção nas 50 imagens; sem ele essa parte é omitida.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tomllib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = PROC / "artigo"
RNG_SEED = 20261125
N_BOOT = 10_000

CONFIGS = {  # id -> (arquivo, plataforma, precisão, threads)
    "SBC_int8_4t": ("SBC_E3_int8_4t.csv", "SBC", "int8", 4),
    "SBC_int8_2t": ("SBC_E3b_int8_2t.csv", "SBC", "int8", 2),
    "SBC_int8_1t": ("SBC_E3b_int8_1t.csv", "SBC", "int8", 1),
    "SBC_fp32_4t": ("SBC_E3_fp32_4t.csv", "SBC", "fp32", 4),
    "MCU_int8": ("MCU_E3_int8.csv", "MCU", "int8", 1),
}
PHASES = ["t_pre", "t_inf", "t_post", "t_total"]
CLASSES = ["helmet", "gloves", "vest", "boots", "goggles", "none", "Person",
           "no_helmet", "no_goggle", "no_gloves", "no_boots"]


# ----------------------------------------------------------------------------- utilidades
def cliffs_delta(x, y):
    x = np.asarray(x)[:, None]
    y = np.asarray(y)[None, :]
    return float(((x > y).sum() - (x < y).sum()) / (x.size * y.size))


def boot_ci_ratio_medians(x, y, rng):
    """IC 95% BCa da razão median(x)/median(y), amostras independentes."""
    res = stats.bootstrap((np.asarray(x), np.asarray(y)),
                          lambda a, b, axis=-1: np.median(a, axis=axis) / np.median(b, axis=axis),
                          paired=False, vectorized=True, n_resamples=N_BOOT,
                          method="BCa", random_state=rng)
    return [float(res.confidence_interval.low), float(res.confidence_interval.high)]


def boot_ci_median(x, rng):
    res = stats.bootstrap((np.asarray(x),), lambda a, axis=-1: np.median(a, axis=axis),
                          vectorized=True, n_resamples=N_BOOT, method="BCa", random_state=rng)
    return [float(res.confidence_interval.low), float(res.confidence_interval.high)]


def acf1(x):
    x = np.asarray(x, float) - np.mean(x)
    return float(np.sum(x[1:] * x[:-1]) / np.sum(x * x)) if np.sum(x * x) > 0 else float("nan")


def r(v, n=3):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), n)


# ----------------------------------------------------------------------------- modelo de potência
def load_power():
    pm = tomllib.loads((ROOT / "scripts" / "power_model.toml").read_text(encoding="utf-8"))
    return pm["SBC"], pm["MCU"]


def sbc_p_inf(sbc, threads, k):
    """Potência de inferência por número de threads: interpolação linear entre 1 e 4 núcleos."""
    p1, p4 = sbc["active_1"][k], sbc["active_4"][k]
    return p1 + (p4 - p1) * (threads - 1) / 3.0


def energy_rows(df, platform, threads, sbc, mcu):
    """Energia por inferência (mJ) nos cenários nominal (0), mínimo (1) e máximo (2). Tempos em ms."""
    out = {}
    for name, k in (("nom", 0), ("min", 1), ("max", 2)):
        if platform == "SBC":
            e = (sbc["active_1"][k] * (df.t_pre + df.t_post) + sbc_p_inf(sbc, threads, k) * df.t_inf)
        else:
            board = {"nom": mcu["board_overhead"][0], "min": mcu["board_overhead"][1],
                     "max": mcu["board_overhead"][2]}[name]
            e = (mcu["active"][k] + board) * df.t_total
        out[name] = e  # W * ms = mJ
    return out


# ----------------------------------------------------------------------------- carga dos dados
def load_e3():
    data = {}
    for cid, (fn, plat, prec, th) in CONFIGS.items():
        d = pd.read_csv(RAW / fn)
        d = d[d.is_warmup == 0].copy()
        for p in PHASES:
            d[p] = d[p + "_us"] / 1000.0
        d["config"] = cid
        data[cid] = d.reset_index(drop=True)
    return data


def describe(x):
    x = np.asarray(x, float)
    q1, q3 = np.percentile(x, [25, 75])
    return {"n": int(x.size), "median": r(np.median(x)), "iqr": r(q3 - q1), "mean": r(np.mean(x)),
            "sd": r(np.std(x, ddof=1)), "cv_pct": r(100 * np.std(x, ddof=1) / np.mean(x), 4),
            "min": r(x.min()), "max": r(x.max())}


# ----------------------------------------------------------------------------- piloto (E2)
def pilot(csv, platform):
    d = pd.read_csv(csv)
    t = d.t_total_us.to_numpy() / 1000.0
    meds = [float(np.median(t[i:i + 10])) for i in range(0, len(t) - len(t) % 10, 10)]
    rel = [abs(meds[i + 1] - meds[i]) / meds[i] * 100 for i in range(len(meds) - 1)]
    w = 5  # piso
    for i in range(len(rel) - 1):  # primeira posição com duas variações consecutivas < 2 %
        if rel[i] < 2 and rel[i + 1] < 2:
            w = max(5, 10 * i)
            break
    post = t[w:]
    cv = float(np.std(post, ddof=1) / np.mean(post))
    n = math.ceil((1.96 * cv / 0.01) ** 2)
    return {"platform": platform, "n_pilot": int(len(t)), "window_medians_ms": [r(m) for m in meds],
            "window_rel_diff_pct": [r(v, 4) for v in rel], "W": int(w), "cv_t_total_pct": r(100 * cv, 4),
            "N_formula": int(n), "N_final": int(max(30, n))}


# ----------------------------------------------------------------------------- detecção nas 50 imagens
def iou_xyxy(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(zip_path, manifest):
    z = zipfile.ZipFile(zip_path)
    gt = {}
    for item in manifest["inputs"]:
        stem = os.path.splitext(item["orig_file"])[0]
        s = item["letterbox_scale"]
        w, h = item["orig_width"], item["orig_height"]
        boxes = []
        txt = z.read(f"labels/test/{stem}.txt").decode()
        for line in txt.strip().splitlines():
            c, cx, cy, bw, bh = line.split()[:5]
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            boxes.append((int(c), [(cx - bw / 2) * s, (cy - bh / 2) * s, (cx + bw / 2) * s, (cy + bh / 2) * s]))
        gt[item["image_id"]] = boxes
    return gt


def match_eval(pred_json, gt, iou_thr=0.5):
    """Casamento guloso por classe (pontuação decrescente), IoU >= 0,5. Retorna TP/FP/FN por classe."""
    P = json.loads(Path(pred_json).read_text(encoding="utf-8"))
    tp = np.zeros(11, int); fp = np.zeros(11, int); fn = np.zeros(11, int)
    for img in P["images"]:
        g = gt[img["image_id"]]
        used = [False] * len(g)
        dets = sorted(img["detections"], key=lambda d: -d["score"])
        for d in dets:
            best, bj = 0.0, -1
            for j, (gc, gb) in enumerate(g):
                if used[j] or gc != d["cls"]:
                    continue
                v = iou_xyxy(d["bbox_xyxy"], gb)
                if v > best:
                    best, bj = v, j
            if best >= iou_thr:
                used[bj] = True; tp[d["cls"]] += 1
            else:
                fp[d["cls"]] += 1
        for j, (gc, _) in enumerate(g):
            if not used[j]:
                fn[gc] += 1
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    rc = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * rc / (p + rc) if p + rc and not math.isnan(p) and not math.isnan(rc) else float("nan")
    return p, rc, f1


# ----------------------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-zip", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    sbc, mcu = load_power()
    R: dict = {"seed": RNG_SEED, "n_boot": N_BOOT}

    # E0 (acurácia) — lido do relatório de avaliação
    e0 = json.loads((PROC / "e0_map_test.json").read_text(encoding="utf-8"))
    fb, ib = e0["yolox_nano_ppe_float32.tflite|BGR"], e0["yolox_nano_ppe_int8.tflite|BGR"]
    bs = e0["bootstrap_ci_95_summary"]
    R["E0"] = {
        "n_images": e0["n_images"], "n_gt": e0["n_gt"],
        "fp32": {"mAP50": fb["mAP50"], "mAP50_95": fb["mAP50_95"], "AP50_vest": fb["AP50_per_class"]["vest"],
                 "mAP50_ci95": bs["fp32_BGR"]["mAP50_ci95"], "mAP50_95_ci95": bs["fp32_BGR"]["mAP50_95_ci95"],
                 "AP50_vest_ci95": bs["fp32_BGR"]["AP50_vest_ci95"]},
        "int8": {"mAP50": ib["mAP50"], "mAP50_95": ib["mAP50_95"], "AP50_vest": ib["AP50_per_class"]["vest"],
                 "mAP50_ci95": bs["int8_BGR"]["mAP50_ci95"], "mAP50_95_ci95": bs["int8_BGR"]["mAP50_95_ci95"],
                 "AP50_vest_ci95": bs["int8_BGR"]["AP50_vest_ci95"]},
        "rho": {"mAP50": r(ib["mAP50"] / fb["mAP50"], 4), "mAP50_95": r(ib["mAP50_95"] / fb["mAP50_95"], 4),
                "AP50_vest": r(ib["AP50_per_class"]["vest"] / fb["AP50_per_class"]["vest"], 4),
                "mAP50_ci95": bs["retention_rho_int8_over_fp32_BGR"]["rho_mAP50_ci95"],
                "mAP50_95_ci95": bs["retention_rho_int8_over_fp32_BGR"]["rho_mAP50_95_ci95"],
                "AP50_vest_ci95": bs["retention_rho_int8_over_fp32_BGR"]["rho_AP50_vest_ci95"]},
        "fp32_RGB_mAP50": e0["yolox_nano_ppe_float32.tflite|RGB"]["mAP50"],
        "AP50_per_class_fp32": fb["AP50_per_class"], "AP50_per_class_int8": ib["AP50_per_class"],
    }

    # E2 (piloto)
    R["E2"] = {"SBC": pilot(RAW / "rpi_e2_pilot_100.csv", "SBC"),
               "MCU": pilot(RAW / "esp32_e2_pilot_100.csv", "MCU")}

    # E3/E3b — descritiva
    data = load_e3()
    rows = []
    R["E3"] = {}
    for cid, d in data.items():
        R["E3"][cid] = {"n": int(len(d)), "blocks": int(d.block_id.nunique()),
                        "images": int(d.image_id.nunique())}
        for p in PHASES:
            s = describe(d[p]); s.update(config=cid, phase=p); rows.append(s)
            R["E3"][cid][p] = s
        R["E3"][cid]["t_total_median_ci95"] = [r(v) for v in boot_ci_median(d.t_total, rng)]
        R["E3"][cid]["throughput_inf_s"] = r(1000.0 / np.median(d.t_total), 4)
        # efeito de bloco e autocorrelação (ordem de execução)
        groups = [g.t_total.to_numpy() for _, g in d.groupby("block_id")]
        R["E3"][cid]["kw_block_p"] = r(stats.kruskal(*groups).pvalue, 4)
        R["E3"][cid]["kw_block_p_tinf"] = r(stats.kruskal(*[g.t_inf.to_numpy() for _, g in d.groupby("block_id")]).pvalue, 4)
        R["E3"][cid]["acf1_t_total"] = r(acf1(d.sort_values(["block_id", "seq_idx"]).t_total), 4)
        R["E3"][cid]["acf1_t_inf"] = r(acf1(d.sort_values(["block_id", "seq_idx"]).t_inf), 4)
        if "cpu_temp_c" in d and d.cpu_temp_c.notna().any():
            R["E3"][cid]["temp_c_range"] = [r(d.cpu_temp_c.min(), 1), r(d.cpu_temp_c.max(), 1)]
            R["E3"][cid]["throttled_any"] = bool((d.throttled.astype(str) != "0x0").any())
    pd.DataFrame(rows)[["config", "phase", "n", "median", "iqr", "mean", "sd", "cv_pct", "min", "max"]] \
        .to_csv(OUT / "descritiva_E3.csv", index=False)

    # Energia por inferência
    erows = []
    E = {}
    for cid, d in data.items():
        _, plat, prec, th = CONFIGS[cid]
        en = energy_rows(d, plat, th, sbc, mcu)
        E[cid] = en
        R["E3"][cid]["energy_mJ"] = {k: describe(v) for k, v in en.items()}
        R["E3"][cid]["inf_per_J_nom"] = r(1000.0 / np.median(en["nom"]), 4)
        R["E3"][cid]["EDP_mJ_s_nom"] = r(np.median(en["nom"] * d.t_total / 1000.0), 3)
        for i in range(len(d)):
            erows.append({"config": cid, "block_id": int(d.block_id[i]), "seq_idx": int(d.seq_idx[i]),
                          "image_id": d.image_id[i], "t_total_ms": d.t_total[i],
                          "E_mJ_nom": en["nom"][i], "E_mJ_min": en["min"][i], "E_mJ_max": en["max"][i]})
    pd.DataFrame(erows).to_csv(OUT / "energia_E3.csv", index=False)

    # H1 — SBC int8 4t x MCU int8
    s, m = data["SBC_int8_4t"], data["MCU_int8"]
    h1 = {}
    h1["throughput"] = {
        "mw_p_one_sided_tSBC_lt_tMCU": float(stats.mannwhitneyu(s.t_total, m.t_total, alternative="less").pvalue),
        "cliff_delta_tSBC_vs_tMCU": cliffs_delta(s.t_total, m.t_total),
        "ratio_median_tMCU_over_tSBC": r(np.median(m.t_total) / np.median(s.t_total), 3),
        "ratio_ci95": [r(v, 3) for v in boot_ci_ratio_medians(m.t_total, s.t_total, rng)],
        "ratio_tinf_MCU_over_SBC": r(np.median(m.t_inf) / np.median(s.t_inf), 3),
        "ratio_tinf_ci95": [r(v, 3) for v in boot_ci_ratio_medians(m.t_inf, s.t_inf, rng)],
        "ratio_tpre_MCU_over_SBC": r(np.median(m.t_pre) / np.median(s.t_pre), 3),
        "ratio_tpost_MCU_over_SBC": r(np.median(m.t_post) / np.median(s.t_post), 3),
    }
    scen = {"nominal": ("nom", "nom"), "favoravel_H1": ("max", "min"), "desfavoravel_H1": ("min", "max")}
    h1["energy"] = {}
    for nm, (ks, km) in scen.items():
        es, em = E["SBC_int8_4t"][ks], E["MCU_int8"][km]
        h1["energy"][nm] = {
            "E_SBC_median_mJ": r(np.median(es)), "E_MCU_median_mJ": r(np.median(em)),
            "mw_p_one_sided_ESBC_ge_EMCU": float(stats.mannwhitneyu(es, em, alternative="greater").pvalue),
            "mw_p_one_sided_ESBC_lt_EMCU": float(stats.mannwhitneyu(es, em, alternative="less").pvalue),
            "mw_p_two_sided": float(stats.mannwhitneyu(es, em, alternative="two-sided").pvalue),
            "cliff_delta_ESBC_vs_EMCU": cliffs_delta(es, em),
            "ratio_EMCU_over_ESBC": r(np.median(em) / np.median(es), 3),
            "ratio_ci95_variabilidade_temporal": [r(v, 3) for v in boot_ci_ratio_medians(em, es, rng)],
        }
    h1["throughput"]["mw_p_two_sided"] = float(stats.mannwhitneyu(s.t_total, m.t_total, alternative="two-sided").pvalue)

    # break-even: tempo máximo do MCU para igualar a energia do SBC; razão de potências exigida
    t_ratio = float(np.median(m.t_total) / np.median(s.t_total))
    be = {}
    for nm, (ks, km) in scen.items():
        p_m = mcu["active"][{"nom": 0, "min": 1, "max": 2}[km]] + {"nom": mcu["board_overhead"][0],
                                                                   "min": mcu["board_overhead"][1],
                                                                   "max": mcu["board_overhead"][2]}[km]
        e_s = float(np.median(E["SBC_int8_4t"][ks]))
        p_s_eff = e_s / float(np.median(s.t_total))  # potência média efetiva do SBC na janela (W)
        be[nm] = {"P_MCU_W": r(p_m, 4), "P_SBC_eff_W": r(p_s_eff, 4),
                  "power_ratio_available": r(p_s_eff / p_m, 3),
                  "t_MCU_breakeven_ms": r(e_s / p_m, 1),
                  "speedup_needed_MCU": r(float(np.median(m.t_total)) / (e_s / p_m), 2)}
    h1["breakeven"] = {"time_ratio_needed": r(t_ratio, 3), **be}
    R["H1"] = h1

    # Energia por ciclo em função do período T (= 1/f)
    def e_cycle(cid, T_s, k):
        d = data[cid]; e = float(np.median(E[cid][k])); t = float(np.median(d.t_total)) / 1000
        if T_s < t:
            return None
        if CONFIGS[cid][1] == "SBC":
            p_idle = sbc["idle"][{"nom": 0, "min": 1, "max": 2}[k]]
        else:
            p_idle = mcu["idle"][{"nom": 0, "min": 1, "max": 2}[k]] + mcu["board_overhead"][{"nom": 0, "min": 1, "max": 2}[k]]
        return e + p_idle * 1000 * (T_s - t)
    R["E_ciclo"] = {}
    for f in (1.0, 0.5, 0.2, 0.1, 1 / 60):
        T = 1 / f
        R["E_ciclo"][f"{f:.4g}Hz"] = {cid: {k: r(e_cycle(cid, T, k), 1) for k in ("nom", "min", "max")}
                                      for cid in ("SBC_int8_4t", "SBC_fp32_4t", "MCU_int8")}
    # período a partir do qual o MCU iguala o SBC (forma fechada; retas em T para T >= t_MCU)
    KI = {"nom": 0, "min": 1, "max": 2}
    def line(cid, k):
        e = float(np.median(E[cid][k])); t = float(np.median(data[cid].t_total)) / 1000
        if CONFIGS[cid][1] == "SBC":
            p = sbc["idle"][KI[k]]
        else:
            p = mcu["idle"][KI[k]] + mcu["board_overhead"][KI[k]]
        return e - p * 1000 * t, p * 1000, t  # E(T) = a + b*T  (mJ, T em s)
    def cross(km, ks):
        am, bm, tm_ = line("MCU_int8", km); as_, bs, _ = line("SBC_int8_4t", ks)
        if am + bm * tm_ <= as_ + bs * tm_:
            return r(tm_, 3)  # MCU já vence no primeiro período viável
        if bm >= bs:
            return None       # nunca empata
        return r((am - as_) / (bs - bm), 2)
    R["E_ciclo"]["T_min_MCU_le_SBC_s"] = {"nominal": cross("nom", "nom"),
                                          "desfavoravel_MCU": cross("max", "min"),
                                          "favoravel_MCU": cross("min", "max")}
    # acréscimo de placa a partir do qual o MCU nunca empata no pior caso (ocioso MCU >= ocioso mínimo do SBC)
    R["E_ciclo"]["overhead_limite_W"] = r(sbc["idle"][1] - mcu["idle"][2], 4)
    R["E_ciclo"]["ratio_SBC_over_MCU_nom"] = {f: r(R["E_ciclo"][f]["SBC_int8_4t"]["nom"] / R["E_ciclo"][f]["MCU_int8"]["nom"], 3)
                                              for f in ("0.2Hz", "0.1Hz", "0.01667Hz")}
    R["E_ciclo"]["f_max_MCU_Hz"] = r(1000 / float(np.median(m.t_total)), 4)

    # H2 — SBC int8 x fp32 (4 threads)
    i8, f32 = data["SBC_int8_4t"], data["SBC_fp32_4t"]
    R["H2"] = {
        "ratio_tinf_fp32_over_int8": r(np.median(f32.t_inf) / np.median(i8.t_inf), 3),
        "ratio_tinf_ci95": [r(v, 3) for v in boot_ci_ratio_medians(f32.t_inf, i8.t_inf, rng)],
        "ratio_ttotal_fp32_over_int8": r(np.median(f32.t_total) / np.median(i8.t_total), 3),
        "ratio_ttotal_ci95": [r(v, 3) for v in boot_ci_ratio_medians(f32.t_total, i8.t_total, rng)],
        "ratio_E_fp32_over_int8_nom": r(np.median(E["SBC_fp32_4t"]["nom"]) / np.median(E["SBC_int8_4t"]["nom"]), 3),
        "ratio_E_ci95": [r(v, 3) for v in boot_ci_ratio_medians(E["SBC_fp32_4t"]["nom"], E["SBC_int8_4t"]["nom"], rng)],
        "mw_p_two_sided_ttotal": float(stats.mannwhitneyu(i8.t_total, f32.t_total).pvalue),
        "cliff_delta_ttotal_int8_vs_fp32": cliffs_delta(i8.t_total, f32.t_total),
    }

    # E3b — Amdahl (t_inf)
    t1 = np.median(data["SBC_int8_1t"].t_inf); t2 = np.median(data["SBC_int8_2t"].t_inf)
    t4 = np.median(data["SBC_int8_4t"].t_inf)
    S = {1: 1.0, 2: t1 / t2, 4: t1 / t4}
    # p por mínimos quadrados em 1/S = (1-p) + p/N
    Ns = np.array([2, 4]); y = 1 / np.array([S[2], S[4]]); xN = 1 / Ns - 1
    p_ls = float(np.sum(xN * (y - 1)) / np.sum(xN * xN))
    R["E3b"] = {"S2": r(S[2], 4), "S4": r(S[4], 4),
                "S2_ci95": [r(v, 4) for v in boot_ci_ratio_medians(data["SBC_int8_1t"].t_inf, data["SBC_int8_2t"].t_inf, rng)],
                "S4_ci95": [r(v, 4) for v in boot_ci_ratio_medians(data["SBC_int8_1t"].t_inf, data["SBC_int8_4t"].t_inf, rng)],
                "p_from_S2": r((1 - 1 / S[2]) / (1 - 1 / 2), 4), "p_from_S4": r((1 - 1 / S[4]) / (1 - 1 / 4), 4),
                "p_ls": r(p_ls, 4), "S_inf_limit": r(1 / (1 - p_ls), 3),
                "efficiency_4": r(S[4] / 4, 4),
                "S4_ttotal": r(np.median(data["SBC_int8_1t"].t_total) / np.median(data["SBC_int8_4t"].t_total), 4)}

    # E4 — 60 s em laço fechado
    e4s = pd.read_csv(RAW / "rpi_e4_continuous_60s.csv")
    e4m = pd.read_csv(RAW / "esp32_e4_continuous_60s.csv")
    first, last = e4s[e4s.elapsed_s <= 10], e4s[e4s.elapsed_s >= e4s.elapsed_s.max() - 10]
    R["E4"] = {
        "SBC": {"n": int(len(e4s)), "elapsed_s": r(e4s.elapsed_s.max(), 2),
                "throughput_inf_s": r(len(e4s) / e4s.elapsed_s.max(), 3),
                "t_inf_median_ms": r(e4s.t_inf_ms.median()), "t_inf_p99_ms": r(e4s.t_inf_ms.quantile(.99)),
                "t_inf_max_ms": r(e4s.t_inf_ms.max()),
                "t_inf_median_first10s": r(first.t_inf_ms.median()), "t_inf_median_last10s": r(last.t_inf_ms.median()),
                "temp_start_c": r(e4s.cpu_temp_c.iloc[0], 1), "temp_max_c": r(e4s.cpu_temp_c.max(), 1),
                "freq_mhz_unique": sorted(e4s.cpu_freq_mhz.unique().tolist()),
                "throttled_unique": sorted(e4s.throttled_hex.astype(str).unique().tolist()),
                "spearman_tinf_vs_time": r(stats.spearmanr(e4s.elapsed_s, e4s.t_inf_ms).statistic, 4)},
        "MCU": {"n": int(len(e4m)), "elapsed_s": r(e4m.elapsed_s.max(), 2),
                "throughput_inf_s": r(len(e4m) / e4m.elapsed_s.max(), 4),
                "t_inf_median_ms": r(e4m.t_inf_ms.median()), "t_inf_range_ms": [r(e4m.t_inf_ms.min()), r(e4m.t_inf_ms.max())]},
    }

    # E5 — memória
    R["E5"] = {"SBC": json.loads((RAW / "rpi_e5_memory_footprint.json").read_text(encoding="utf-8")),
               "MCU": json.loads((RAW / "esp32_e5_memory_footprint.json").read_text(encoding="utf-8"))}
    R["E5"]["SBC"].pop("e2_pilot_summary", None)

    # E1 — equivalência (relatório) + detecção no dispositivo
    e1 = json.loads((PROC / "e1_equivalence_report.json").read_text(encoding="utf-8"))
    R["E1"] = {c["comparison"]: {k: v for k, v in c.items() if k != "per_image"} for c in e1["comparisons"]}
    if a.dataset_zip:
        man = json.loads((PROC / "inputs_50_manifest.json").read_text(encoding="utf-8"))
        gt = load_gt(a.dataset_zip, man)
        n_gt = np.zeros(11, int)
        for g in gt.values():
            for c, _ in g:
                n_gt[c] += 1
        det_rows = []
        R["deteccao_50"] = {"n_images": len(gt), "n_gt": int(n_gt.sum()), "n_gt_vest": int(n_gt[2]),
                            "conf_thr": 0.25, "nms_thr": 0.45, "iou_thr": 0.5}
        for plat, fn in (("PC (LiteRT)", "pc_e1_50_predictions.json"), ("SBC (LiteRT)", "sbc_e1_50_predictions.json"),
                         ("PC (TFLM ref.)", "pc_tflm_e1_50_predictions.json"), ("MCU (TFLM+ESP-NN)", "mcu_e1_50_predictions.json")):
            tp, fp, fn_ = match_eval(RAW / fn, gt)
            P_, Rc, F1 = prf(tp.sum(), fp.sum(), fn_.sum())
            Pv, Rv, Fv = prf(tp[2], fp[2], fn_[2])
            row = {"platform": plat, "n_det": int(tp.sum() + fp.sum()), "TP": int(tp.sum()), "FP": int(fp.sum()),
                   "FN": int(fn_.sum()), "precision": r(P_, 4), "recall": r(Rc, 4), "F1": r(F1, 4),
                   "vest_TP": int(tp[2]), "vest_FP": int(fp[2]), "vest_FN": int(fn_[2]),
                   "vest_precision": r(Pv, 4), "vest_recall": r(Rv, 4), "vest_F1": r(Fv, 4)}
            det_rows.append(row)
        pd.DataFrame(det_rows).to_csv(OUT / "deteccao_50.csv", index=False)
        R["deteccao_50"]["por_plataforma"] = det_rows

    R["power_model"] = {"SBC": sbc, "MCU": mcu}
    (OUT / "resultados.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"ok -> {OUT}")


if __name__ == "__main__":
    main()
