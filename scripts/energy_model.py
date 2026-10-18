#!/usr/bin/env python3
"""Energia por inferência estimada por software (protocolo v1.3, D24/D29).

E_i = P_pre*t_pre + P_inf*t_inf + P_post*t_post, com potências de scripts/power_model.toml.
Gera, para cada CSV bruto, a energia nominal e nas bordas da faixa (sensibilidade), inf/J e EDP;
e um resumo com o break-even  P_SBC/P_MCU  =  t_MCU/t_SBC.

Uso:
  python scripts/energy_model.py data/raw/*_E3_*.csv --out data/processed
"""
from __future__ import annotations

import argparse
import glob
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def phase_power(cfg: dict, platform: str, phase: str, threads: int, k: int) -> float:
    """k: 0 = nominal, 1 = mínimo, 2 = máximo."""
    p = cfg[platform]
    mode = p[phase]
    if mode == "by_threads":
        lo, hi = p["active_1"][k], p["active_4"][k]
        val = lo + (hi - lo) * (min(max(threads, 1), 4) - 1) / 3
    else:
        val = p[mode][k]
    if platform == "MCU":
        val += p["board_overhead"][k]
    return val


def energy(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    for k, tag in enumerate(("nom", "min", "max")):
        e = np.zeros(len(df))
        for phase in ("pre", "inf", "post"):
            pw = np.array([phase_power(cfg, pl, phase, th, k)
                           for pl, th in zip(df["platform"], df["threads"])])
            e += pw * df[f"t_{phase}_us"].to_numpy() * 1e-6  # J
        out[f"E_mJ_{tag}"] = e * 1e3
    out["inf_per_J_nom"] = 1e3 / out["E_mJ_nom"]
    out["EDP_mJms_nom"] = out["E_mJ_nom"] * out["t_total_us"] / 1e3
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--power", type=Path, default=HERE / "power_model.toml")
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--include-warmup", action="store_true")
    a = ap.parse_args()

    cfg = tomllib.loads(a.power.read_text(encoding="utf-8"))
    files = [f for pat in a.csv for f in glob.glob(pat)]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if not a.include_warmup:
        df = df[df["is_warmup"] == 0]
    res = energy(df, cfg)
    a.out.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out / "energy_per_inference.csv", index=False)

    summ = res.groupby("platform").agg(
        n=("t_total_us", "size"),
        t_total_ms_median=("t_total_us", lambda s: s.median() / 1e3),
        E_mJ_nom_median=("E_mJ_nom", "median"),
        E_mJ_min_median=("E_mJ_min", "median"),
        E_mJ_max_median=("E_mJ_max", "median"),
        inf_per_J_nom_median=("inf_per_J_nom", "median"),
        EDP_nom_median=("EDP_mJms_nom", "median"))
    summ.to_csv(a.out / "energy_summary.csv")
    print(summ.round(3).to_string())

    if {"SBC", "MCU"} <= set(summ.index):
        t_ratio = summ.loc["MCU", "t_total_ms_median"] / summ.loc["SBC", "t_total_ms_median"]
        p_sbc = res.loc[res.platform == "SBC", "E_mJ_nom"].median() / (
            summ.loc["SBC", "t_total_ms_median"])  # W médio efetivo
        p_mcu = res.loc[res.platform == "MCU", "E_mJ_nom"].median() / (
            summ.loc["MCU", "t_total_ms_median"])
        print(f"\nbreak-even: H1 (E_SBC >= E_MCU) exige P_SBC/P_MCU >= t_MCU/t_SBC = {t_ratio:.2f}"
              f" | razão nominal do modelo = {p_sbc / p_mcu:.2f}")
        for tag in ("nom", "min", "max"):
            # pior caso p/ o MCU: potência MCU máx. contra SBC mín. e vice-versa
            e_s = summ.loc["SBC", f"E_mJ_{tag}_median"]
            e_m = summ.loc["MCU", f"E_mJ_{tag}_median"]
            print(f"  [{tag}] E_SBC={e_s:.2f} mJ  E_MCU={e_m:.2f} mJ  -> E_SBC>=E_MCU: {e_s >= e_m}")
        e_s_min, e_m_max = summ.loc["SBC", "E_mJ_min_median"], summ.loc["MCU", "E_mJ_max_median"]
        e_s_max, e_m_min = summ.loc["SBC", "E_mJ_max_median"], summ.loc["MCU", "E_mJ_min_median"]
        print(f"  [extremos] SBC mín vs MCU máx: {e_s_min >= e_m_max} | "
              f"SBC máx vs MCU mín: {e_s_max >= e_m_min}  (conclusão robusta se ambos iguais)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
