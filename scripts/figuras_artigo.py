# -*- coding: utf-8 -*-
"""Figuras do artigo (docs/projeto/figs/) a partir de data/processed/artigo/ (gerado por analise_artigo.py)."""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "data" / "processed" / "artigo"
FIG = ROOT / "docs" / "projeto" / "figs"
RAW = ROOT / "data" / "raw"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlelocation": "left", "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.formatter.use_locale": False,
})
C = {"SBC_int8_4t": "#1f5fa8", "SBC_int8_2t": "#5b8fd0", "SBC_int8_1t": "#9dbbe3",
     "SBC_fp32_4t": "#d07a1c", "MCU_int8": "#7b3fa0"}
LBL = {"SBC_int8_4t": "SBC int8\n4 threads", "SBC_int8_2t": "SBC int8\n2 threads", "SBC_int8_1t": "SBC int8\n1 thread",
       "SBC_fp32_4t": "SBC fp32\n4 threads", "MCU_int8": "MCU int8\nESP32-S3"}
ORDER = ["SBC_int8_4t", "SBC_int8_2t", "SBC_int8_1t", "SBC_fp32_4t", "MCU_int8"]
GREY = "#6b6b6b"


def br(x, nd=1):
    """Número com vírgula decimal."""
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def letter(ax, s):
    ax.text(-0.13, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def main():
    R = json.loads((A / "resultados.json").read_text(encoding="utf-8"))
    en = pd.read_csv(A / "energia_E3.csv")
    pm = tomllib.loads((ROOT / "scripts" / "power_model.toml").read_text(encoding="utf-8"))
    FIG.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ Fig. 1: latência e Amdahl
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.6), gridspec_kw={"width_ratios": [1.45, 1]})
    rng = np.random.default_rng(0)
    for i, cid in enumerate(ORDER):
        t = en[en.config == cid].t_total_ms.to_numpy()
        a1.scatter(i + rng.uniform(-0.13, 0.13, t.size), t, s=7, color=C[cid], alpha=0.75, lw=0, zorder=2)
        med = np.median(t)
        a1.plot([i - 0.25, i + 0.25], [med, med], color="black", lw=1.0, zorder=3)
        a1.text(i, med * 1.32, f"{br(med)} ms", ha="center", va="bottom", fontsize=6.5)
    a1.set_yscale("log")
    a1.set_ylim(70, 12000)
    a1.set_yticks([100, 300, 1000, 3000, 10000]); a1.set_yticklabels(["100", "300", "1.000", "3.000", "10.000"])
    a1.set_xticks(range(len(ORDER))); a1.set_xticklabels([LBL[c] for c in ORDER], fontsize=6)
    a1.set_xlim(-0.6, len(ORDER) - 0.4)
    a1.set_ylabel(r"Latência $t_{total}$ (ms, escala log.)")
    a1.set_title(f"MCU leva {br(R['H1']['throughput']['ratio_median_tMCU_over_tSBC'])}× mais que o SBC int8 (4 threads)")
    letter(a1, "a")

    e3b = R["E3b"]; p = e3b["p_ls"]
    N = np.linspace(1, 8, 200)
    a2.plot(N, N, ls="--", color=GREY, lw=0.8)
    a2.text(5.05, 4.35, "ideal ($S=N$)", color=GREY, fontsize=6.5)
    Nin, Nex = N[N <= 4], N[N >= 4]
    a2.plot(Nin, 1 / ((1 - p) + p / Nin), color=C["SBC_int8_4t"], lw=1.2)
    a2.plot(Nex, 1 / ((1 - p) + p / Nex), color=C["SBC_int8_4t"], lw=1.0, ls="--")
    a2.text(5.0, 2.45, f"Amdahl, $p$ = {br(100 * p)}% (extrapolado)", color=C["SBC_int8_4t"], fontsize=6.5)
    Sm = [1.0, e3b["S2"], e3b["S4"]]
    a2.scatter([1, 2, 4], Sm, s=18, color=C["SBC_int8_4t"], zorder=3, edgecolor="white", lw=0.5)
    for n_, s_ in zip([2, 4], Sm[1:]):
        a2.annotate(f"{br(s_, 2)}×", (n_, s_), xytext=(4, -9), textcoords="offset points", fontsize=6.5)
    a2.axhline(e3b["S_inf_limit"], color=GREY, lw=0.6, ls=":")
    a2.text(1.1, e3b["S_inf_limit"] + 0.12, f"limite {br(e3b['S_inf_limit'], 2)}×", color=GREY, fontsize=6.5)
    a2.set_xlim(0.8, 8.2); a2.set_ylim(0.8, 6.2)
    a2.set_xlabel("Threads $N$ (SBC, int8)"); a2.set_ylabel(r"Aceleração $S(N)=t_{inf}(1)/t_{inf}(N)$")
    a2.set_title("4 threads: 2,47× (eficiência de 62%)")
    letter(a2, "b")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / "fig_latencia.pdf"); fig.savefig(FIG / "fig_latencia.png")
    plt.close(fig)

    # ------------------------------------------------------------------ Fig. 2: energia por inferência e break-even
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(6.3, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
    for i, cid in enumerate(ORDER):
        e = R["E3"][cid]["energy_mJ"]
        b1.plot([i, i], [e["min"]["median"], e["max"]["median"]], color=C[cid], lw=2.2, alpha=0.45, solid_capstyle="butt")
        b1.scatter([i], [e["nom"]["median"]], s=22, color=C[cid], zorder=3, edgecolor="white", lw=0.5)
        b1.text(i + 0.12, e["nom"]["median"], br(e["nom"]["median"], 0), fontsize=6.5, va="center")
    b1.set_xticks(range(len(ORDER))); b1.set_xticklabels([LBL[c] for c in ORDER])
    b1.set_xlim(-0.5, len(ORDER) - 0.3)
    b1.set_ylim(0, 4200)
    b1.set_ylabel("Energia por inferência $E$ (mJ)")
    b1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: br(v, 0)))
    b1.set_title("Ponto: nominal; barra: faixa de potência")
    letter(b1, "a")

    s_e = R["E3"]["SBC_int8_4t"]["energy_mJ"]; t_m = R["E3"]["MCU_int8"]["t_total"]["median"]
    t = np.linspace(0, 5600, 300)
    ymax2 = 4600
    b2.axhspan(s_e["min"]["median"], s_e["max"]["median"], color=C["SBC_int8_4t"], alpha=0.15, lw=0)
    b2.axhline(s_e["nom"]["median"], color=C["SBC_int8_4t"], lw=1.0)
    b2.text(5550, s_e["max"]["median"] + 60, "SBC int8 (4 threads)", color=C["SBC_int8_4t"], fontsize=6.5, ha="right")
    act = pm["MCU"]["active"]; ov = pm["MCU"]["board_overhead"][2]
    for pw, ls, lab in ((act[2] + ov, "--", "máxima + placa"), (act[0], "-", "nominal"), (act[1], ":", "mínima")):
        b2.plot(t, pw * t, color=C["MCU_int8"], ls=ls, lw=1.0, label=f"MCU, {lab} ({br(pw, 3)} W)")
    b2.legend(loc="upper left", handlelength=2.2, borderaxespad=0.2)
    b2.axvline(t_m, color=GREY, lw=0.6)
    b2.text(t_m - 70, 3000, "MCU medido", color=GREY, fontsize=6.5, ha="right", va="center", rotation=90)
    tb = R["H1"]["breakeven"]["nominal"]["t_MCU_breakeven_ms"]
    b2.scatter([tb], [s_e["nom"]["median"]], s=16, color="black", zorder=4)
    b2.annotate(f"equilíbrio nominal: {br(tb, 0)} ms", (tb, s_e["nom"]["median"]), xytext=(1900, 95),
                textcoords="data", va="center", fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.5))
    b2.set_xlim(0, 5600); b2.set_ylim(0, ymax2)
    b2.set_xlabel("Latência do MCU $t_{MCU}$ (ms)"); b2.set_ylabel("Energia por inferência (mJ)")
    b2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: br(v, 0))); b2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: br(v, 0)))
    b2.set_title(f"MCU precisaria ser {br(R['H1']['breakeven']['nominal']['speedup_needed_MCU'])}× mais rápido")
    letter(b2, "b")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / "fig_energia.pdf"); fig.savefig(FIG / "fig_energia.png")
    plt.close(fig)

    # ------------------------------------------------------------------ Fig. 3: energia por ciclo em função do período
    def ecyc(cid, T, k):
        e = R["E3"][cid]["energy_mJ"][k]["median"]; tt = R["E3"][cid]["t_total"]["median"] / 1000
        idx = {"nom": 0, "min": 1, "max": 2}[k]
        if cid.startswith("SBC"):
            pi = pm["SBC"]["idle"][idx]
        else:
            pi = pm["MCU"]["idle"][idx] + pm["MCU"]["board_overhead"][idx]
        return np.where(T >= tt, e + pi * 1000 * (T - tt), np.nan)

    fig, c1 = plt.subplots(figsize=(6.3, 2.7))
    T = np.logspace(np.log10(0.1), np.log10(300), 700)
    for cid, name in (("SBC_int8_4t", "SBC int8 (4 threads)"), ("MCU_int8", "MCU int8 (ESP32-S3)")):
        lo, hi, nom = ecyc(cid, T, "min"), ecyc(cid, T, "max"), ecyc(cid, T, "nom")
        c1.fill_between(T, lo / 1000, hi / 1000, color=C[cid], alpha=0.15, lw=0)
        c1.plot(T, nom / 1000, color=C[cid], lw=1.3)
        j = np.nanargmin(np.abs(T - 25))
        c1.text(T[j], nom[j] / 1000 * (1.9 if cid.startswith("SBC") else 0.45), name, color=C[cid], fontsize=7, ha="center")
    tm = R["E3"]["MCU_int8"]["t_total"]["median"] / 1000
    c1.axvspan(0.1, tm, color=GREY, alpha=0.08, lw=0)
    c1.text(0.12, 150, "MCU não cumpre\no período ($t_{MCU}$ > 1/$f$)", color=GREY, fontsize=6.5, va="top")
    for f in (1, 0.5, 0.2):
        c1.axvline(1 / f, color=GREY, lw=0.5, ls=":")
        c1.text(1 / f * 1.04, 0.13, f"{br(f, 1)} Hz", color=GREY, fontsize=6, rotation=90, va="bottom")
    tc = R["E_ciclo"]["T_min_MCU_le_SBC_s"]["desfavoravel_MCU"]
    c1.axvline(tc, color=C["MCU_int8"], lw=0.5, ls="--")
    c1.text(tc * 1.05, 0.13, f"{br(tc, 0)} s", color=C["MCU_int8"], fontsize=6, rotation=90, va="bottom")
    c1.set_xscale("log"); c1.set_yscale("log")
    c1.set_xlim(0.1, 300); c1.set_ylim(0.1, 300)
    c1.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 60, 120, 300]); c1.set_xticklabels(["0,1", "0,2", "0,5", "1", "2", "5", "10", "20", "60", "120", "300"])
    c1.set_yticks([0.1, 0.3, 1, 3, 10, 30, 100, 300]); c1.set_yticklabels(["0,1", "0,3", "1", "3", "10", "30", "100", "300"])
    c1.set_xlabel("Período de inspeção $1/f$ (s)"); c1.set_ylabel(r"Energia por ciclo $E_{ciclo}$ (J)")
    c1.set_title("No cenário nominal o MCU gasta menos por ciclo em todo período que consegue cumprir")
    fig.tight_layout()
    fig.savefig(FIG / "fig_ciclo.pdf"); fig.savefig(FIG / "fig_ciclo.png")
    plt.close(fig)
    print("figuras ok")


if __name__ == "__main__":
    main()
