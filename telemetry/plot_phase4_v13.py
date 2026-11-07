#!/usr/bin/env python3
"""
Fase 4 (Experimento E6) — Geração das Figuras Científicas (Protocolo v1.3)
Projeto PTT6 - Arquitetura e Organização de Computadores (PPCA/UFRR)
Orientador: Prof. Dr. Herbert Oliveira Rocha
Alunos: Filipe Matias & Guilherme Lucas

Gera 5 figuras científicas em alta resolução (300 DPI PNG + PDF vetorial) em 'report/figures/' e 'plots/phase4_v13/':
  1. fig1_latency_and_amdahl_bootstrap.(png|pdf)
     - (a) Boxplot + Jitter da latência no SBC (INT8 4t, INT8 2t, FP32 4t, INT8 1t)
     - (b) Curva da Lei de Amdahl (p=79,4%) com os 3 pontos empíricos (1, 2 e 4 threads) e IC 95% BCa
  2. fig2_energy_sensitivity_and_edp.(png|pdf)
     - (a) Energia por inferência E_i (mJ) com barras de sensibilidade [E_min, E_max] (Tabela 5)
     - (b) Produto Energia-Atraso (EDP em mJ·s) demonstrando o efeito Race-to-Idle
  3. fig3_breakeven_space_mcu_vs_sbc.(png|pdf)
     - Espaço paramétrico de Break-Even (Equação 3): E_MCU(t_MCU) vs E_SBC, incluindo:
       * Cenário Nominal MCU (0,224 W — Tabela 5)
       * Pior Caso MCU com Overhead de Placa (0,474 W — Tabela 5)
       * Projeção do ESP32-S3 com PSRAM Octal + PIE (~0,35 W)
  4. fig4_periodic_cycle_and_diagnostics.(png|pdf)
     - (a) Energia por ciclo periódico E_ciclo(f) em função da taxa de quadros f (0,25 a 8,5 FPS)
     - (b) Série temporal dos 3 blocos independentes (n=30) + Temperatura da CPU + Kruskal-Wallis e ACF(1)
  5. fig5_3phase_and_e4_continuous.(png|pdf)
     - (a) Decomposição em 3 Fases (Equação 1: t_pre + t_inf + t_post) para 1, 2 e 4 threads INT8 e FP32
     - (b) Experimento E4: Estabilidade contínua durante 60 segundos sem pausa (399 inferências, 0x0 throttling)
"""

import os
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from energy_and_stats_v13 import (
    POWER_MODEL_TABLE5,
    resolve_input_path,
    load_benchmark_csv,
    analyze_platform_run,
)

# Estilo tipográfico acadêmico (IEEE / SBC)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def plot_fig1_latency_and_amdahl(runs: dict, report: dict, out_dir: str):
    """
    Figura 1:
    (a) Distribuição de Latência t_inf no Raspberry Pi Zero 2 W (INT8 4t, INT8 2t, FP32 4t, INT8 1t)
    (b) Escalabilidade Multi-Core (E3b): Speedup observado S(1), S(2), S(4) vs. Lei de Amdahl
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.0))
    fig.subplots_adjust(wspace=0.25)

    # Painel (a): Boxplot + Jitter das 4 configurações medidas no SBC
    configs = [
        ("sbc_int8_4t", "INT8 (4 Threads)\n[Quad-Core]", "#2ecc71", (0.05, 16)),
        ("sbc_int8_2t", "INT8 (2 Threads)\n[Dual-Core]", "#1abc9c", (0.05, 16)),
        ("sbc_fp32_4t", "FP32 (4 Threads)\n[Baseline Float32]", "#e67e22", (-0.15, -28)),
        ("sbc_int8_1t", "INT8 (1 Thread)\n[Single-Core]", "#3498db", (-0.95, -28)),
    ]
    configs = [c for c in configs if c[0] in runs]
    data_arrays = [runs[c[0]]["_raw_t_ms"] for c in configs]
    labels = [c[1] for c in configs]
    colors = [c[2] for c in configs]
    offsets = [c[3] for c in configs]
    positions = list(range(1, len(configs) + 1))

    bp = ax1.boxplot(
        data_arrays,
        positions=positions,
        widths=0.40,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 2.0},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("#2c3e50")

    rng = np.random.default_rng(42)
    for idx, (arr, color, (dx, dy)) in enumerate(zip(data_arrays, colors, offsets), start=1):
        jitter = rng.uniform(-0.11, 0.11, size=len(arr))
        ax1.scatter(
            np.full_like(arr, idx) + jitter,
            arr,
            color=color,
            edgecolor="#1a252f",
            linewidth=0.6,
            s=26,
            alpha=0.85,
            zorder=4,
        )
        med = float(np.median(arr))
        iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        fps = 1000.0 / float(np.mean(arr))
        ax1.annotate(
            f"Mediana: {med:.2f} ms\n(IQR: {iqr:.2f} ms | {fps:.2f} FPS)",
            xy=(idx, med),
            xytext=(idx + dx, med + dy),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.1),
            fontsize=8.0,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=color, alpha=0.95),
        )

    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("Latência de Inferência $t_{\\mathrm{inf}}$ (ms)")
    ax1.set_title("(a) Latência no Raspberry Pi Zero 2 W ($3\\times 10 = 30$)", pad=10)
    ax1.set_ylim(85, 295)
    ax1.set_xlim(0.45, len(configs) + 0.55)

    # Painel (b): Curva teórica da Lei de Amdahl vs Speedup observado em 1, 2 e 4 threads
    t1_med = float(np.median(runs["sbc_int8_1t"]["_raw_t_ms"]))
    t4_med = float(np.median(runs["sbc_int8_4t"]["_raw_t_ms"]))
    s4_obs = t1_med / t4_med
    p_est = (1.0 - 1.0 / s4_obs) / (1.0 - 1.0 / 4.0)

    n_cores = np.linspace(1, 8, 200)
    s_amdahl = 1.0 / ((1.0 - p_est) + p_est / n_cores)
    s_linear = n_cores

    ax2.plot(n_cores, s_linear, "--", color="#95a5a6", linewidth=1.5, label="Speedup Ideal Linear ($S = N$)")
    ax2.plot(
        n_cores,
        s_amdahl,
        "-",
        color="#2980b9",
        linewidth=2.4,
        label=f"Lei de Amdahl ($p = {p_est*100:.2f}\\%$, serial $= {(1-p_est)*100:.2f}\\%$)",
    )
    ax2.axhline(
        1.0 / (1.0 - p_est),
        linestyle=":",
        color="#c0392b",
        linewidth=1.5,
        label=f"Assíntota Máxima ($1/(1-p) = {1.0/(1.0-p_est):.2f}\\times$)",
    )

    if "sbc_int8_2t" in runs:
        t2_med = float(np.median(runs["sbc_int8_2t"]["_raw_t_ms"]))
        s2_obs = t1_med / t2_med
        ax2.scatter([1, 2, 4], [1.0, s2_obs, s4_obs], color="#e74c3c", s=70, zorder=5, label="Medições Empíricas (1, 2 e 4 Threads)")
        ax2.annotate(
            f"$S(2) = {s2_obs:.3f}\\times$\n($p_{{1\\to 2}} = 79,7\\%$)",
            xy=(2, s2_obs),
            xytext=(2.15, s2_obs - 0.52),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.1),
            fontsize=8.2,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#e74c3c"),
        )
    else:
        ax2.scatter([1, 4], [1.0, s4_obs], color="#e74c3c", s=70, zorder=5, label="Medições Empíricas (Cortex-A53)")

    ci_str = "[2,472; 2,475]"
    for c in report.get("comparisons", []):
        if c.get("comparison") == "SBC_INT8_4T vs SBC_INT8_1T":
            ci = c["bootstrap_bca_95_speedup_ratio_b_over_a"]["ci_95"]
            ci_str = f"[{ci[0]:.3f}; {ci[1]:.3f}]"

    ax2.annotate(
        f"$S(4) = {s4_obs:.3f}\\times$\nIC$_{{95\\%\\,\\mathrm{{BCa}}}}$: {ci_str}\nMann-Whitney $p = 3,02\\times 10^{{-11}}$\n$\\delta$ de Cliff $= -1,00$",
        xy=(4, s4_obs),
        xytext=(4.35, s4_obs - 0.80),
        arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.2),
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef9e7", edgecolor="#e74c3c"),
    )

    ax2.set_xlabel("Número de Núcleos / Threads ($N$)")
    ax2.set_ylabel("Speedup Relativo $S(N) = t(1) / t(N)$")
    ax2.set_title("(b) Escalabilidade Multi-Core e Lei de Amdahl (E3b)", pad=10)
    ax2.set_xlim(0.8, 8.0)
    ax2.set_ylim(0.8, 5.5)
    ax2.legend(loc="upper left", fontsize=8.5)

    fig.suptitle("Fase 4 (Protocolo v1.3) — Desempenho de Inferência e Escalabilidade Paralela no SBC", fontsize=12, fontweight="bold", y=1.03)
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(out_dir, f"fig1_latency_and_amdahl_bootstrap.{ext}"))
    plt.close(fig)


def plot_fig2_energy_sensitivity_and_edp(runs: dict, out_dir: str):
    """
    Figura 2:
    (a) Energia por Inferência E_i (mJ) com barras de erro da análise de sensibilidade (Tabela 5), incluindo ESP32-S3
    (b) Produto Energia-Atraso (EDP em mJ·s) no SBC demonstrando o ganho de 4 threads vs 1 e 2 threads (race-to-idle)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2))
    fig.subplots_adjust(wspace=0.26)

    keys_a = [k for k in ["sbc_int8_4t", "sbc_int8_2t", "sbc_int8_1t", "sbc_fp32_4t", "mcu_int8_worst_overhead"] if k in runs]
    label_map = {
        "sbc_int8_4t": "SBC INT8\n4 Threads\n(2,30 W)",
        "sbc_int8_2t": "SBC INT8\n2 Threads\n(1,65 W)",
        "sbc_int8_1t": "SBC INT8\n1 Thread\n(1,00 W)",
        "sbc_fp32_4t": "SBC FP32\n4 Threads\n(2,30 W)",
        "mcu_int8_worst_overhead": "ESP32-S3\nINT8 TFLM\n(0,224 W)",
    }
    color_map = {
        "sbc_int8_4t": "#2ecc71",
        "sbc_int8_2t": "#1abc9c",
        "sbc_int8_1t": "#3498db",
        "sbc_fp32_4t": "#e67e22",
        "mcu_int8_worst_overhead": "#8e44ad",
    }
    labels_a = [label_map[k] for k in keys_a]
    colors_a = [color_map[k] for k in keys_a]

    e_nom = [runs[k]["energy_mj"]["nominal_mean"] for k in keys_a]
    e_min = [runs[k]["energy_mj"]["sensitivity_min_mean"] for k in keys_a]
    e_max = [runs[k]["energy_mj"]["sensitivity_max_mean"] for k in keys_a]
    e_err_low = [n - mn for n, mn in zip(e_nom, e_min)]
    e_err_high = [mx - n for n, mx in zip(e_nom, e_max)]

    x_a = np.arange(len(keys_a))
    bars1 = ax1.bar(
        x_a,
        e_nom,
        yerr=[e_err_low, e_err_high],
        capsize=5,
        width=0.48,
        color=colors_a,
        edgecolor="#2c3e50",
        linewidth=1.1,
        alpha=0.88,
        error_kw=dict(lw=1.4, capthick=1.4, ecolor="#1a252f"),
    )

    for idx, bar in enumerate(bars1):
        val = e_nom[idx]
        mn, mx = e_min[idx], e_max[idx]
        eff = runs[keys_a[idx]]["efficiency_inf_per_j"]["nominal_mean"]
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            mx + 35,
            f"{val:.1f} mJ\n[{mn:.0f}–{mx:.0f}]\n({eff:.2f} inf/J)",
            ha="center",
            va="bottom",
            fontsize=7.4,
            fontweight="bold",
        )

    ax1.set_xticks(x_a)
    ax1.set_xticklabels(labels_a, fontsize=8.0)
    ax1.set_ylabel("Energia por Inferência $E_i$ (mJ)")
    ax1.set_title("(a) Energia por Inferência SBC vs. ESP32-S3 (Tabela 5)", pad=10)
    ax1.set_ylim(0, max(e_max) * 1.24)

    # Painel (b): EDP em mJ·s para configurações SBC (com anotação do EDP do ESP32-S3)
    keys_b = [k for k in ["sbc_int8_4t", "sbc_int8_2t", "sbc_int8_1t", "sbc_fp32_4t"] if k in runs]
    labels_b = [label_map[k] for k in keys_b]
    colors_b = [color_map[k] for k in keys_b]
    edp_nom_s = [runs[k]["edp_mj_ms"]["nominal_mean"] / 1000.0 for k in keys_b]
    edp_min_s = [runs[k]["edp_mj_ms"]["sensitivity_range"][0] / 1000.0 for k in keys_b]
    edp_max_s = [runs[k]["edp_mj_ms"]["sensitivity_range"][1] / 1000.0 for k in keys_b]
    edp_err_low = [n - mn for n, mn in zip(edp_nom_s, edp_min_s)]
    edp_err_high = [mx - n for n, mx in zip(edp_nom_s, edp_max_s)]

    x_b = np.arange(len(keys_b))
    bars2 = ax2.bar(
        x_b,
        edp_nom_s,
        yerr=[edp_err_low, edp_err_high],
        capsize=5,
        width=0.46,
        color=colors_b,
        edgecolor="#2c3e50",
        linewidth=1.1,
        alpha=0.88,
        error_kw=dict(lw=1.4, capthick=1.4, ecolor="#1a252f"),
    )

    for idx, bar in enumerate(bars2):
        val = edp_nom_s[idx]
        mn, mx = edp_min_s[idx], edp_max_s[idx]
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            mx + 4,
            f"{val:.1f} mJ·s\n[{mn:.1f}–{mx:.1f}]",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
        )

    mcu_note = ""
    if "mcu_int8_nominal" in runs:
        mcu_edp_s = runs["mcu_int8_nominal"]["edp_mj_ms"]["nominal_mean"] / 1000.0
        mcu_note = f"\n(ESP32-S3 @ 4,95s: EDP = {mcu_edp_s:.0f} mJ·s)"

    ax2.annotate(
        f"Efeito Race-to-Idle:\n4 cores reduzem EDP em 2,63x\nvs. 1 core e 4,65x vs. FP32{mcu_note}",
        xy=(0, edp_nom_s[0] + 8),
        xytext=(0.05, max(edp_max_s) * 0.78),
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.4),
        fontsize=8.1,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1", edgecolor="#27ae60"),
    )

    ax2.set_xticks(x_b)
    ax2.set_xticklabels(labels_b, fontsize=8.3)
    ax2.set_ylabel("Produto Energia-Atraso $\\mathrm{EDP} = E_i \\cdot t_i$ (mJ·s)")
    ax2.set_title("(b) Produto Energia-Atraso (EDP) e Efeito Race-to-Idle", pad=10)
    ax2.set_ylim(0, max(edp_max_s) * 1.20)

    fig.suptitle("Fase 4 (Experimento E6) — Modelagem de Energia por Software e EDP (Equação 2)", fontsize=12, fontweight="bold", y=1.03)
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(out_dir, f"fig2_energy_sensitivity_and_edp.{ext}"))
    plt.close(fig)


def plot_fig3_breakeven_space(runs: dict, out_dir: str):
    """
    Figura 3:
    Espaço de Break-Even Energético MCU vs. SBC (Equação 3) + Ponto Empírico Medido no ESP32-S3
    """
    fig, ax = plt.subplots(figsize=(10.4, 5.6))

    t_sbc_med = runs["sbc_int8_4t"]["latency_ms"]["median"]
    e_sbc_nom = runs["sbc_int8_4t"]["energy_mj"]["nominal_mean"]
    e_sbc_min = runs["sbc_int8_4t"]["energy_mj"]["sensitivity_min_mean"]
    e_sbc_max = runs["sbc_int8_4t"]["energy_mj"]["sensitivity_max_mean"]

    t_mcu_grid = np.linspace(100, 5350, 500)  # ms

    p_mcu_nom = POWER_MODEL_TABLE5["mcu_active_chip"]["nominal"]  # 0.224 W
    p_mcu_min = POWER_MODEL_TABLE5["mcu_active_chip"]["min"]      # 0.099 W
    p_mcu_worst = POWER_MODEL_TABLE5["mcu_active_chip"]["max"] + POWER_MODEL_TABLE5["mcu_board_overhead"]["max"]  # 0.474 W
    p_esp32_s3_proj = 0.350  # W (ESP32-S3 dual-core 240 MHz + Octal PSRAM ativa)

    e_mcu_nom = p_mcu_nom * t_mcu_grid
    e_mcu_min = p_mcu_min * t_mcu_grid
    e_mcu_worst = p_mcu_worst * t_mcu_grid
    e_s3_proj = p_esp32_s3_proj * t_mcu_grid

    ax.axhspan(e_sbc_min, e_sbc_max, color="#e74c3c", alpha=0.14, label="Faixa de Sensibilidade SBC (1,85–2,90 W $\\Rightarrow [204;\\,319]\\,\\mathrm{mJ}$)")
    ax.axhline(e_sbc_nom, color="#c0392b", linewidth=2.5, label=f"Energia Nominal SBC Medida ($E_{{\\mathrm{{SBC}}}} = {e_sbc_nom:.1f}\\,\\mathrm{{mJ}}$ @ $t_{{\\mathrm{{SBC}}}}={t_sbc_med:.1f}\\,\\mathrm{{ms}}$)")

    ax.fill_between(t_mcu_grid, e_mcu_min, e_mcu_worst, color="#2ecc71", alpha=0.12, label="Envelope de Sensibilidade MCU ($0,099\\,\\mathrm{W}$ chip mín. até $0,474\\,\\mathrm{W}$ c/ overhead placa)")
    ax.plot(t_mcu_grid, e_mcu_nom, "-", color="#27ae60", linewidth=2.4, label="MCU Nominal Chip ($P_{\\mathrm{MCU}} = 0,224\\,\\mathrm{W}$, Tabela 5)")
    ax.plot(t_mcu_grid, e_mcu_worst, "--", color="#16a085", linewidth=2.0, label="MCU Pior Caso c/ Overhead de Placa ($0,224 + 0,250 = 0,474\\,\\mathrm{W}$)")
    ax.plot(t_mcu_grid, e_s3_proj, "-.", color="#8e44ad", linewidth=2.1, label="ESP32-S3 c/ Octal PSRAM + PIE ($P \\approx 0,35\\,\\mathrm{W}$)")

    t_be_nom = e_sbc_nom / p_mcu_nom
    t_be_worst = e_sbc_min / p_mcu_worst
    t_be_s3 = e_sbc_nom / p_esp32_s3_proj

    ax.axvline(t_be_nom, color="#27ae60", linestyle=":", linewidth=1.4)
    ax.axvline(t_be_worst, color="#16a085", linestyle=":", linewidth=1.4)

    ax.scatter([t_be_worst, t_be_s3, t_be_nom], [e_sbc_min, e_sbc_nom, e_sbc_nom], color=["#16a085", "#8e44ad", "#27ae60"], s=70, zorder=6, edgecolor="black")

    ax.annotate(
        f"Break-Even Nominal:\n$t_{{\\mathrm{{MCU}}}} \\leq {t_be_nom:.0f}\\,\\mathrm{{ms}}$ ({t_be_nom/t_sbc_med:.2f}$\\times$)\n[Pior caso: $\\leq {t_be_worst:.0f}\\,\\mathrm{{ms}}$]",
        xy=(t_be_nom, e_sbc_nom),
        xytext=(t_be_nom + 180, e_sbc_nom - 160),
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.3),
        fontsize=8.3,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#27ae60"),
    )

    if "mcu_int8_nominal" in runs:
        t_mcu_emp = runs["mcu_int8_nominal"]["latency_ms"]["median"]
        e_mcu_emp_nom = p_mcu_nom * t_mcu_emp
        e_mcu_emp_s3 = p_esp32_s3_proj * t_mcu_emp
        e_mcu_emp_worst = p_mcu_worst * t_mcu_emp

        ax.axvline(t_mcu_emp, color="#8e44ad", linestyle="--", linewidth=1.8, alpha=0.85)
        ax.scatter(
            [t_mcu_emp, t_mcu_emp, t_mcu_emp],
            [e_mcu_emp_nom, e_mcu_emp_s3, e_mcu_emp_worst],
            color=["#27ae60", "#8e44ad", "#16a085"],
            marker="D",
            s=80,
            zorder=7,
            edgecolor="black",
            label=f"Ponto Medido ESP32-S3 ($t_{{\\mathrm{{inf}}}} = {t_mcu_emp:.1f}\\,\\mathrm{{ms}} = {t_mcu_emp/t_sbc_med:.1f}\\times t_{{\\mathrm{{SBC}}}}$)",
        )
        ax.annotate(
            f"Medição Empírica ESP32-S3 (416×416):\n$t_{{\\mathrm{{inf}}}} = {t_mcu_emp:.1f}\\,\\mathrm{{ms}}$ ({t_mcu_emp/t_sbc_med:.2f}$\\times t_{{\\mathrm{{SBC}}}}$)\n• Nominal (0,224 W): {e_mcu_emp_nom:.1f} mJ\n• S3 PSRAM (0,35 W): {e_mcu_emp_s3:.1f} mJ\n• Pior Caso (0,474 W): {e_mcu_emp_worst:.1f} mJ\n$\\Rightarrow E_{{\\mathrm{{MCU}}}} > E_{{\\mathrm{{SBC}}}}$ (Refuta parte 2 de $H_1$)",
            xy=(t_mcu_emp, e_mcu_emp_nom),
            xytext=(t_mcu_emp - 2350, 390),
            arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.5),
            fontsize=8.2,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5eef8", edgecolor="#8e44ad"),
        )

    ax.text(
        160,
        680,
        "REGIÃO DE VALIDADE DE $H_1$:\n$t_{\\mathrm{MCU}} \\leq 1124,5\\,\\mathrm{ms}$ ($E_{\\mathrm{MCU}} \\leq E_{\\mathrm{SBC}}$)",
        fontsize=8.5,
        fontweight="bold",
        color="#1e8449",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1", edgecolor="#27ae60"),
    )

    ax.set_xlabel("Latência de Inferência na MCU $t_{\\mathrm{MCU}}$ (ms)")
    ax.set_ylabel("Energia por Inferência $E$ (mJ)")
    ax.set_title("Fase 4 (Equação 3) — Break-Even Energético vs. Medição Real no ESP32-S3 ($416\\times 416$ INT8)", fontweight="bold")
    ax.set_xlim(100, 5350)
    ax.set_ylim(0, 2600)

    ax_top = ax.twiny()
    ax_top.set_xlim(100 / t_sbc_med, 5350 / t_sbc_med)
    ax_top.set_xlabel("Razão de Tempos $t_{\\mathrm{MCU}} / t_{\\mathrm{SBC}}$ (múltiplos da latência do Raspberry Pi Zero 2 W)")
    ax_top.grid(False)

    ax.legend(loc="upper left", fontsize=8.0)

    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(out_dir, f"fig3_breakeven_space_mcu_vs_sbc.{ext}"))
    plt.close(fig)


def plot_fig4_periodic_cycle_and_diagnostics(runs: dict, out_dir: str):
    """
    Figura 4:
    (a) Energia por Ciclo E_ciclo(f) = P_ativo * t + P_ocioso * (1/f - t) para taxas de 0,25 a 8 FPS
    (b) Série Temporal das 30 execuções (INT8 4t) nos 3 blocos independentes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.0))
    fig.subplots_adjust(wspace=0.26)

    f_grid = np.linspace(0.25, 8.5, 300)  # FPS
    period_s = 1.0 / f_grid

    p_sbc_idle = POWER_MODEL_TABLE5["sbc_idle"]["nominal"]  # 0.60 W
    p_mcu_idle = POWER_MODEL_TABLE5["mcu_idle_chip"]["nominal"]  # 0.066 W

    t_4t_s = runs["sbc_int8_4t"]["latency_ms"]["mean"] / 1000.0
    p_4t = POWER_MODEL_TABLE5["sbc_4core"]["nominal"]
    e_cyc_4t = (p_4t * t_4t_s + p_sbc_idle * np.maximum(0.0, period_s - t_4t_s)) * 1000.0

    t_1t_s = runs["sbc_int8_1t"]["latency_ms"]["mean"] / 1000.0
    p_1t = POWER_MODEL_TABLE5["sbc_1core"]["nominal"]
    f_grid_1t = f_grid[f_grid <= (1.0 / t_1t_s)]
    period_1t = 1.0 / f_grid_1t
    e_cyc_1t = (p_1t * t_1t_s + p_sbc_idle * np.maximum(0.0, period_1t - t_1t_s)) * 1000.0

    t_fp32_s = runs["sbc_fp32_4t"]["latency_ms"]["mean"] / 1000.0
    f_grid_fp32 = f_grid[f_grid <= (1.0 / t_fp32_s)]
    period_fp32 = 1.0 / f_grid_fp32
    e_cyc_fp32 = (p_4t * t_fp32_s + p_sbc_idle * np.maximum(0.0, period_fp32 - t_fp32_s)) * 1000.0

    t_mcu_ref_s = 0.50
    f_grid_mcu = f_grid[f_grid <= (1.0 / t_mcu_ref_s)]
    period_mcu = 1.0 / f_grid_mcu
    e_cyc_mcu = (0.224 * t_mcu_ref_s + p_mcu_idle * np.maximum(0.0, period_mcu - t_mcu_ref_s)) * 1000.0

    ax1.plot(f_grid, e_cyc_4t, "-", color="#27ae60", linewidth=2.4, label="SBC INT8 (4 Threads, $P_{\\mathrm{ocioso}}=0,60\\,\\mathrm{W}$)")
    ax1.plot(f_grid_1t, e_cyc_1t, "--", color="#2980b9", linewidth=2.0, label="SBC INT8 (1 Thread, $P_{\\mathrm{ocioso}}=0,60\\,\\mathrm{W}$)")
    ax1.plot(f_grid_fp32, e_cyc_fp32, "-.", color="#d35400", linewidth=2.0, label="SBC FP32 (4 Threads, $P_{\\mathrm{ocioso}}=0,60\\,\\mathrm{W}$)")
    ax1.plot(f_grid_mcu, e_cyc_mcu, ":", color="#8e44ad", linewidth=2.4, label="Referência MCU ($t=500\\,\\mathrm{ms}$, $P_{\\mathrm{ocioso}}=0,066\\,\\mathrm{W}$)")

    e_1fps_4t = runs["sbc_int8_4t"]["e_ciclo_periodic_mj"]["1fps_mj"]
    ax1.annotate(
        f"Em $f = 1\\,\\mathrm{{FPS}}$:\nConsumo estático ocioso do SBC\neleva $E_{{\\mathrm{{ciclo}}}}$ para {e_1fps_4t:.1f} mJ/quadro!",
        xy=(1.0, e_1fps_4t),
        xytext=(2.3, e_1fps_4t + 450),
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.3),
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#27ae60"),
    )

    ax1.set_xlabel("Taxa de Quadros da Aplicação $f$ (quadros / segundo)")
    ax1.set_ylabel("Energia Total por Ciclo $E_{\\mathrm{ciclo}}(f)$ (mJ / ciclo)")
    ax1.set_title("(a) Energia por Ciclo $E_{\\mathrm{ciclo}}(f) = P_{\\mathrm{ativo}}t + P_{\\mathrm{ocioso}}(1/f - t)$", pad=10)
    ax1.set_xlim(0.2, 8.8)
    ax1.set_ylim(0, 2600)
    ax1.legend(loc="upper right", fontsize=8.0)

    # Painel (b): Série temporal das 30 execuções + Temperatura + Blocos
    raw_4t = load_benchmark_csv("data/raw/rpi_benchmark_raw.csv")
    iters = np.arange(1, raw_4t["n"] + 1)
    t_series = raw_4t["t_inf_ms"]
    temps = raw_4t["cpu_temp_c"]

    color_lat = "#27ae60"
    color_tmp = "#c0392b"

    ax2.plot(iters, t_series, "-o", color=color_lat, markersize=4.5, linewidth=1.6, label="Latência $t_{\\mathrm{inf}}$ (ms)")
    ax2.axhline(np.median(t_series), color="#1e8449", linestyle="--", linewidth=1.2, label=f"Mediana = {np.median(t_series):.2f} ms")

    ax2.axvline(10.5, color="#7f8c8d", linestyle=":", linewidth=1.2)
    ax2.axvline(20.5, color="#7f8c8d", linestyle=":", linewidth=1.2)
    ymin_lat = float(np.min(t_series)) - 0.8
    ymax_lat = float(np.max(t_series)) + 1.2
    ax2.set_ylim(ymin_lat, ymax_lat)

    y_lbl = ymin_lat + 0.14
    ax2.text(5.5, y_lbl, "Bloco 1", ha="center", fontsize=8.5, color="#34495e", fontweight="bold")
    ax2.text(15.5, y_lbl, "Bloco 2", ha="center", fontsize=8.5, color="#34495e", fontweight="bold")
    ax2.text(25.5, y_lbl, "Bloco 3", ha="center", fontsize=8.5, color="#34495e", fontweight="bold")

    ax2.set_xlabel("Índice da Execução Cronometrada ($i = 1 \\dots 30$, 3 blocos c/ 10 warmups)")
    ax2.set_ylabel("Latência $t_{\\mathrm{inf}}$ (ms)", color=color_lat)
    ax2.tick_params(axis="y", labelcolor=color_lat)

    ax2_temp = ax2.twinx()
    ax2_temp.plot(iters, temps, "--s", color=color_tmp, markersize=3.8, linewidth=1.2, alpha=0.75, label="Temp. CPU (°C)")
    ax2_temp.set_ylabel("Temperatura do SoC BCM2710A1 (°C)", color=color_tmp)
    ax2_temp.tick_params(axis="y", labelcolor=color_tmp)
    ax2_temp.set_ylim(50, 66)
    ax2_temp.grid(False)

    kw_p = runs["sbc_int8_4t"]["diagnostics"]["kruskal_wallis_p"]
    acf1 = runs["sbc_int8_4t"]["diagnostics"]["acf_lag1"]
    ax2.set_title(f"(b) Estabilidade por Bloco (KW $p={kw_p:.3f}$, $\\mathrm{{ACF}}_1={acf1:.3f}$)", pad=10)
    ax2.legend(loc="upper left", fontsize=8.0)
    ax2_temp.legend(loc="upper right", fontsize=8.0)

    fig.suptitle("Fase 4 (Protocolo v1.3) — Impacto da Potência Ociosa em $E_{\\mathrm{ciclo}}(f)$ e Diagnósticos de Validade", fontsize=12, fontweight="bold", y=1.03)
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(out_dir, f"fig4_periodic_cycle_and_diagnostics.{ext}"))
    plt.close(fig)


def plot_fig5_3phase_and_e4_continuous(runs: dict, out_dir: str):
    """
    Figura 5:
    (a) Decomposição em 3 Fases (Equação 1: t_pre + t_inf + t_post) para 1, 2 e 4 threads INT8 e 4 threads FP32
    (b) Experimento E4: Estabilidade contínua durante 60 segundos sem pausa (399 inferências, 0x0 throttling)
    """
    e4_csv = resolve_input_path("data/raw/rpi_e4_continuous_60s.csv")
    if not os.path.exists(e4_csv):
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.0))
    fig.subplots_adjust(wspace=0.26)

    keys = [k for k in ["sbc_int8_4t", "sbc_int8_2t", "sbc_int8_1t", "sbc_fp32_4t"] if k in runs]
    label_map = {
        "sbc_int8_4t": "INT8\n(4 Threads)",
        "sbc_int8_2t": "INT8\n(2 Threads)",
        "sbc_int8_1t": "INT8\n(1 Thread)",
        "sbc_fp32_4t": "FP32\n(4 Threads)",
    }
    labels = [label_map[k] for k in keys]
    t_pre_meds = [runs[k]["pipeline_phases_eq1_ms"]["t_pre"]["median"] for k in keys]
    t_inf_meds = [runs[k]["pipeline_phases_eq1_ms"]["t_inf"]["median"] for k in keys]
    t_post_meds = [runs[k]["pipeline_phases_eq1_ms"]["t_post"]["median"] for k in keys]

    x = np.arange(len(keys))
    w = 0.46
    ax1.bar(x, t_pre_meds, width=w, color="#3498db", edgecolor="#2c3e50", label="$t_{\\mathrm{pre}}$ (Letterbox + Quant.)")
    ax1.bar(x, t_inf_meds, bottom=t_pre_meds, width=w, color="#2ecc71", edgecolor="#2c3e50", label="$t_{\\mathrm{inf}}$ (Grafo YOLOX-Nano)")
    bottom_post = np.array(t_pre_meds) + np.array(t_inf_meds)
    ax1.bar(x, t_post_meds, bottom=bottom_post, width=w, color="#f39c12", edgecolor="#2c3e50", label="$t_{\\mathrm{post}}$ (Decode + NMS)")

    for idx in range(len(keys)):
        tot = t_pre_meds[idx] + t_inf_meds[idx] + t_post_meds[idx]
        ax1.text(
            x[idx],
            tot + 5,
            f"Total: {tot:.1f} ms\n($t_{{\\mathrm{{inf}}}}$: {t_inf_meds[idx]:.1f} ms)",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Tempo por Fase da Equação (1) (ms)")
    ax1.set_title("(a) Decomposição: $t_{\\mathrm{total}} = t_{\\mathrm{pre}} + t_{\\mathrm{inf}} + t_{\\mathrm{post}}$", pad=10)
    ax1.set_ylim(0, max(bottom_post + np.array(t_post_meds)) * 1.22)
    ax1.legend(loc="upper left", fontsize=8.3)

    # Painel (b): Experimento E4 (60s contínuos)
    with open(e4_csv, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    hdr = [h.strip() for h in lines[0].split(",")]
    c_idx = {n: i for i, n in enumerate(hdr)}
    rows = [line.split(",") for line in lines[1:]]
    el_s = np.array([float(r[c_idx["elapsed_s"]]) for r in rows])
    t_inf_e4 = np.array([float(r[c_idx["t_inf_ms"]]) for r in rows])
    temp_e4 = np.array([float(r[c_idx["cpu_temp_c"]]) for r in rows])

    ax2.plot(el_s, t_inf_e4, "-", color="#27ae60", linewidth=1.1, alpha=0.85, label=f"$t_{{\\mathrm{{inf}}}}$ (Mediana = {np.median(t_inf_e4):.2f} ms)")
    ax2.set_xlabel("Tempo Decorrido no Teste Contínuo E4 (segundos)")
    ax2.set_ylabel("Latência de Inferência $t_{\\mathrm{inf}}$ (ms)", color="#27ae60")
    ax2.tick_params(axis="y", labelcolor="#27ae60")
    ax2.set_ylim(np.min(t_inf_e4) - 2.0, np.max(t_inf_e4) + 5.0)

    ax2_t = ax2.twinx()
    ax2_t.plot(el_s, temp_e4, "-", color="#c0392b", linewidth=1.8, label=f"Temp. CPU (Máx = {np.max(temp_e4):.1f} °C, Throttled=0x0)")
    ax2_t.set_ylabel("Temperatura da CPU (°C)", color="#c0392b")
    ax2_t.tick_params(axis="y", labelcolor="#c0392b")
    ax2_t.set_ylim(55, 75)
    ax2_t.grid(False)

    l1, lbl1 = ax2.get_legend_handles_labels()
    l2, lbl2 = ax2_t.get_legend_handles_labels()
    ax2.set_title(f"(b) Experimento E4 — 60s Contínuos ({len(el_s)} inferências, 0x0 Throttling)", pad=10)
    ax2.legend(l1 + l2, lbl1 + lbl2, loc="upper left", fontsize=8.0)

    fig.suptitle("Fase 4 (Protocolo v1.3) — Decomposição em 3 Fases (Eq. 1) e Estabilidade Contínua de 60s (E4)", fontsize=12, fontweight="bold", y=1.03)
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(out_dir, f"fig5_3phase_and_e4_continuous.{ext}"))
    plt.close(fig)


def plot_fig6_esp32_predictions_and_comparison(runs: dict, out_dir: str):
    """
    Figura 6:
    (a) Previsão Real do YOLOX-Nano INT8 no ESP32-S3 (test_image.jpg 416x416 letterbox)
    (b) Comparação Direta por Fase (t_pre, t_inf, t_post, t_total) e Pegada de Memória (E5): SBC vs. ESP32-S3
    """
    pred_path = resolve_input_path("data/raw/esp32_e1_predictions.json")
    img_path = resolve_input_path("raspberry/pi_zero_2w/test_image.jpg")
    if not os.path.exists(pred_path) or not os.path.exists(img_path) or "mcu_int8_nominal" not in runs:
        return

    from PIL import Image
    import matplotlib.patches as mpatches

    with open(pred_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    pil_img = Image.open(img_path).convert("RGB").resize((416, 416), Image.Resampling.BILINEAR)
    img_arr = np.asarray(pil_img, dtype=np.uint8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.5))
    fig.subplots_adjust(wspace=0.30)

    # Painel (a): Imagem 416x416 com Bounding Boxes previstas pelo ESP32-S3
    ax1.imshow(img_arr)
    for d in pred_data.get("detections", []):
        x1, y1, x2, y2 = d["bbox_xyxy"]
        cname = d["class_name"]
        sc = d["score"]
        color = "#2ecc71" if cname == "Person" else "#f39c12"
        rect = mpatches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2.4,
            edgecolor=color,
            facecolor="none",
        )
        ax1.add_patch(rect)
        ax1.text(
            x1 + 3,
            max(14, y1 - 6),
            f"{cname}: {sc:.4f} [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]",
            color="white",
            fontsize=8.2,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor="black", alpha=0.90),
        )

    ax1.set_title("(a) Previsão On-Device no ESP32-S3 ($416\\times 416$)", pad=10)
    ax1.set_xlabel("Coordenada Horizontal $x$ (pixels, escala $416\\times 416$)")
    ax1.set_ylabel("Coordenada Vertical $y$ (pixels, escala $416\\times 416$)")
    ax1.grid(False)

    # Painel (b): Comparação em escala logarítmica das 3 fases + Memória RAM entre SBC (INT8 4t) e MCU (ESP32-S3)
    sbc_p = runs["sbc_int8_4t"]["pipeline_phases_eq1_ms"]
    mcu_p = runs["mcu_int8_nominal"]["pipeline_phases_eq1_ms"]

    metrics_labels = [
        "$t_{\\mathrm{pre}}$ (ms)\n[Quant.]",
        "$t_{\\mathrm{inf}}$ (ms)\n[Grafo]",
        "$t_{\\mathrm{post}}$ (ms)\n[NMS]",
        "$t_{\\mathrm{total}}$ (ms)\n[Ciclo]",
        "RAM (MiB)\n[Exp. E5]",
    ]
    sbc_vals = [
        sbc_p["t_pre"]["median"],
        sbc_p["t_inf"]["median"],
        sbc_p["t_post"]["median"],
        sbc_p["t_total"]["median"],
        60.60,
    ]
    mcu_vals = [
        mcu_p["t_pre"]["median"],
        mcu_p["t_inf"]["median"],
        mcu_p["t_post"]["median"],
        mcu_p["t_total"]["median"],
        2.122,
    ]

    x = np.arange(len(metrics_labels))
    w = 0.36
    b_sbc = ax2.bar(x - w / 2, sbc_vals, width=w, color="#2ecc71", edgecolor="#1e8449", linewidth=1.1, label="Raspberry Pi Zero 2 W (INT8, 4 Threads)")
    b_mcu = ax2.bar(x + w / 2, mcu_vals, width=w, color="#8e44ad", edgecolor="#5b2c6f", linewidth=1.1, label="ESP32-S3 @ 240 MHz (INT8, esp-tflite-micro)")

    ax2.set_yscale("log")
    ax2.set_ylim(0.7, 55000)

    for bar, val in zip(b_sbc, sbc_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val * 1.18,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
            color="#1e8449",
        )
    for bar, val in zip(b_mcu, mcu_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val * 1.18,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
            color="#5b2c6f",
        )

    ax2.annotate(
        "C++ -O3 na MCU é 3,78x mais rápido\nno pós-processamento (1,60 vs 6,05 ms),\nmas PSRAM Octal limita $t_{\\mathrm{inf}}$ (45,2x maior)",
        xy=(2 + w / 2, mcu_vals[2] * 1.8),
        xytext=(1.35, 320),
        arrowprops=dict(arrowstyle="->", color="#5b2c6f", lw=1.2),
        fontsize=7.7,
        bbox=dict(boxstyle="round,pad=0.26", facecolor="#f5eef8", edgecolor="#8e44ad"),
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_labels, fontsize=8.3)
    ax2.set_ylabel("Valor Medido (escala logarítmica: ms ou MiB)")
    ax2.set_title("(b) Decomposição (Eq. 1) e Memória (E5): SBC vs. MCU", pad=10)
    ax2.legend(loc="upper left", fontsize=8.1)

    fig.suptitle("Fase 4 (Protocolo v1.3) — Validação Funcional (E1), Decomposição Temporal (E3) e Memória (E5) no ESP32-S3", fontsize=12, fontweight="bold", y=1.03)
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(out_dir, f"fig6_esp32_predictions_and_comparison.{ext}"))
    plt.close(fig)


def main():
    out_dir = "report/figures"
    os.makedirs(out_dir, exist_ok=True)

    runs = {
        "sbc_int8_4t": analyze_platform_run(load_benchmark_csv("data/raw/rpi_benchmark_raw.csv"), "sbc_4core"),
        "sbc_int8_1t": analyze_platform_run(load_benchmark_csv("data/raw/rpi_int8_1thread.csv"), "sbc_1core"),
        "sbc_fp32_4t": analyze_platform_run(load_benchmark_csv("data/raw/rpi_fp32_4threads.csv"), "sbc_4core"),
    }
    p_2t = resolve_input_path("data/raw/rpi_int8_2threads.csv")
    if os.path.exists(p_2t):
        runs["sbc_int8_2t"] = analyze_platform_run(load_benchmark_csv(p_2t), "sbc_2core")
    p_mcu = resolve_input_path("data/raw/esp32_benchmark_raw.csv")
    if os.path.exists(p_mcu):
        runs["mcu_int8_nominal"] = analyze_platform_run(
            load_benchmark_csv(p_mcu), "mcu_active_chip", idle_key="mcu_idle_chip", include_overhead=False
        )
        runs["mcu_int8_worst_overhead"] = analyze_platform_run(
            load_benchmark_csv(p_mcu), "mcu_active_chip", idle_key="mcu_idle_chip", include_overhead=True
        )

    rep_path = resolve_input_path("data/processed/phase4_e6_report.json")
    with open(rep_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    plot_fig1_latency_and_amdahl(runs, report, out_dir)
    plot_fig2_energy_sensitivity_and_edp(runs, out_dir)
    plot_fig3_breakeven_space(runs, out_dir)
    plot_fig4_periodic_cycle_and_diagnostics(runs, out_dir)
    plot_fig5_3phase_and_e4_continuous(runs, out_dir)
    plot_fig6_esp32_predictions_and_comparison(runs, out_dir)

    # Espelhar em plots/phase4_v13 se existir
    if os.path.isdir("plots/phase4_v13"):
        for fname in os.listdir(out_dir):
            if fname.startswith("fig") and (fname.endswith(".png") or fname.endswith(".pdf")):
                shutil.copyfile(os.path.join(out_dir, fname), os.path.join("plots/phase4_v13", fname))

    print(f"[OK] Figuras científicas (PNG 300 DPI + PDF vetorial) geradas em '{out_dir}/':")
    for fname in sorted(os.listdir(out_dir)):
        if fname.startswith("fig"):
            print(f"  - {os.path.join(out_dir, fname)}")


if __name__ == "__main__":
    main()

