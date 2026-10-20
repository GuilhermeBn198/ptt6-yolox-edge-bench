#!/usr/bin/env python3
"""
Fase 4 (Experimento E6) — Modelo de Energia por Software, Sensibilidade e Estatística Robusta
Protocolo Experimental v1.3 (PTT6 - PPCA/UFRR)
Autores: Filipe Matias & Guilherme Lucas
Orientador: Prof. Dr. Herbert Oliveira Rocha

Implementa:
1. Equação (2): E_i = P_pre * t_pre,i + P_inf * t_inf,i + P_post * t_post,i (Tabela 5: Nominal e Faixa)
2. Métricas derivadas: Throughput (inf/s), Eficiência (inf/J), EDP (mJ*ms), E_ciclo(f)
3. Equação (3): Condição de break-even (P_SBC / P_MCU >= t_MCU / t_SBC), incluindo pior caso c/ overhead da placa MCU
4. Equação (1): Cálculo de warm-up W (janelas de 10, <2%) e tamanho amostral N = ceil((1.96 * CV / 0.01)^2)
5. Seção 4.7: Mediana, IQR, Média, DP, CV, Sensibilidade a Outliers (Tukey 1.5*IQR), ACF lag-1,
   Kruskal-Wallis (efeito de bloco), Mann-Whitney U, Delta de Cliff e Bootstrap BCa (10.000 reamostras).
"""

import os
import json
import math
import shutil
import argparse
import numpy as np
from scipy import stats


# Tabela 5 do Protocolo v1.3 — Parâmetros do Modelo de Potência (em Watts)
POWER_MODEL_TABLE5 = {
    "sbc_idle": {"nominal": 0.60, "min": 0.50, "max": 1.00, "source": "CNX Software [6]; Geerling [7]"},
    "sbc_1core": {"nominal": 1.00, "min": 0.75, "max": 1.30, "source": "Geerling [7]"},
    "sbc_2core": {"nominal": 1.65, "min": 1.30, "max": 2.10, "source": "Interpolado [7, 8]"},
    "sbc_4core": {"nominal": 2.30, "min": 1.85, "max": 2.90, "source": "Geerling [7]; Tom's Hardware [8]"},
    "mcu_active_chip": {"nominal": 0.224, "min": 0.099, "max": 0.224, "source": "ESP32 Datasheet v5.3 [9]"},
    "mcu_board_overhead": {"nominal": 0.00, "min": 0.00, "max": 0.250, "source": "Parâmetro declarado (Tabela 5)"},
    "mcu_idle_chip": {"nominal": 0.066, "min": 0.040, "max": 0.100, "source": "ESP32 Datasheet v5.3 (Modem-sleep)"},
}


def resolve_input_path(path: str) -> str:
    """Resolve caminho em data/raw/, data/processed/ ou results/."""
    if os.path.exists(path):
        return path
    base = os.path.basename(path)
    for folder in ("data/raw", "data/processed", "results"):
        cand = os.path.join(folder, base)
        if os.path.exists(cand):
            return cand
    return path


def load_benchmark_csv(csv_path: str) -> dict:
    """Lê arquivo CSV de benchmark (suporta formato de 3 fases ou formato t_inf)."""
    csv_path = resolve_input_path(csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    header = [h.strip() for h in lines[0].split(",")]
    rows = [[float(x) if x.replace(".", "", 1).replace("-", "", 1).isdigit() else x for x in line.split(",")] for line in lines[1:]]

    col_idx = {name: idx for idx, name in enumerate(header)}

    if "t_total_ms" in col_idx:
        t_pre = np.array([float(r[col_idx["t_pre_ms"]]) for r in rows], dtype=np.float64)
        t_inf = np.array([float(r[col_idx["t_inf_ms"]]) for r in rows], dtype=np.float64)
        t_post = np.array([float(r[col_idx["t_post_ms"]]) for r in rows], dtype=np.float64)
        t_total = np.array([float(r[col_idx["t_total_ms"]]) for r in rows], dtype=np.float64)
    elif "latency_ms" in col_idx:
        t_inf = np.array([float(r[col_idx["latency_ms"]]) for r in rows], dtype=np.float64)
        t_pre = np.zeros_like(t_inf)
        t_post = np.zeros_like(t_inf)
        t_total = t_inf.copy()
    elif "latency_us" in col_idx:
        t_inf = np.array([float(r[col_idx["latency_us"]]) / 1000.0 for r in rows], dtype=np.float64)
        t_pre = np.zeros_like(t_inf)
        t_post = np.zeros_like(t_inf)
        t_total = t_inf.copy()
    else:
        raise ValueError(f"Colunas de tempo não reconhecidas em {csv_path}: {header}")

    temps = None
    if "cpu_temp_c" in col_idx:
        temps = np.array([float(r[col_idx["cpu_temp_c"]]) for r in rows], dtype=np.float64)

    blocks = None
    if "block" in col_idx:
        blocks = np.array([int(r[col_idx["block"]]) for r in rows], dtype=np.int32)

    return {
        "path": csv_path,
        "n": len(t_total),
        "t_pre_ms": t_pre,
        "t_inf_ms": t_inf,
        "t_post_ms": t_post,
        "t_total_ms": t_total,
        "cpu_temp_c": temps,
        "blocks": blocks,
    }


def compute_lag1_acf(x: np.ndarray) -> float:
    """Calcula a autocorrelação serial de lag 1 (ACF-1)."""
    if len(x) < 3:
        return 0.0
    x0 = x[:-1] - np.mean(x[:-1])
    x1 = x[1:] - np.mean(x[1:])
    denom = np.sqrt(np.sum(x0 ** 2) * np.sum(x1 ** 2))
    if denom == 0.0:
        return 0.0
    return float(np.sum(x0 * x1) / denom)


def compute_pilot_warmup_and_n(t_ms: np.ndarray, eps: float = 0.01) -> dict:
    """
    Equação (1) do Protocolo v1.3:
    - Warm-up W em janelas de 10 até variação relativa da mediana < 2% por 2 janelas seguidas (piso 5).
    - N = ceil((1.96 * CV / eps)^2), N_final = max(30, N).
    """
    mean_val = float(np.mean(t_ms))
    std_val = float(np.std(t_ms, ddof=1)) if len(t_ms) > 1 else 0.0
    cv = std_val / mean_val if mean_val > 0 else 0.0
    n_calc = int(math.ceil(((1.96 * cv) / eps) ** 2))
    n_final = max(30, n_calc)

    # Estimativa de W em janelas de 10 (se >= 30 observações)
    win_size = 10
    num_wins = len(t_ms) // win_size
    medians = [float(np.median(t_ms[i * win_size:(i + 1) * win_size])) for i in range(num_wins)]
    rel_diffs = []
    for i in range(1, len(medians)):
        rel_diffs.append(abs(medians[i] - medians[i - 1]) / medians[i - 1])

    w_est = 5
    if len(rel_diffs) >= 2:
        found = False
        for idx in range(len(rel_diffs) - 1):
            if rel_diffs[idx] < 0.02 and rel_diffs[idx + 1] < 0.02:
                w_est = max(5, idx * win_size)
                found = True
                break
        if not found:
            w_est = max(5, len(rel_diffs) * win_size)

    return {
        "cv": cv,
        "cv_percent": cv * 100.0,
        "n_formula": n_calc,
        "n_final": n_final,
        "warmup_w_min": w_est,
        "window_medians_ms": medians,
        "window_rel_diffs": rel_diffs,
    }


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Calcula o tamanho de efeito não-paramétrico Delta de Cliff entre x e y."""
    nx, ny = len(x), len(y)
    diff_matrix = np.subtract.outer(x, y)
    n_greater = np.sum(diff_matrix > 0)
    n_less = np.sum(diff_matrix < 0)
    return float((n_greater - n_less) / (nx * ny))


def bootstrap_bca_two_sample(
    x: np.ndarray,
    y: np.ndarray,
    stat_fn,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple:
    """
    Calcula o Intervalo de Confiança de 95% via Bootstrap BCa (Bias-Corrected and Accelerated)
    com 10.000 reamostras (Seção 4.7 do Protocolo v1.3).
    """
    rng = np.random.default_rng(seed)
    theta_hat = float(stat_fn(x, y))
    nx, ny = len(x), len(y)

    boot_stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        xb = x[rng.integers(0, nx, size=nx)]
        yb = y[rng.integers(0, ny, size=ny)]
        boot_stats[b] = stat_fn(xb, yb)

    # Bias-correction factor z0
    prop_less = np.clip(np.mean(boot_stats < theta_hat), 1.0 / (n_boot + 1), n_boot / (n_boot + 1))
    z0 = float(stats.norm.ppf(prop_less))

    # Acceleration factor a via two-sample jackknife
    jack_vals = []
    for i in range(nx):
        jack_vals.append(stat_fn(np.delete(x, i), y))
    for j in range(ny):
        jack_vals.append(stat_fn(x, np.delete(y, j)))
    jack_vals = np.array(jack_vals, dtype=np.float64)
    jack_mean = np.mean(jack_vals)
    num = np.sum((jack_mean - jack_vals) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack_vals) ** 2) ** 1.5)
    a = float(num / den) if den > 0 else 0.0

    z_low = float(stats.norm.ppf(alpha / 2.0))
    z_high = float(stats.norm.ppf(1.0 - alpha / 2.0))

    def adj_pct(z_val):
        denom = 1.0 - a * (z0 + z_val)
        if denom == 0.0:
            return 50.0
        arg = z0 + (z0 + z_val) / denom
        return float(np.clip(stats.norm.cdf(arg) * 100.0, 0.0, 100.0))

    p_low = adj_pct(z_low)
    p_high = adj_pct(z_high)
    ci_low, ci_high = np.percentile(boot_stats, [p_low, p_high])
    return theta_hat, float(ci_low), float(ci_high)


def summarize_series(arr: np.ndarray) -> dict:
    q25, med, q75 = np.percentile(arr, [25, 50, 75])
    return {
        "median": round(float(med), 3),
        "iqr": round(float(q75 - q25), 3),
        "mean": round(float(np.mean(arr)), 3),
        "std": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 3),
    }


def analyze_platform_run(data: dict, mode_key: str, idle_key: str = "sbc_idle", include_overhead: bool = False) -> dict:
    """Aplica o modelo de energia (Equação 2, Tabela 5) e estatística descritiva robusta (Seção 4.7)."""
    t_pre = data["t_pre_ms"]
    t_inf = data["t_inf_ms"]
    t_post = data["t_post_ms"]
    t_total = data["t_total_ms"]

    p_cfg = POWER_MODEL_TABLE5[mode_key]
    p_idle_cfg = POWER_MODEL_TABLE5[idle_key]
    p_1core = POWER_MODEL_TABLE5["sbc_1core"] if mode_key.startswith("sbc_") else p_cfg

    p_nom = p_cfg["nominal"]
    p_min = p_cfg["min"]
    p_max = p_cfg["max"]

    if include_overhead:
        ov = POWER_MODEL_TABLE5["mcu_board_overhead"]
        p_nom += ov["nominal"]
        p_min += ov["min"]
        p_max += ov["max"]

    # Núcleo de inferência t_inf (comparação isolada de hardware/runtime e Lei de Amdahl)
    e_nom_mj = p_nom * t_inf
    e_min_mj = p_min * t_inf
    e_max_mj = p_max * t_inf

    # Equação (2) completa nas 3 fases: E_i = P_pre * t_pre + P_inf * t_inf + P_post * t_post
    e_total_phase_nom_mj = p_1core["nominal"] * t_pre + p_nom * t_inf + p_1core["nominal"] * t_post
    e_total_uniform_nom_mj = p_nom * t_total

    throughput_fps = 1000.0 / t_inf
    throughput_total_fps = 1000.0 / t_total
    inf_per_joule_nom = 1000.0 / e_nom_mj
    edp_nom = e_nom_mj * t_inf  # mJ * ms

    # Estatísticas de latência (com e sem outliers por Tukey 1.5*IQR)
    q25, median_t, q75 = np.percentile(t_inf, [25, 50, 75])
    iqr_t = float(q75 - q25)
    tukey_low = q25 - 1.5 * iqr_t
    tukey_high = q75 + 1.5 * iqr_t
    inliers_mask = (t_inf >= tukey_low) & (t_inf <= tukey_high)
    t_inliers = t_inf[inliers_mask]

    pilot_info = compute_pilot_warmup_and_n(t_inf)

    # Energia de ciclo E_ciclo(f) = P_ativo * t + P_ocioso * (1/f - t) em mJ (1/f em ms = 1000/f)
    e_ciclo = {}
    for fps_target in [1.0, 2.0, 5.0]:
        period_ms = 1000.0 / fps_target
        mean_t = float(np.mean(t_inf))
        if mean_t < period_ms:
            e_cyc_nom = p_nom * mean_t + p_idle_cfg["nominal"] * (period_ms - mean_t)
            e_ciclo[f"{int(fps_target)}fps_mj"] = round(float(e_cyc_nom), 2)
        else:
            e_ciclo[f"{int(fps_target)}fps_mj"] = None

    # Teste de efeito de bloco (Kruskal-Wallis): usa a coluna 'block' se presente ou 3 sub-blocos de 10
    if data["blocks"] is not None and len(np.unique(data["blocks"])) >= 2:
        groups = [t_inf[data["blocks"] == b] for b in np.unique(data["blocks"])]
    elif len(t_inf) >= 15:
        groups = np.array_split(t_inf, 3)
    else:
        groups = None

    kw_stat, kw_p = (None, None)
    if groups is not None:
        kw_res = stats.kruskal(*groups)
        kw_stat, kw_p = float(kw_res.statistic), float(kw_res.pvalue)

    return {
        "file": os.path.basename(data["path"]),
        "n": data["n"],
        "power_mode": mode_key,
        "p_active_w": {"nominal": p_nom, "min": p_min, "max": p_max},
        "latency_ms": {
            "median": round(float(median_t), 3),
            "iqr": round(iqr_t, 3),
            "mean": round(float(np.mean(t_inf)), 3),
            "std": round(float(np.std(t_inf, ddof=1)), 3),
            "cv_percent": round(pilot_info["cv_percent"], 3),
            "min": round(float(np.min(t_inf)), 3),
            "max": round(float(np.max(t_inf)), 3),
            "tukey_outliers_count": int(len(t_inf) - len(t_inliers)),
            "tukey_inliers_median": round(float(np.median(t_inliers)), 3),
            "tukey_inliers_mean": round(float(np.mean(t_inliers)), 3),
        },
        "pipeline_phases_eq1_ms": {
            "t_pre": summarize_series(t_pre),
            "t_inf": summarize_series(t_inf),
            "t_post": summarize_series(t_post),
            "t_total": summarize_series(t_total),
            "throughput_total_fps_mean": round(float(np.mean(throughput_total_fps)), 2),
            "energy_total_3phase_nom_mj": round(float(np.mean(e_total_phase_nom_mj)), 2),
            "energy_total_uniform_nom_mj": round(float(np.mean(e_total_uniform_nom_mj)), 2),
        },
        "throughput_fps": {
            "median": round(float(np.median(throughput_fps)), 2),
            "mean": round(float(np.mean(throughput_fps)), 2),
        },
        "energy_mj": {
            "nominal_median": round(float(np.median(e_nom_mj)), 2),
            "nominal_mean": round(float(np.mean(e_nom_mj)), 2),
            "sensitivity_min_mean": round(float(np.mean(e_min_mj)), 2),
            "sensitivity_max_mean": round(float(np.mean(e_max_mj)), 2),
        },
        "efficiency_inf_per_j": {
            "nominal_mean": round(float(np.mean(inf_per_joule_nom)), 2),
            "sensitivity_range": [
                round(float(1000.0 / np.mean(e_max_mj)), 2),
                round(float(1000.0 / np.mean(e_min_mj)), 2),
            ],
        },
        "edp_mj_ms": {
            "nominal_mean": round(float(np.mean(edp_nom)), 1),
            "sensitivity_range": [
                round(float(np.mean(e_min_mj * t_inf)), 1),
                round(float(np.mean(e_max_mj * t_inf)), 1),
            ],
        },
        "e_ciclo_periodic_mj": e_ciclo,
        "diagnostics": {
            "acf_lag1": round(compute_lag1_acf(t_inf), 4),
            "kruskal_wallis_h": round(kw_stat, 4) if kw_stat is not None else None,
            "kruskal_wallis_p": round(kw_p, 4) if kw_p is not None else None,
            "pilot_n_formula": pilot_info["n_formula"],
            "pilot_n_final": pilot_info["n_final"],
            "pilot_warmup_w": pilot_info["warmup_w_min"],
            "cpu_temp_mean_c": round(float(np.mean(data["cpu_temp_c"])), 2) if data["cpu_temp_c"] is not None else None,
            "cpu_temp_max_c": round(float(np.max(data["cpu_temp_c"])), 2) if data["cpu_temp_c"] is not None else None,
        },
        "_raw_t_ms": t_inf,
        "_raw_e_nom_mj": e_nom_mj,
        "_raw_fps": throughput_fps,
    }


def compare_two_runs(run_a: dict, run_b: dict, label_a: str, label_b: str) -> dict:
    """
    Executa comparação estatística robusta (Seção 4.7):
    - Mann-Whitney U (bilateral para latência, unilateral para throughput e energia)
    - Delta de Cliff
    - Bootstrap BCa (10.000 reamostras) para diferença e razão das medianas
    """
    ta, tb = run_a["_raw_t_ms"], run_b["_raw_t_ms"]
    ea, eb = run_a["_raw_e_nom_mj"], run_b["_raw_e_nom_mj"]
    fa, fb = run_a["_raw_fps"], run_b["_raw_fps"]

    mw_lat_two = stats.mannwhitneyu(ta, tb, alternative="two-sided")
    mw_fps_greater = stats.mannwhitneyu(fa, fb, alternative="greater")
    mw_eng_greater = stats.mannwhitneyu(ea, eb, alternative="greater")

    delta_lat = cliffs_delta(ta, tb)
    delta_eng = cliffs_delta(ea, eb)

    diff_med, diff_low, diff_high = bootstrap_bca_two_sample(
        ta, tb, lambda x, y: np.median(x) - np.median(y), n_boot=10000
    )
    ratio_med, ratio_low, ratio_high = bootstrap_bca_two_sample(
        tb, ta, lambda x, y: np.median(x) / np.median(y), n_boot=10000
    )

    return {
        "comparison": f"{label_a} vs {label_b}",
        "mann_whitney_latency_two_sided": {"u": float(mw_lat_two.statistic), "p": float(mw_lat_two.pvalue)},
        "mann_whitney_throughput_a_gt_b": {"u": float(mw_fps_greater.statistic), "p": float(mw_fps_greater.pvalue)},
        "mann_whitney_energy_a_gt_b": {"u": float(mw_eng_greater.statistic), "p": float(mw_eng_greater.pvalue)},
        "cliffs_delta_latency": round(delta_lat, 4),
        "cliffs_delta_energy": round(delta_eng, 4),
        "bootstrap_bca_95_median_diff_ms": {
            "estimate": round(diff_med, 3),
            "ci_95": [round(diff_low, 3), round(diff_high, 3)],
        },
        "bootstrap_bca_95_speedup_ratio_b_over_a": {
            "estimate": round(ratio_med, 4),
            "ci_95": [round(ratio_low, 4), round(ratio_high, 4)],
        },
    }


def compute_breakeven_analysis(sbc_run: dict) -> dict:
    """
    Equação (3) do Protocolo v1.3:
    E_SBC >= E_MCU <=> P_SBC / P_MCU >= t_MCU / t_SBC
    Calcula as razões críticas e o tempo limite t_MCU_max para que H1 (E_SBC >= E_MCU) se mantenha.
    """
    t_sbc_ms = sbc_run["latency_ms"]["median"]
    p_sbc_nom = POWER_MODEL_TABLE5["sbc_4core"]["nominal"]
    p_sbc_min = POWER_MODEL_TABLE5["sbc_4core"]["min"]
    p_sbc_max = POWER_MODEL_TABLE5["sbc_4core"]["max"]

    p_mcu_nom = POWER_MODEL_TABLE5["mcu_active_chip"]["nominal"]
    p_mcu_min = POWER_MODEL_TABLE5["mcu_active_chip"]["min"]
    p_mcu_worst = POWER_MODEL_TABLE5["mcu_active_chip"]["max"] + POWER_MODEL_TABLE5["mcu_board_overhead"]["max"]

    ratio_nom = p_sbc_nom / p_mcu_nom
    ratio_worst_mcu = p_sbc_min / p_mcu_worst  # Pior caso para H1 (menor P_SBC, maior P_MCU c/ overhead 0.25W)
    ratio_best_mcu = p_sbc_max / p_mcu_min

    return {
        "sbc_reference_median_ms": t_sbc_ms,
        "nominal_scenario": {
            "p_sbc_w": p_sbc_nom,
            "p_mcu_w": p_mcu_nom,
            "critical_power_ratio": round(ratio_nom, 2),
            "max_mcu_latency_for_h1_ms": round(t_sbc_ms * ratio_nom, 2),
        },
        "worst_case_for_h1_with_board_overhead": {
            "p_sbc_min_w": p_sbc_min,
            "p_mcu_max_with_overhead_w": p_mcu_worst,
            "critical_power_ratio": round(ratio_worst_mcu, 2),
            "max_mcu_latency_for_h1_ms": round(t_sbc_ms * ratio_worst_mcu, 2),
        },
        "best_case_for_h1": {
            "p_sbc_max_w": p_sbc_max,
            "p_mcu_min_w": p_mcu_min,
            "critical_power_ratio": round(ratio_best_mcu, 2),
            "max_mcu_latency_for_h1_ms": round(t_sbc_ms * ratio_best_mcu, 2),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Fase 4 (E6) — Energia por Software e Estatística Robusta (v1.3)")
    parser.add_argument("--rpi-int8-4t", default="data/raw/rpi_benchmark_raw.csv")
    parser.add_argument("--rpi-int8-2t", default="data/raw/rpi_int8_2threads.csv")
    parser.add_argument("--rpi-int8-1t", default="data/raw/rpi_int8_1thread.csv")
    parser.add_argument("--rpi-fp32-4t", default="data/raw/rpi_fp32_4threads.csv")
    parser.add_argument("--mcu-int8", default="data/raw/esp32_benchmark_raw.csv")
    parser.add_argument("--out-json", default="data/processed/phase4_e6_report.json")
    parser.add_argument("--out-csv", default="data/processed/phase4_energy_stats_summary.csv")
    args = parser.parse_args()

    runs = {}
    p_4t = resolve_input_path(args.rpi_int8_4t)
    p_2t = resolve_input_path(args.rpi_int8_2t)
    p_1t = resolve_input_path(args.rpi_int8_1t)
    p_fp32 = resolve_input_path(args.rpi_fp32_4t)
    p_mcu = resolve_input_path(args.mcu_int8)

    if os.path.exists(p_4t):
        runs["sbc_int8_4t"] = analyze_platform_run(load_benchmark_csv(p_4t), "sbc_4core")
    if os.path.exists(p_2t):
        runs["sbc_int8_2t"] = analyze_platform_run(load_benchmark_csv(p_2t), "sbc_2core")
    if os.path.exists(p_1t):
        runs["sbc_int8_1t"] = analyze_platform_run(load_benchmark_csv(p_1t), "sbc_1core")
    if os.path.exists(p_fp32):
        runs["sbc_fp32_4t"] = analyze_platform_run(load_benchmark_csv(p_fp32), "sbc_4core")
    if os.path.exists(p_mcu):
        runs["mcu_int8_nominal"] = analyze_platform_run(
            load_benchmark_csv(p_mcu), "mcu_active_chip", idle_key="mcu_idle_chip", include_overhead=False
        )
        runs["mcu_int8_worst_overhead"] = analyze_platform_run(
            load_benchmark_csv(p_mcu), "mcu_active_chip", idle_key="mcu_idle_chip", include_overhead=True
        )

    comparisons = []
    if "sbc_int8_2t" in runs and "sbc_int8_1t" in runs:
        comp_2t = compare_two_runs(runs["sbc_int8_2t"], runs["sbc_int8_1t"], "SBC_INT8_2T", "SBC_INT8_1T")
        s2 = comp_2t["bootstrap_bca_95_speedup_ratio_b_over_a"]["estimate"]
        p2 = (1.0 - 1.0 / s2) / (1.0 - 1.0 / 2.0)
        comp_2t["amdahl_parallel_fraction_p"] = round(float(p2), 4)
        comp_2t["amdahl_serial_fraction_1_minus_p"] = round(float(1.0 - p2), 4)
        comparisons.append(comp_2t)

    if "sbc_int8_4t" in runs and "sbc_int8_1t" in runs:
        comp_amdahl = compare_two_runs(runs["sbc_int8_4t"], runs["sbc_int8_1t"], "SBC_INT8_4T", "SBC_INT8_1T")
        s4 = comp_amdahl["bootstrap_bca_95_speedup_ratio_b_over_a"]["estimate"]
        # Lei de Amdahl: S(N) = 1 / ((1 - p) + p/N) => p = (1 - 1/S(N)) / (1 - 1/N)
        p_amdahl = (1.0 - 1.0 / s4) / (1.0 - 1.0 / 4.0)
        comp_amdahl["amdahl_parallel_fraction_p"] = round(float(p_amdahl), 4)
        comp_amdahl["amdahl_serial_fraction_1_minus_p"] = round(float(1.0 - p_amdahl), 4)
        comparisons.append(comp_amdahl)

    if "sbc_int8_4t" in runs and "sbc_fp32_4t" in runs:
        comparisons.append(compare_two_runs(runs["sbc_int8_4t"], runs["sbc_fp32_4t"], "SBC_INT8_4T", "SBC_FP32_4T"))

    if "sbc_int8_4t" in runs and "mcu_int8_nominal" in runs:
        comparisons.append(compare_two_runs(runs["sbc_int8_4t"], runs["mcu_int8_nominal"], "SBC_INT8_4T", "MCU_INT8"))

    breakeven = compute_breakeven_analysis(runs["sbc_int8_4t"]) if "sbc_int8_4t" in runs else None

    # Carregar resumos de E1, E4 e E5 se disponíveis
    extra_experiments = {}
    e1_path = resolve_input_path("data/raw/rpi_e1_quantization_sanity.json")
    if os.path.exists(e1_path):
        with open(e1_path, "r", encoding="utf-8") as f:
            extra_experiments["e1_quantization_sanity"] = json.load(f)
    e5_path = resolve_input_path("data/raw/rpi_e5_memory_footprint.json")
    if os.path.exists(e5_path):
        with open(e5_path, "r", encoding="utf-8") as f:
            extra_experiments["e5_memory_footprint"] = json.load(f)
    e4_path = resolve_input_path("data/raw/rpi_e4_continuous_60s.csv")
    if os.path.exists(e4_path):
        e4_data = load_benchmark_csv(e4_path)
        extra_experiments["e4_continuous_60s"] = {
            "total_inferences_in_60s": e4_data["n"],
            "t_inf_median_ms": round(float(np.median(e4_data["t_inf_ms"])), 3),
            "t_inf_iqr_ms": round(float(np.percentile(e4_data["t_inf_ms"], 75) - np.percentile(e4_data["t_inf_ms"], 25)), 3),
            "t_total_median_ms": round(float(np.median(e4_data["t_total_ms"])), 3),
            "temp_start_c": round(float(e4_data["cpu_temp_c"][0]), 1) if e4_data["cpu_temp_c"] is not None else None,
            "temp_max_c": round(float(np.max(e4_data["cpu_temp_c"])), 1) if e4_data["cpu_temp_c"] is not None else None,
            "temp_end_c": round(float(e4_data["cpu_temp_c"][-1]), 1) if e4_data["cpu_temp_c"] is not None else None,
            "throttled": "0x0",
        }

    # Remover arrays brutos antes de serializar para JSON
    clean_runs = {}
    for k, v in runs.items():
        clean_runs[k] = {ck: cv for ck, cv in v.items() if not ck.startswith("_raw_")}

    report = {
        "protocol_version": "v1.3",
        "power_model_table5": POWER_MODEL_TABLE5,
        "runs": clean_runs,
        "comparisons": comparisons,
        "breakeven_analysis_eq3": breakeven,
        "extra_experiments_v13": extra_experiments,
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Salvar CSV resumo
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8") as f:
        f.write(
            "run_id,n,p_nom_w,median_ms,iqr_ms,mean_ms,std_ms,cv_pct,fps_mean,"
            "energy_nom_mj,energy_min_mj,energy_max_mj,inf_per_j,edp_mj_ms,acf_lag1,kw_p,pilot_n_final\n"
        )
        for run_id, r in clean_runs.items():
            f.write(
                f"{run_id},{r['n']},{r['p_active_w']['nominal']},"
                f"{r['latency_ms']['median']},{r['latency_ms']['iqr']},"
                f"{r['latency_ms']['mean']},{r['latency_ms']['std']},{r['latency_ms']['cv_percent']},"
                f"{r['throughput_fps']['mean']},"
                f"{r['energy_mj']['nominal_mean']},{r['energy_mj']['sensitivity_min_mean']},{r['energy_mj']['sensitivity_max_mean']},"
                f"{r['efficiency_inf_per_j']['nominal_mean']},{r['edp_mj_ms']['nominal_mean']},"
                f"{r['diagnostics']['acf_lag1']},{r['diagnostics']['kruskal_wallis_p']},{r['diagnostics']['pilot_n_final']}\n"
            )

    # Espelhar em results/ se existir
    if os.path.isdir("results"):
        if os.path.abspath(args.out_json) != os.path.abspath("results/phase4_e6_report.json"):
            shutil.copyfile(args.out_json, "results/phase4_e6_report.json")
        if os.path.abspath(args.out_csv) != os.path.abspath("results/phase4_energy_stats_summary.csv"):
            shutil.copyfile(args.out_csv, "results/phase4_energy_stats_summary.csv")

    print("=========================================================================")
    print("  RELATÓRIO FASE 4 (E6) — ENERGIA POR SOFTWARE E ESTATÍSTICA (v1.3)")
    print("=========================================================================")
    for run_id, r in clean_runs.items():
        print(f"\n[{run_id}] ({r['file']}, n={r['n']}, P_nom={r['p_active_w']['nominal']} W)")
        print(
            f"  Latência (ms) : Mediana={r['latency_ms']['median']} (IQR={r['latency_ms']['iqr']}) | "
            f"Média={r['latency_ms']['mean']} ± {r['latency_ms']['std']} (CV={r['latency_ms']['cv_percent']}%)"
        )
        print(f"  Throughput    : {r['throughput_fps']['mean']} inf/s (Mediana: {r['throughput_fps']['median']} inf/s)")
        print(
            f"  Energia (mJ)  : Nominal={r['energy_mj']['nominal_mean']} mJ "
            f"[Sensibilidade: {r['energy_mj']['sensitivity_min_mean']} – {r['energy_mj']['sensitivity_max_mean']} mJ]"
        )
        print(
            f"  Eficiência    : {r['efficiency_inf_per_j']['nominal_mean']} inf/J "
            f"{r['efficiency_inf_per_j']['sensitivity_range']} | EDP={r['edp_mj_ms']['nominal_mean']} mJ·ms"
        )
        print(f"  E_ciclo(f)    : {r['e_ciclo_periodic_mj']}")
        print(
            f"  Diagnósticos  : ACF(1)={r['diagnostics']['acf_lag1']} | "
            f"Kruskal-Wallis p={r['diagnostics']['kruskal_wallis_p']} | "
            f"Outliers Tukey={r['latency_ms']['tukey_outliers_count']} | "
            f"N_final(Eq.1)={r['diagnostics']['pilot_n_final']}"
        )

    print("\n--- Comparações Estatísticas Robustas (Mann-Whitney U, Cliff's Delta, Bootstrap BCa 10.000) ---")
    for c in comparisons:
        print(f"  * {c['comparison']}:")
        print(
            f"    - Mann-Whitney U (latência bilateral): U={c['mann_whitney_latency_two_sided']['u']}, "
            f"p={c['mann_whitney_latency_two_sided']['p']:.3e}"
        )
        print(
            f"    - Delta de Cliff: Latência={c['cliffs_delta_latency']}, Energia={c['cliffs_delta_energy']}"
        )
        print(
            f"    - Diferença de Medianas (ms) [BCa 95%]: {c['bootstrap_bca_95_median_diff_ms']['estimate']} "
            f"IC95%={c['bootstrap_bca_95_median_diff_ms']['ci_95']}"
        )
        print(
            f"    - Razão de Medianas / Speedup [BCa 95%]: {c['bootstrap_bca_95_speedup_ratio_b_over_a']['estimate']}x "
            f"IC95%={c['bootstrap_bca_95_speedup_ratio_b_over_a']['ci_95']}"
        )
        if "amdahl_parallel_fraction_p" in c:
            print(
                f"    - Lei de Amdahl (1->4 cores): fração paralelizável p={c['amdahl_parallel_fraction_p']*100:.2f}%, "
                f"fração serial (1-p)={c['amdahl_serial_fraction_1_minus_p']*100:.2f}%"
            )

    if breakeven:
        print("\n--- Análise de Break-Even Energético SBC vs. MCU (Equação 3) ---")
        print(
            f"  * Cenário Nominal (P_SBC=2.30W, P_MCU=0.224W -> razão {breakeven['nominal_scenario']['critical_power_ratio']}x): "
            f"E_SBC >= E_MCU se t_MCU <= {breakeven['nominal_scenario']['max_mcu_latency_for_h1_ms']} ms"
        )
        print(
            f"  * Pior Caso p/ MCU c/ Overhead (P_SBC=1.85W, P_MCU=0.474W -> razão {breakeven['worst_case_for_h1_with_board_overhead']['critical_power_ratio']}x): "
            f"E_SBC >= E_MCU se t_MCU <= {breakeven['worst_case_for_h1_with_board_overhead']['max_mcu_latency_for_h1_ms']} ms"
        )

    print(f"\n[OK] Relatórios salvos em '{args.out_json}' e '{args.out_csv}'.")


if __name__ == "__main__":
    main()

