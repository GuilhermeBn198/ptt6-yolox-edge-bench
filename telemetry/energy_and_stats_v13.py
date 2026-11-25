#!/usr/bin/env python3
"""
Fase 4 (Experimento E5 de main.tex / E6 de experimentacao.md) — Modelo de Energia por Software,
Sensibilidade, Break-Even, Energia por Ciclo e Estatística Robusta (H1 e H2)
Protocolo Experimental v1.3–v1.5 (PTT6 - PPCA/UFRR)
Autores: Filipe Matias & Guilherme Lucas
Orientador: Prof. Dr. Herbert Oliveira Rocha

Implementa integralmente todas as métricas de docs/projeto/main.tex:
1. Equação (2): E_i = P_pre * t_pre,i + P_inf * t_inf,i + P_post * t_post,i (Tabela 5: Nominal e Faixa)
   - Análise Principal na janela completa (t_total = t_pre + t_inf + t_post e E_3phase)
   - Análise Secundária isolando o núcleo de inferência (t_inf e E_inf, Seção 4.12 de main.tex)
2. Métricas derivadas: Throughput (inf/s), Eficiência (inf/J), EDP (mJ*ms), E_ciclo(f) para f in {0.2, 0.5, 1, 2, 5} Hz
3. Equação (3): Condição de break-even (P_SBC / P_MCU >= t_MCU / t_SBC), incluindo pior caso c/ overhead da placa MCU
4. Equação (1): Cálculo de warm-up W (janelas de 10, <2% por 2 janelas seguidas, piso 5) e N = ceil((1.96 * CV / 0.01)^2)
5. Seção 4.12: Mediana, IQR, Média, DP, CV, Sensibilidade a Outliers (Tukey 1.5*IQR), ACF lag-1,
   Kruskal-Wallis (efeito de bloco), Mann-Whitney U, Delta de Cliff, Bootstrap BCa (10.000 reamostras),
   Teste de Interseção-União (IUT) para H1 (nominal e extremos) e Avaliação conjunta de H2 (acurácia rho + latência + energia).
"""

import os
import json
import math
import shutil
import argparse
import numpy as np
from scipy import stats


# Tabela 5 do Protocolo / Tabela 5 de main.tex — Parâmetros do Modelo de Potência (em Watts)
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
    Equação (1) de docs/projeto/main.tex:
    - Warm-up W em janelas de 10 até variação relativa da mediana < 2% por 2 janelas seguidas (piso 5).
    - N = ceil((1.96 * CV / eps)^2), N_final = max(30, N).
    """
    mean_val = float(np.mean(t_ms))
    std_val = float(np.std(t_ms, ddof=1)) if len(t_ms) > 1 else 0.0
    cv = std_val / mean_val if mean_val > 0 else 0.0
    n_calc = int(math.ceil(((1.96 * cv) / eps) ** 2))
    n_final = max(30, n_calc)

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
    com 10.000 reamostras (Seção 4.12 de docs/projeto/main.tex).
    """
    rng = np.random.default_rng(seed)
    theta_hat = float(stat_fn(x, y))
    nx, ny = len(x), len(y)

    boot_stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        xb = x[rng.integers(0, nx, size=nx)]
        yb = y[rng.integers(0, ny, size=ny)]
        boot_stats[b] = stat_fn(xb, yb)

    prop_less = np.clip(np.mean(boot_stats < theta_hat), 1.0 / (n_boot + 1), n_boot / (n_boot + 1))
    z0 = float(stats.norm.ppf(prop_less))

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


def summarize_series_full(arr: np.ndarray) -> dict:
    q25, med, q75 = np.percentile(arr, [25, 50, 75])
    iqr = float(q75 - q25)
    t_low = q25 - 1.5 * iqr
    t_high = q75 + 1.5 * iqr
    inliers = arr[(arr >= t_low) & (arr <= t_high)]
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv_pct = (std_v / mean_v * 100.0) if mean_v > 0 else 0.0
    return {
        "median": round(float(med), 3),
        "iqr": round(iqr, 3),
        "mean": round(mean_v, 3),
        "std": round(std_v, 3),
        "cv_percent": round(cv_pct, 3),
        "min": round(float(np.min(arr)), 3),
        "max": round(float(np.max(arr)), 3),
        "tukey_outliers_count": int(len(arr) - len(inliers)),
        "tukey_inliers_median": round(float(np.median(inliers)), 3),
        "tukey_inliers_mean": round(float(np.mean(inliers)), 3),
    }


def analyze_platform_run(data: dict, mode_key: str, idle_key: str = "sbc_idle", include_overhead: bool = False) -> dict:
    """
    Aplica o modelo de energia (Equação 2, Tabela 5 de main.tex) e estatística descritiva robusta:
      - Janela Completa (3 Fases): E_3phase = P_pre * t_pre + P_inf * t_inf + P_post * t_post
      - Janela Secundária (Núcleo de Inferência): E_inf = P_inf * t_inf
    """
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

    p_pre_nom, p_pre_min, p_pre_max = p_1core["nominal"], p_1core["min"], p_1core["max"]

    if include_overhead:
        ov = POWER_MODEL_TABLE5["mcu_board_overhead"]
        p_nom += ov["nominal"]
        p_min += ov["min"]
        p_max += ov["max"]
        p_pre_nom += ov["nominal"]
        p_pre_min += ov["min"]
        p_pre_max += ov["max"]

    # 1. Análise Secundária (isolamento de linguagem/runtime: apenas t_inf)
    e_nom_mj = p_nom * t_inf
    e_min_mj = p_min * t_inf
    e_max_mj = p_max * t_inf

    # 2. Análise Principal na Janela Completa de 3 Fases (Equação 2 de main.tex):
    #    E_i = P_pre * t_pre,i + P_inf * t_inf,i + P_post * t_post,i
    e_3phase_nom_mj = p_pre_nom * t_pre + p_nom * t_inf + p_pre_nom * t_post
    e_3phase_min_mj = p_pre_min * t_pre + p_min * t_inf + p_pre_min * t_post
    e_3phase_max_mj = p_pre_max * t_pre + p_max * t_inf + p_pre_max * t_post
    e_total_uniform_nom_mj = p_nom * t_total

    throughput_fps = 1000.0 / t_inf
    throughput_total_fps = 1000.0 / t_total

    inf_per_joule_nom = 1000.0 / e_nom_mj
    inf_per_joule_3phase_nom = 1000.0 / e_3phase_nom_mj

    edp_nom = e_nom_mj * t_inf  # mJ * ms (apenas t_inf)
    edp_3phase_nom = e_3phase_nom_mj * t_total  # mJ * ms (janela completa 3 fases)

    pilot_inf = compute_pilot_warmup_and_n(t_inf)
    pilot_tot = compute_pilot_warmup_and_n(t_total)

    # Energia de ciclo E_ciclo(f) = P_ativo * t + P_ocioso * (1/f - t) em mJ
    # Frequências de main.tex (linha 306): f in {0.2, 0.5, 1} Hz + {2, 5} Hz
    e_ciclo = {}
    e_ciclo_3phase = {}
    mean_t_inf = float(np.mean(t_inf))
    mean_t_tot = float(np.mean(t_total))
    mean_e_3phase = float(np.mean(e_3phase_nom_mj))

    for f_hz, key_lbl in [(0.2, "0.2hz_mj"), (0.5, "0.5hz_mj"), (1.0, "1fps_mj"), (2.0, "2fps_mj"), (5.0, "5fps_mj")]:
        period_ms = 1000.0 / f_hz
        if mean_t_inf < period_ms:
            e_cyc_inf = p_nom * mean_t_inf + p_idle_cfg["nominal"] * (period_ms - mean_t_inf)
            e_ciclo[key_lbl] = round(float(e_cyc_inf), 2)
        else:
            e_ciclo[key_lbl] = None

        if mean_t_tot < period_ms:
            e_cyc_tot = mean_e_3phase + p_idle_cfg["nominal"] * (period_ms - mean_t_tot)
            e_ciclo_3phase[key_lbl] = round(float(e_cyc_tot), 2)
        else:
            e_ciclo_3phase[key_lbl] = None

    # Teste de efeito de bloco (Kruskal-Wallis) para t_inf e t_total
    if data["blocks"] is not None and len(np.unique(data["blocks"])) >= 2:
        groups_inf = [t_inf[data["blocks"] == b] for b in np.unique(data["blocks"])]
        groups_tot = [t_total[data["blocks"] == b] for b in np.unique(data["blocks"])]
    elif len(t_inf) >= 15:
        groups_inf = np.array_split(t_inf, 3)
        groups_tot = np.array_split(t_total, 3)
    else:
        groups_inf, groups_tot = None, None

    kw_stat, kw_p = (None, None)
    kw_tot_stat, kw_tot_p = (None, None)
    if groups_inf is not None:
        kw_res = stats.kruskal(*groups_inf)
        kw_stat, kw_p = float(kw_res.statistic), float(kw_res.pvalue)
        kw_tot_res = stats.kruskal(*groups_tot)
        kw_tot_stat, kw_tot_p = float(kw_tot_res.statistic), float(kw_tot_res.pvalue)

    return {
        "file": os.path.basename(data["path"]),
        "n": data["n"],
        "power_mode": mode_key,
        "p_active_w": {"nominal": p_nom, "min": p_min, "max": p_max},
        "latency_ms": summarize_series_full(t_inf),
        "latency_total_3phase_ms": summarize_series_full(t_total),
        "pipeline_phases_eq1_ms": {
            "t_pre": summarize_series_full(t_pre),
            "t_inf": summarize_series_full(t_inf),
            "t_post": summarize_series_full(t_post),
            "t_total": summarize_series_full(t_total),
            "throughput_total_fps_median": round(float(np.median(throughput_total_fps)), 3),
            "throughput_total_fps_mean": round(float(np.mean(throughput_total_fps)), 3),
            "energy_total_3phase_nom_median_mj": round(float(np.median(e_3phase_nom_mj)), 2),
            "energy_total_3phase_nom_mj": round(float(np.mean(e_3phase_nom_mj)), 2),
            "energy_total_3phase_min_mj": round(float(np.mean(e_3phase_min_mj)), 2),
            "energy_total_3phase_max_mj": round(float(np.mean(e_3phase_max_mj)), 2),
            "energy_total_uniform_nom_mj": round(float(np.mean(e_total_uniform_nom_mj)), 2),
            "efficiency_3phase_inf_per_j_nom": round(float(np.mean(inf_per_joule_3phase_nom)), 2),
            "efficiency_3phase_range": [
                round(float(1000.0 / np.mean(e_3phase_max_mj)), 2),
                round(float(1000.0 / np.mean(e_3phase_min_mj)), 2),
            ],
            "edp_3phase_nom_mj_ms": round(float(np.mean(edp_3phase_nom)), 1),
            "edp_3phase_range_mj_ms": [
                round(float(np.mean(e_3phase_min_mj * t_total)), 1),
                round(float(np.mean(e_3phase_max_mj * t_total)), 1),
            ],
        },
        "throughput_fps": {
            "median": round(float(np.median(throughput_fps)), 3),
            "mean": round(float(np.mean(throughput_fps)), 3),
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
        "e_ciclo_3phase_periodic_mj": e_ciclo_3phase,
        "diagnostics": {
            "acf_lag1": round(compute_lag1_acf(t_inf), 4),
            "acf_lag1_t_total": round(compute_lag1_acf(t_total), 4),
            "kruskal_wallis_h": round(kw_stat, 4) if kw_stat is not None else None,
            "kruskal_wallis_p": round(kw_p, 4) if kw_p is not None else None,
            "kruskal_wallis_t_total_h": round(kw_tot_stat, 4) if kw_tot_stat is not None else None,
            "kruskal_wallis_t_total_p": round(kw_tot_p, 4) if kw_tot_p is not None else None,
            "pilot_n_formula": pilot_inf["n_formula"],
            "pilot_n_final": pilot_inf["n_final"],
            "pilot_warmup_w": pilot_inf["warmup_w_min"],
            "pilot_t_total_n_formula": pilot_tot["n_formula"],
            "pilot_t_total_n_final": pilot_tot["n_final"],
            "pilot_t_total_warmup_w": pilot_tot["warmup_w_min"],
            "cpu_temp_mean_c": round(float(np.mean(data["cpu_temp_c"])), 2) if data["cpu_temp_c"] is not None else None,
            "cpu_temp_max_c": round(float(np.max(data["cpu_temp_c"])), 2) if data["cpu_temp_c"] is not None else None,
        },
        "_raw_t_ms": t_inf,
        "_raw_t_total_ms": t_total,
        "_raw_e_nom_mj": e_nom_mj,
        "_raw_e_min_mj": e_min_mj,
        "_raw_e_max_mj": e_max_mj,
        "_raw_e_3phase_nom_mj": e_3phase_nom_mj,
        "_raw_e_3phase_min_mj": e_3phase_min_mj,
        "_raw_e_3phase_max_mj": e_3phase_max_mj,
        "_raw_fps": throughput_fps,
        "_raw_total_fps": throughput_total_fps,
    }


def compare_two_runs(run_a: dict, run_b: dict, label_a: str, label_b: str) -> dict:
    """
    Executa comparação estatística robusta (Seção 4.12 de main.tex):
    - Mann-Whitney U (bilateral para latência, unilateral para throughput e energia)
    - Delta de Cliff
    - Bootstrap BCa (10.000 reamostras) para diferença e razão das medianas
    Reporta tanto para t_inf (núcleo isolado) quanto para t_total (janela completa de 3 fases).
    """
    ta, tb = run_a["_raw_t_ms"], run_b["_raw_t_ms"]
    ea, eb = run_a["_raw_e_nom_mj"], run_b["_raw_e_nom_mj"]
    fa, fb = run_a["_raw_fps"], run_b["_raw_fps"]

    tta, ttb = run_a["_raw_t_total_ms"], run_b["_raw_t_total_ms"]
    e3a, e3b = run_a["_raw_e_3phase_nom_mj"], run_b["_raw_e_3phase_nom_mj"]
    fta, ftb = run_a["_raw_total_fps"], run_b["_raw_total_fps"]

    mw_lat_two = stats.mannwhitneyu(ta, tb, alternative="two-sided")
    mw_fps_greater = stats.mannwhitneyu(fa, fb, alternative="greater")
    mw_eng_greater = stats.mannwhitneyu(ea, eb, alternative="greater")

    mw_tot_lat_two = stats.mannwhitneyu(tta, ttb, alternative="two-sided")
    mw_tot_fps_greater = stats.mannwhitneyu(fta, ftb, alternative="greater")
    mw_tot_eng_greater = stats.mannwhitneyu(e3a, e3b, alternative="greater")

    delta_lat = cliffs_delta(ta, tb)
    delta_eng = cliffs_delta(ea, eb)
    delta_tot_lat = cliffs_delta(tta, ttb)
    delta_tot_eng = cliffs_delta(e3a, e3b)

    diff_med, diff_low, diff_high = bootstrap_bca_two_sample(
        ta, tb, lambda x, y: np.median(x) - np.median(y), n_boot=10000
    )
    ratio_med, ratio_low, ratio_high = bootstrap_bca_two_sample(
        tb, ta, lambda x, y: np.median(x) / np.median(y), n_boot=10000
    )

    diff_tot_med, diff_tot_low, diff_tot_high = bootstrap_bca_two_sample(
        tta, ttb, lambda x, y: np.median(x) - np.median(y), n_boot=10000
    )
    ratio_tot_med, ratio_tot_low, ratio_tot_high = bootstrap_bca_two_sample(
        ttb, tta, lambda x, y: np.median(x) / np.median(y), n_boot=10000
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
        "full_window_3phase": {
            "mann_whitney_t_total_two_sided": {"u": float(mw_tot_lat_two.statistic), "p": float(mw_tot_lat_two.pvalue)},
            "mann_whitney_throughput_total_a_gt_b": {"u": float(mw_tot_fps_greater.statistic), "p": float(mw_tot_fps_greater.pvalue)},
            "mann_whitney_energy_3phase_a_gt_b": {"u": float(mw_tot_eng_greater.statistic), "p": float(mw_tot_eng_greater.pvalue)},
            "cliffs_delta_t_total": round(delta_tot_lat, 4),
            "cliffs_delta_energy_3phase": round(delta_tot_eng, 4),
            "bootstrap_bca_95_t_total_median_diff_ms": {
                "estimate": round(diff_tot_med, 3),
                "ci_95": [round(diff_tot_low, 3), round(diff_tot_high, 3)],
            },
            "bootstrap_bca_95_t_total_speedup_ratio_b_over_a": {
                "estimate": round(ratio_tot_med, 4),
                "ci_95": [round(ratio_tot_low, 4), round(ratio_tot_high, 4)],
            },
        },
    }


def evaluate_h1_intersection_union(sbc_run: dict, mcu_nom_run: dict, mcu_worst_run: dict) -> dict:
    """
    Teste de Interseção-União (IUT) para H1 (Seção 4.12 de docs/projeto/main.tex):
      Parte 1: throughput_SBC > throughput_MCU (a alpha = 0,05)
      Parte 2: E_SBC >= E_MCU (a alpha = 0,05), avaliada no valor nominal e nos extremos da Tabela 5.
    Avalia tanto na janela principal de 3 fases (t_total, E_3phase) quanto na análise secundária (t_inf, E_inf).
    """
    def eval_window(fps_sbc, fps_mcu, e_sbc_nom, e_sbc_min, e_sbc_max, e_mcu_nom, e_mcu_min, e_mcu_worst):
        mw_fps = stats.mannwhitneyu(fps_sbc, fps_mcu, alternative="greater")
        mw_e_nom = stats.mannwhitneyu(e_sbc_nom, e_mcu_nom, alternative="greater")
        mw_e_worst_h1 = stats.mannwhitneyu(e_sbc_min, e_mcu_worst, alternative="greater")
        mw_e_best_h1 = stats.mannwhitneyu(e_sbc_max, e_mcu_min, alternative="greater")

        p1_ok = float(mw_fps.pvalue) < 0.05 and float(np.median(fps_sbc)) > float(np.median(fps_mcu))
        p2_nom_ok = float(mw_e_nom.pvalue) < 0.05 and float(np.median(e_sbc_nom)) >= float(np.median(e_mcu_nom))
        p2_worst_ok = float(mw_e_worst_h1.pvalue) < 0.05 and float(np.median(e_sbc_min)) >= float(np.median(e_mcu_worst))
        p2_best_ok = float(mw_e_best_h1.pvalue) < 0.05 and float(np.median(e_sbc_max)) >= float(np.median(e_mcu_min))

        return {
            "part1_throughput_sbc_gt_mcu": {
                "sbc_median_fps": round(float(np.median(fps_sbc)), 3),
                "mcu_median_fps": round(float(np.median(fps_mcu)), 3),
                "mann_whitney_u": float(mw_fps.statistic),
                "p_value": float(mw_fps.pvalue),
                "cliffs_delta": round(cliffs_delta(fps_sbc, fps_mcu), 4),
                "confirmed_alpha_005": p1_ok,
            },
            "part2_energy_sbc_ge_mcu": {
                "nominal": {
                    "sbc_median_mj": round(float(np.median(e_sbc_nom)), 2),
                    "mcu_median_mj": round(float(np.median(e_mcu_nom)), 2),
                    "mann_whitney_u": float(mw_e_nom.statistic),
                    "p_value": float(mw_e_nom.pvalue),
                    "cliffs_delta": round(cliffs_delta(e_sbc_nom, e_mcu_nom), 4),
                    "confirmed_alpha_005": p2_nom_ok,
                },
                "worst_case_for_h1_sbc_min_vs_mcu_max_overhead": {
                    "sbc_min_median_mj": round(float(np.median(e_sbc_min)), 2),
                    "mcu_max_overhead_median_mj": round(float(np.median(e_mcu_worst)), 2),
                    "mann_whitney_u": float(mw_e_worst_h1.statistic),
                    "p_value": float(mw_e_worst_h1.pvalue),
                    "cliffs_delta": round(cliffs_delta(e_sbc_min, e_mcu_worst), 4),
                    "confirmed_alpha_005": p2_worst_ok,
                },
                "best_case_for_h1_sbc_max_vs_mcu_min": {
                    "sbc_max_median_mj": round(float(np.median(e_sbc_max)), 2),
                    "mcu_min_median_mj": round(float(np.median(e_mcu_min)), 2),
                    "mann_whitney_u": float(mw_e_best_h1.statistic),
                    "p_value": float(mw_e_best_h1.pvalue),
                    "cliffs_delta": round(cliffs_delta(e_sbc_max, e_mcu_min), 4),
                    "confirmed_alpha_005": p2_best_ok,
                },
            },
            "iut_h1_confirmed_nominal": bool(p1_ok and p2_nom_ok),
            "iut_h1_robust_across_all_extremes": bool(p1_ok and p2_nom_ok and p2_worst_ok and p2_best_ok),
        }

    return {
        "primary_full_window_3phase": eval_window(
            sbc_run["_raw_total_fps"], mcu_nom_run["_raw_total_fps"],
            sbc_run["_raw_e_3phase_nom_mj"], sbc_run["_raw_e_3phase_min_mj"], sbc_run["_raw_e_3phase_max_mj"],
            mcu_nom_run["_raw_e_3phase_nom_mj"], mcu_nom_run["_raw_e_3phase_min_mj"], mcu_worst_run["_raw_e_3phase_max_mj"],
        ),
        "secondary_inference_only_t_inf": eval_window(
            sbc_run["_raw_fps"], mcu_nom_run["_raw_fps"],
            sbc_run["_raw_e_nom_mj"], sbc_run["_raw_e_min_mj"], sbc_run["_raw_e_max_mj"],
            mcu_nom_run["_raw_e_nom_mj"], mcu_nom_run["_raw_e_min_mj"], mcu_worst_run["_raw_e_max_mj"],
        ),
    }


def evaluate_h2_tradeoff_sbc(sbc_int8: dict, sbc_fp32: dict, e0_data: dict | None) -> dict:
    """
    Avaliação Formal de H2 (Seções 1.1, 4.3 e 4.12 de docs/projeto/main.tex):
    No SBC, a quantização INT8 reduz latência e energia por inferência em relação ao modelo FP32,
    ao custo de uma perda de acurácia quantificada pela retenção rho = mAP_int8 / mAP_fp32 (com IC 95%).
    """
    t_int8_tot, t_fp32_tot = sbc_int8["_raw_t_total_ms"], sbc_fp32["_raw_t_total_ms"]
    e_int8_3p, e_fp32_3p = sbc_int8["_raw_e_3phase_nom_mj"], sbc_fp32["_raw_e_3phase_nom_mj"]

    t_int8_inf, t_fp32_inf = sbc_int8["_raw_t_ms"], sbc_fp32["_raw_t_ms"]
    e_int8_inf, e_fp32_inf = sbc_int8["_raw_e_nom_mj"], sbc_fp32["_raw_e_nom_mj"]

    mw_t_tot_less = stats.mannwhitneyu(t_int8_tot, t_fp32_tot, alternative="less")
    mw_e_3p_less = stats.mannwhitneyu(e_int8_3p, e_fp32_3p, alternative="less")
    mw_t_inf_less = stats.mannwhitneyu(t_int8_inf, t_fp32_inf, alternative="less")
    mw_e_inf_less = stats.mannwhitneyu(e_int8_inf, e_fp32_inf, alternative="less")

    sp_tot, sp_tot_lo, sp_tot_hi = bootstrap_bca_two_sample(
        t_fp32_tot, t_int8_tot, lambda x, y: np.median(x) / np.median(y), n_boot=10000
    )
    er_3p, er_3p_lo, er_3p_hi = bootstrap_bca_two_sample(
        e_fp32_3p, e_int8_3p, lambda x, y: np.median(x) / np.median(y), n_boot=10000
    )
    sp_inf, sp_inf_lo, sp_inf_hi = bootstrap_bca_two_sample(
        t_fp32_inf, t_int8_inf, lambda x, y: np.median(x) / np.median(y), n_boot=10000
    )

    acc_block = None
    if e0_data and "bootstrap_ci_95_summary" in e0_data:
        acc_block = e0_data["bootstrap_ci_95_summary"]

    return {
        "hypothesis": "H2: No SBC, INT8 reduz latência e energia vs. FP32, ao custo de retenção rho = mAP_int8 / mAP_fp32",
        "primary_full_window_3phase": {
            "int8_t_total_median_ms": sbc_int8["latency_total_3phase_ms"]["median"],
            "fp32_t_total_median_ms": sbc_fp32["latency_total_3phase_ms"]["median"],
            "speedup_t_total_bca_95": {"estimate": round(sp_tot, 4), "ci_95": [round(sp_tot_lo, 4), round(sp_tot_hi, 4)]},
            "mann_whitney_t_total_int8_lt_fp32_p": float(mw_t_tot_less.pvalue),
            "int8_energy_3phase_median_mj": sbc_int8["pipeline_phases_eq1_ms"]["energy_total_3phase_nom_median_mj"],
            "fp32_energy_3phase_median_mj": sbc_fp32["pipeline_phases_eq1_ms"]["energy_total_3phase_nom_median_mj"],
            "energy_reduction_ratio_3phase_bca_95": {"estimate": round(er_3p, 4), "ci_95": [round(er_3p_lo, 4), round(er_3p_hi, 4)]},
            "mann_whitney_energy_3phase_int8_lt_fp32_p": float(mw_e_3p_less.pvalue),
            "cliffs_delta_energy_3phase": round(cliffs_delta(e_int8_3p, e_fp32_3p), 4),
        },
        "secondary_inference_only_t_inf": {
            "int8_t_inf_median_ms": sbc_int8["latency_ms"]["median"],
            "fp32_t_inf_median_ms": sbc_fp32["latency_ms"]["median"],
            "speedup_t_inf_bca_95": {"estimate": round(sp_inf, 4), "ci_95": [round(sp_inf_lo, 4), round(sp_inf_hi, 4)]},
            "mann_whitney_t_inf_int8_lt_fp32_p": float(mw_t_inf_less.pvalue),
            "int8_energy_inf_median_mj": sbc_int8["energy_mj"]["nominal_median"],
            "fp32_energy_inf_median_mj": sbc_fp32["energy_mj"]["nominal_median"],
            "mann_whitney_energy_inf_int8_lt_fp32_p": float(mw_e_inf_less.pvalue),
        },
        "accuracy_retention_e0_test_bgr": acc_block,
    }


def compute_breakeven_analysis(sbc_run: dict) -> dict:
    """
    Equação (3) de docs/projeto/main.tex:
    E_SBC >= E_MCU <=> P_SBC / P_MCU >= t_MCU / t_SBC
    Calcula as razões críticas e o tempo limite t_MCU_max tanto para t_inf quanto para a janela de 3 fases.
    """
    t_sbc_ms = sbc_run["latency_ms"]["median"]
    t_sbc_tot_ms = sbc_run["latency_total_3phase_ms"]["median"]
    e_sbc_3p_nom = sbc_run["pipeline_phases_eq1_ms"]["energy_total_3phase_nom_mj"]
    e_sbc_3p_min = sbc_run["pipeline_phases_eq1_ms"]["energy_total_3phase_min_mj"]
    e_sbc_3p_max = sbc_run["pipeline_phases_eq1_ms"]["energy_total_3phase_max_mj"]

    p_sbc_nom = POWER_MODEL_TABLE5["sbc_4core"]["nominal"]
    p_sbc_min = POWER_MODEL_TABLE5["sbc_4core"]["min"]
    p_sbc_max = POWER_MODEL_TABLE5["sbc_4core"]["max"]

    p_mcu_nom = POWER_MODEL_TABLE5["mcu_active_chip"]["nominal"]
    p_mcu_min = POWER_MODEL_TABLE5["mcu_active_chip"]["min"]
    p_mcu_worst = POWER_MODEL_TABLE5["mcu_active_chip"]["max"] + POWER_MODEL_TABLE5["mcu_board_overhead"]["max"]

    ratio_nom = p_sbc_nom / p_mcu_nom
    ratio_worst_mcu = p_sbc_min / p_mcu_worst
    ratio_best_mcu = p_sbc_max / p_mcu_min

    return {
        "sbc_reference_median_ms": t_sbc_ms,
        "sbc_reference_t_total_median_ms": t_sbc_tot_ms,
        "nominal_scenario": {
            "p_sbc_w": p_sbc_nom,
            "p_mcu_w": p_mcu_nom,
            "critical_power_ratio": round(ratio_nom, 2),
            "max_mcu_latency_for_h1_ms": round(t_sbc_ms * ratio_nom, 2),
            "max_mcu_t_total_3phase_for_h1_ms": round(e_sbc_3p_nom / p_mcu_nom, 2),
        },
        "worst_case_for_h1_with_board_overhead": {
            "p_sbc_min_w": p_sbc_min,
            "p_mcu_max_with_overhead_w": p_mcu_worst,
            "critical_power_ratio": round(ratio_worst_mcu, 2),
            "max_mcu_latency_for_h1_ms": round(t_sbc_ms * ratio_worst_mcu, 2),
            "max_mcu_t_total_3phase_for_h1_ms": round(e_sbc_3p_min / p_mcu_worst, 2),
        },
        "best_case_for_h1": {
            "p_sbc_max_w": p_sbc_max,
            "p_mcu_min_w": p_mcu_min,
            "critical_power_ratio": round(ratio_best_mcu, 2),
            "max_mcu_latency_for_h1_ms": round(t_sbc_ms * ratio_best_mcu, 2),
            "max_mcu_t_total_3phase_for_h1_ms": round(e_sbc_3p_max / p_mcu_min, 2),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Fase 4 (E5/E6) — Energia por Software e Estatística Robusta (v1.5 / main.tex)")
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
        p_amdahl = (1.0 - 1.0 / s4) / (1.0 - 1.0 / 4.0)
        comp_amdahl["amdahl_parallel_fraction_p"] = round(float(p_amdahl), 4)
        comp_amdahl["amdahl_serial_fraction_1_minus_p"] = round(float(1.0 - p_amdahl), 4)
        comparisons.append(comp_amdahl)

    if "sbc_int8_4t" in runs and "sbc_fp32_4t" in runs:
        comparisons.append(compare_two_runs(runs["sbc_int8_4t"], runs["sbc_fp32_4t"], "SBC_INT8_4T", "SBC_FP32_4T"))

    if "sbc_int8_4t" in runs and "mcu_int8_nominal" in runs:
        comparisons.append(compare_two_runs(runs["sbc_int8_4t"], runs["mcu_int8_nominal"], "SBC_INT8_4T", "MCU_INT8"))

    breakeven = compute_breakeven_analysis(runs["sbc_int8_4t"]) if "sbc_int8_4t" in runs else None

    # Carregar resumos de E0, E1, E2, E4 e E5
    extra_experiments = {}
    for key, fname in [
        ("e0_map_test", "data/processed/e0_map_test.json"),
        ("e1_equivalence_50_bgr", "data/processed/e1_equivalence_report.json"),
        ("e1_quantization_sanity", "data/raw/rpi_e1_quantization_sanity.json"),
        ("sbc_e2_pilot_100", "data/raw/sbc_e2_pilot_summary.json"),
        ("mcu_e2_pilot_100", "data/raw/mcu_e2_pilot_summary.json"),
        ("e5_memory_footprint", "data/raw/rpi_e5_memory_footprint.json"),
        ("esp32_e1_predictions", "data/raw/esp32_e1_predictions.json"),
        ("esp32_e5_memory_footprint", "data/raw/esp32_e5_memory_footprint.json"),
    ]:
        p = resolve_input_path(fname)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                extra_experiments[key] = json.load(f)

    e4_path = resolve_input_path("data/raw/rpi_e4_continuous_60s.csv")
    if os.path.exists(e4_path):
        e4_data = load_benchmark_csv(e4_path)
        extra_experiments["e4_continuous_60s"] = {
            "total_inferences_in_60s": e4_data["n"],
            "throughput_sustained_inf_s": round(e4_data["n"] / 60.0, 3),
            "t_inf_median_ms": round(float(np.median(e4_data["t_inf_ms"])), 3),
            "t_inf_iqr_ms": round(float(np.percentile(e4_data["t_inf_ms"], 75) - np.percentile(e4_data["t_inf_ms"], 25)), 3),
            "t_total_median_ms": round(float(np.median(e4_data["t_total_ms"])), 3),
            "temp_start_c": round(float(e4_data["cpu_temp_c"][0]), 1) if e4_data["cpu_temp_c"] is not None else None,
            "temp_max_c": round(float(np.max(e4_data["cpu_temp_c"])), 1) if e4_data["cpu_temp_c"] is not None else None,
            "temp_end_c": round(float(e4_data["cpu_temp_c"][-1]), 1) if e4_data["cpu_temp_c"] is not None else None,
            "throttled": "0x0",
        }

    mcu_e4_path = resolve_input_path("data/raw/esp32_e4_continuous_60s.csv")
    if os.path.exists(mcu_e4_path):
        mcu_e4_data = load_benchmark_csv(mcu_e4_path)
        extra_experiments["mcu_e4_continuous_60s"] = {
            "total_inferences_in_60s": mcu_e4_data["n"],
            "throughput_sustained_inf_s": round(mcu_e4_data["n"] / 60.0, 3),
            "t_inf_median_ms": round(float(np.median(mcu_e4_data["t_inf_ms"])), 3),
            "t_inf_iqr_ms": round(float(np.percentile(mcu_e4_data["t_inf_ms"], 75) - np.percentile(mcu_e4_data["t_inf_ms"], 25)), 3),
            "t_total_median_ms": round(float(np.median(mcu_e4_data["t_total_ms"])), 3),
        }

    h1_iut = None
    if "sbc_int8_4t" in runs and "mcu_int8_nominal" in runs and "mcu_int8_worst_overhead" in runs:
        h1_iut = evaluate_h1_intersection_union(
            runs["sbc_int8_4t"], runs["mcu_int8_nominal"], runs["mcu_int8_worst_overhead"]
        )

    h2_eval = None
    if "sbc_int8_4t" in runs and "sbc_fp32_4t" in runs:
        h2_eval = evaluate_h2_tradeoff_sbc(
            runs["sbc_int8_4t"], runs["sbc_fp32_4t"], extra_experiments.get("e0_map_test")
        )

    clean_runs = {}
    for k, v in runs.items():
        clean_runs[k] = {ck: cv for ck, cv in v.items() if not ck.startswith("_raw_")}

    report = {
        "protocol_version": "v1.5 (docs/projeto/main.tex)",
        "power_model_table5": POWER_MODEL_TABLE5,
        "runs": clean_runs,
        "comparisons": comparisons,
        "hypothesis_h1_iut": h1_iut,
        "hypothesis_h2_sbc_int8_vs_fp32": h2_eval,
        "breakeven_analysis_eq3": breakeven,
        "extra_experiments_v13": extra_experiments,
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8") as f:
        f.write(
            "run_id,n,p_nom_w,t_inf_median_ms,t_inf_iqr_ms,t_total_median_ms,t_total_iqr_ms,"
            "fps_inf_mean,fps_total_mean,energy_inf_nom_mj,energy_3phase_nom_mj,energy_3phase_min_mj,energy_3phase_max_mj,"
            "inf_per_j_inf,inf_per_j_3phase,edp_inf_mj_ms,edp_3phase_mj_ms,e_ciclo_02hz_mj,e_ciclo_05hz_mj,e_ciclo_1hz_mj,"
            "acf_lag1,kw_p,pilot_n_final\n"
        )
        for run_id, r in clean_runs.items():
            p3 = r["pipeline_phases_eq1_ms"]
            ec = r["e_ciclo_periodic_mj"]
            f.write(
                f"{run_id},{r['n']},{r['p_active_w']['nominal']},"
                f"{r['latency_ms']['median']},{r['latency_ms']['iqr']},"
                f"{p3['t_total']['median']},{p3['t_total']['iqr']},"
                f"{r['throughput_fps']['mean']},{p3['throughput_total_fps_mean']},"
                f"{r['energy_mj']['nominal_mean']},{p3['energy_total_3phase_nom_mj']},"
                f"{p3['energy_total_3phase_min_mj']},{p3['energy_total_3phase_max_mj']},"
                f"{r['efficiency_inf_per_j']['nominal_mean']},{p3['efficiency_3phase_inf_per_j_nom']},"
                f"{r['edp_mj_ms']['nominal_mean']},{p3['edp_3phase_nom_mj_ms']},"
                f"{ec.get('0.2hz_mj')},{ec.get('0.5hz_mj')},{ec.get('1fps_mj')},"
                f"{r['diagnostics']['acf_lag1']},{r['diagnostics']['kruskal_wallis_p']},{r['diagnostics']['pilot_n_final']}\n"
            )

    if os.path.isdir("results"):
        if os.path.abspath(args.out_json) != os.path.abspath("results/phase4_e6_report.json"):
            shutil.copyfile(args.out_json, "results/phase4_e6_report.json")
        if os.path.abspath(args.out_csv) != os.path.abspath("results/phase4_energy_stats_summary.csv"):
            shutil.copyfile(args.out_csv, "results/phase4_energy_stats_summary.csv")

    print("=========================================================================")
    print("  RELATÓRIO FASE 4 (E5/E6) — ENERGIA E ESTATÍSTICA (v1.5 / main.tex)")
    print("=========================================================================")
    for run_id, r in clean_runs.items():
        p3 = r["pipeline_phases_eq1_ms"]
        print(f"\n[{run_id}] ({r['file']}, n={r['n']}, P_nom={r['p_active_w']['nominal']} W)")
        print(
            f"  t_inf   (ms)  : Mediana={r['latency_ms']['median']} (IQR={r['latency_ms']['iqr']}) | "
            f"Média={r['latency_ms']['mean']} ± {r['latency_ms']['std']} (CV={r['latency_ms']['cv_percent']}%)"
        )
        print(
            f"  t_total (ms)  : Mediana={p3['t_total']['median']} (IQR={p3['t_total']['iqr']}) | "
            f"Média={p3['t_total']['mean']} ± {p3['t_total']['std']} (CV={p3['t_total']['cv_percent']}%)"
        )
        print(f"  Throughput    : Janela 3 fases={p3['throughput_total_fps_mean']} inf/s | Núcleo t_inf={r['throughput_fps']['mean']} inf/s")
        print(
            f"  Energia 3F(mJ): Nominal={p3['energy_total_3phase_nom_mj']} mJ "
            f"[{p3['energy_total_3phase_min_mj']} – {p3['energy_total_3phase_max_mj']} mJ] | "
            f"Núcleo t_inf={r['energy_mj']['nominal_mean']} mJ [{r['energy_mj']['sensitivity_min_mean']} – {r['energy_mj']['sensitivity_max_mean']} mJ]"
        )
        print(
            f"  Eficiência/EDP: 3F={p3['efficiency_3phase_inf_per_j_nom']} inf/J (EDP={p3['edp_3phase_nom_mj_ms']} mJ·ms) | "
            f"t_inf={r['efficiency_inf_per_j']['nominal_mean']} inf/J (EDP={r['edp_mj_ms']['nominal_mean']} mJ·ms)"
        )
        print(f"  E_ciclo(f) 3F : {r['e_ciclo_3phase_periodic_mj']}")
        print(f"  E_ciclo(f) inf: {r['e_ciclo_periodic_mj']}")

    if h1_iut:
        print("\n--- Teste de Interseção-União (IUT) para H1 (main.tex §4.12) ---")
        p_full = h1_iut["primary_full_window_3phase"]
        print(
            f"  * Janela Completa (3 Fases): Parte 1 (Throughput SBC > MCU) = {p_full['part1_throughput_sbc_gt_mcu']['confirmed_alpha_005']} "
            f"(p={p_full['part1_throughput_sbc_gt_mcu']['p_value']:.3e}) | "
            f"Parte 2 (E_SBC >= E_MCU Nominal) = {p_full['part2_energy_sbc_ge_mcu']['nominal']['confirmed_alpha_005']} "
            f"({p_full['part2_energy_sbc_ge_mcu']['nominal']['sbc_median_mj']} vs {p_full['part2_energy_sbc_ge_mcu']['nominal']['mcu_median_mj']} mJ) "
            f"=> H1 Confirmada = {p_full['iut_h1_confirmed_nominal']}"
        )

    if h2_eval:
        print("\n--- Avaliação de H2 (INT8 vs. FP32 no SBC — main.tex §1.1, §4.3, §4.12) ---")
        h2p = h2_eval["primary_full_window_3phase"]
        print(
            f"  * Speedup t_total: {h2p['speedup_t_total_bca_95']['estimate']}x IC95%={h2p['speedup_t_total_bca_95']['ci_95']} "
            f"(p={h2p['mann_whitney_t_total_int8_lt_fp32_p']:.3e})"
        )
        print(
            f"  * Redução Energia 3F: {h2p['energy_reduction_ratio_3phase_bca_95']['estimate']}x IC95%={h2p['energy_reduction_ratio_3phase_bca_95']['ci_95']} "
            f"({h2p['int8_energy_3phase_median_mj']} vs {h2p['fp32_energy_3phase_median_mj']} mJ, p={h2p['mann_whitney_energy_3phase_int8_lt_fp32_p']:.3e})"
        )

    print(f"\n[OK] Relatórios salvos em '{args.out_json}' e '{args.out_csv}'.")


if __name__ == "__main__":
    main()
