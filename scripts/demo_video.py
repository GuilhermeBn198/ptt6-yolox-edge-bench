#!/usr/bin/env python3
"""
Utilitário de exibição rápida para o vídeo de demonstração (3–5 min) — PTT6 (Protocolo v1.5)
"""
import json
import sys
import pandas as pd

CLASS_NAMES = [
    "helmet", "gloves", "vest", "boots", "goggles",
    "none", "Person", "no_helmet", "no_goggle", "no_gloves", "no_boots"
]


def fmt_det(d: dict):
    cname = d.get("class_name") or (CLASS_NAMES[d["cls"]] if 0 <= d.get("cls", -1) < len(CLASS_NAMES) else f"cls_{d.get('cls')}")
    return (cname, round(float(d["score"]), 3), [round(float(x), 1) for x in d["bbox_xyxy"]])


def show_sbc_results():
    e1 = json.load(open("data/raw/sbc_e1_50_predictions.json", encoding="utf-8"))
    print(f"=== E1 (SBC): Total de detecções nas 50 imagens BGR: {e1['total_detections']} ===")
    for img in e1["images"][:3]:
        dets_short = [fmt_det(d) for d in img["detections"]]
        print(f"  Imagem {img['image_id']}: {img['num_detections']} detecções -> {dets_short}")

    print("\n=== E3 / E3b (SBC - 30 execuções válidas em 3 blocos) ===")
    configs = [
        ("INT8 4 threads", "data/raw/SBC_E3_int8_4t.csv"),
        ("INT8 2 threads", "data/raw/SBC_E3b_int8_2t.csv"),
        ("INT8 1 thread ", "data/raw/SBC_E3b_int8_1t.csv"),
        ("FP32 4 threads", "data/raw/SBC_E3_fp32_4t.csv"),
    ]
    for label, path in configs:
        df = pd.read_csv(path)
        df = df[df["is_warmup"] == 0]
        t_pre = df["t_pre_us"].median() / 1e3
        t_inf = df["t_inf_us"].median() / 1e3
        t_post = df["t_post_us"].median() / 1e3
        t_tot = df["t_total_us"].median() / 1e3
        print(
            f"  {label}: t_pre={t_pre:5.2f} ms | t_inf={t_inf:6.2f} ms | "
            f"t_post={t_post:4.2f} ms | t_total={t_tot:6.2f} ms ({1000.0/t_tot:4.2f} inf/s)"
        )

    e4 = pd.read_csv("data/raw/rpi_e4_continuous_60s.csv")
    print(
        f"\n=== E4 (SBC - 60s laço fechado) ===\n"
        f"  {len(e4)} inferências em {e4['elapsed_s'].max():.1f} s ({len(e4)/e4['elapsed_s'].max():.2f} inf/s sustentado) | "
        f"Temp máx: {e4['cpu_temp_c'].max():.1f} °C | Throttled: 0x0"
    )


def show_mcu_results():
    e5 = json.load(open("data/raw/esp32_e5_memory_footprint.json", encoding="utf-8"))
    e1 = json.load(open("data/raw/mcu_e1_50_predictions.json", encoding="utf-8"))
    df = pd.read_csv("data/raw/MCU_E3_int8.csv")
    df = df[df["is_warmup"] == 0]
    e4 = pd.read_csv("data/raw/esp32_e4_continuous_60s.csv")

    print("=== E5 (ESP32-S3 Memória On-Device) ===")
    print(
        f"  Flash Modelo: {e5['int8_flash_mb']} MiB | Tensor Arena TFLM: {e5['tflm_arena_used_kb']} KiB ({e5['tflm_arena_used_mb']} MiB) | "
        f"SRAM Livre: {e5['free_internal_sram_kb']} KiB | PSRAM Livre: {e5['free_octal_psram_kb']} KiB"
    )
    print("\n=== E1 (ESP32-S3 - 50 imagens BGR) ===")
    print(f"  Total de detecções nas 50 imagens: {e1['total_detections']}")
    for img in e1["images"][:3]:
        dets_short = [fmt_det(d) for d in img["detections"]]
        print(f"  Imagem {img['image_id']}: {img['num_detections']} detecções -> {dets_short}")

    t_pre = df["t_pre_us"].median() / 1e3
    t_inf = df["t_inf_us"].median() / 1e3
    iqr_inf = (df["t_inf_us"].quantile(0.75) - df["t_inf_us"].quantile(0.25)) / 1e3
    t_post = df["t_post_us"].median() / 1e3
    t_tot = df["t_total_us"].median() / 1e3
    print("\n=== E3 (ESP32-S3 - 30 execuções válidas em 3 blocos) ===")
    print(
        f"  t_pre={t_pre:.2f} ms | t_inf={t_inf:.2f} ms (IQR={iqr_inf:.3f} ms) | "
        f"t_post={t_post:.2f} ms | t_total={t_tot:.2f} ms ({1000.0/t_tot:.3f} inf/s)"
    )
    print(
        f"\n=== E4 (ESP32-S3 - 60s laço fechado) ===\n"
        f"  {len(e4)} inferências concluídas em {e4['elapsed_s'].max():.2f} s ({len(e4)/e4['elapsed_s'].max():.3f} inf/s sustentado)"
    )


def show_summary():
    r = json.load(open("data/processed/phase4_e6_report.json", encoding="utf-8"))
    s = r["runs"]["sbc_int8_4t"]
    m = r["runs"]["mcu_int8_nominal"]
    h2 = r["hypothesis_h2_sbc_int8_vs_fp32"]["primary_full_window_3phase"]
    acc = r["hypothesis_h2_sbc_int8_vs_fp32"]["accuracy_retention_e0_test_bgr"]
    rho_vest = acc["int8_BGR"]["AP50_vest"] / acc["fp32_BGR"]["AP50_vest"]

    sbc_t = s["latency_total_3phase_ms"]["median"]
    mcu_t = m["latency_total_3phase_ms"]["median"]
    sbc_e = s["pipeline_phases_eq1_ms"]["energy_total_3phase_nom_mj"]
    mcu_e = m["pipeline_phases_eq1_ms"]["energy_total_3phase_nom_mj"]
    sbc_c02 = s["e_ciclo_3phase_periodic_mj"]["0.2hz_mj"]
    mcu_c02 = m["e_ciclo_3phase_periodic_mj"]["0.2hz_mj"]

    print("================ RESUMO COMPARATIVO FINAL (SBC vs. MCU) ================")
    print(
        f"1. Latência Total (E3)     : SBC = {sbc_t:6.2f} ms (8.13 inf/s) | "
        f"MCU = {mcu_t:7.2f} ms (0.201 inf/s) -> Speedup SBC = {mcu_t / sbc_t:.2f}x"
    )
    print(
        f"2. Energia 1 Inferência(E6): SBC = {sbc_e:6.2f} mJ (3.77 inf/J) | "
        f"MCU = {mcu_e:7.2f} mJ (0.90 inf/J)  -> SBC {mcu_e / sbc_e:.2f}x mais eficiente (Race-to-Idle)"
    )
    print(
        f"3. Ciclo Periódico f=0.2 Hz: SBC = {sbc_c02:7.2f} mJ/ciclo       | "
        f"MCU = {mcu_c02:7.2f} mJ/ciclo        -> MCU {sbc_c02 / mcu_c02:.2f}x mais eficiente (Idle Linux 0.60W)"
    )
    print(
        "4. Pegada de Memória (E5)  : SBC =  70.35 MiB (RSS Linux)     | "
        "MCU =   2.12 MiB (Tensor Arena) -> MCU 33.1x mais enxuta"
    )
    print(
        f"5. Ganho INT8 vs FP32 (H2) : Latência {h2['speedup_t_total_bca_95']['estimate']:.2f}x menor | "
        f"Energia {h2['energy_reduction_ratio_3phase_bca_95']['estimate']:.2f}x menor | "
        f"Retenção AP@50(vest) = {rho_vest:.1%}"
    )
    print("========================================================================")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("sbc", "sbc-results", "3"):
        show_sbc_results()
    elif mode in ("mcu", "mcu-results", "4"):
        show_mcu_results()
    elif mode in ("summary", "resumo", "5"):
        show_summary()
    else:
        show_sbc_results()
        print()
        show_mcu_results()
        print()
        show_summary()
