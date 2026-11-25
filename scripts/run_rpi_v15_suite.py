#!/usr/bin/env python3
"""
Orquestrador Remoto (Host -> Raspberry Pi Zero 2 W via USB Ethernet Gadget 10.42.0.197)
Protocolo Experimental v1.5 (docs/projeto/main.tex)

Uso:
  /mnt/ssd_dados/venvs/yolox_edge/bin/python -u scripts/run_rpi_v15_suite.py
"""
import os
import shutil
import subprocess

HOST = "filipe@10.42.0.197"
KEY = "/home/filipe/Projetos/Python/Vis_Comp_Disp_Borda/Git_Borda/sbc_rpi/id_ed25519"


def main() -> int:
    ssh_opts = ["-i", KEY, "-o", "StrictHostKeyChecking=no"]

    print("[1/3] Sincronizando benchmark_yolox.py, run_all_v13_experiments.py e as 50 imagens .bin BGR...")
    subprocess.run([
        "scp", *ssh_opts, "-r",
        "raspberry/pi_zero_2w/benchmark_yolox.py",
        "raspberry/pi_zero_2w/run_all_v13_experiments.py",
        "data/inputs_50_bgr_416",
        f"{HOST}:/home/filipe/ppe_benchmark/sbc_rpi/"
    ], check=True)

    print("\n[2/3] Executando bateria completa v1.5 no Raspberry Pi Zero 2 W (E0 ocioso, E1 50 BGR, E2 Piloto 100, E3, E3b, E4 60s, E5)...")
    subprocess.run([
        "ssh", *ssh_opts, HOST,
        "python3 -u /home/filipe/ppe_benchmark/sbc_rpi/run_all_v13_experiments.py"
    ], check=True)

    print("\n[3/3] Coletando todos os arquivos gerados em /home/filipe/ppe_benchmark/data/raw/ para data/raw/...")
    subprocess.run([
        "scp", *ssh_opts,
        f"{HOST}:/home/filipe/ppe_benchmark/data/raw/*",
        "data/raw/"
    ], check=True)

    if os.path.isdir("results"):
        for fname in os.listdir("data/raw"):
            src = os.path.join("data/raw", fname)
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join("results", fname))

    print("\n[OK] Execução e coleta no Raspberry Pi Zero 2 W concluídas com sucesso!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
