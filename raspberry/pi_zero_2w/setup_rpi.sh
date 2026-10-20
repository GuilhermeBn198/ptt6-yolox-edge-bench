#!/bin/bash
# Script de configuração do ambiente de benchmark no Raspberry Pi Zero 2 W
# Projeto PTT6 - PPCA/UFRR

set -e

echo "=== Configurando Ambiente no Raspberry Pi Zero 2 W ==="

# 1. Ajustar o governador de CPU para Performance (máximo clock fixo em 1.0 GHz)
echo "Fixando governador de CPU em performance..."
for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    if [ -f "$gov" ]; then
        echo "performance" | sudo tee "$gov" > /dev/null
    fi
done
echo -n "Governador atual: "
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo -n "Frequência atual (kHz): "
sudo cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq || cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq

# 2. Atualizar pacotes e instalar dependências essenciais
echo "Instalando dependências de sistema..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-numpy python3-pil python3-libgpiod gpiod

# 3. Instalar LiteRT (ai-edge-litert) / tflite-runtime
echo "Instalando motor de inferência LiteRT / TFLite..."
pip3 install --break-system-packages ai-edge-litert || pip3 install --break-system-packages tflite-runtime || sudo apt-get install -y python3-tflite-runtime

echo ""
echo "=== Configuração Concluída com Sucesso! ==="

