#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PORT="${1:-/dev/ttyUSB0}"

if [ ! -r "$PORT" ] || [ ! -w "$PORT" ]; then
    echo "ERRO: Permissao negada na porta $PORT."
    echo "Execute no seu terminal: sudo chmod 666 $PORT"
    exit 1
fi

echo "======================================================"
echo "  Gravando Firmware YOLOX-Nano INT8 no ESP32 ($PORT)  "
echo "======================================================"

. /home/filipe/.espressif/v6.1/esp-idf/export.sh
idf.py -p "$PORT" flash

echo ""
echo "======================================================"
echo "  Iniciando Coleta Serial Automatizada do Benchmark   "
echo "======================================================"

python ../../../telemetry/record_esp32_serial.py --port "$PORT"

