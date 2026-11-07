#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"
cd "$DIR"

if [ -n "$1" ]; then
    PORT="$1"
elif [ -e "/dev/ttyACM0" ]; then
    PORT="/dev/ttyACM0"
else
    PORT="/dev/ttyUSB0"
fi

if [ ! -r "$PORT" ] || [ ! -w "$PORT" ]; then
    echo "ERRO: Permissao negada na porta $PORT."
    echo "Execute no seu terminal: sudo chmod 666 $PORT"
    exit 1
fi

echo "======================================================"
echo "  Compilando e Gravando YOLOX-Nano INT8 ($PORT)       "
echo "======================================================"

. /home/filipe/.espressif/v6.1/esp-idf/export.sh
idf.py build
idf.py -p "$PORT" flash

echo ""
echo "======================================================"
echo "  Iniciando Coleta Serial Automatizada do Benchmark   "
echo "======================================================"

cd "$REPO_ROOT"
python3 telemetry/record_esp32_serial.py --port "$PORT"
