#!/usr/bin/env python3
"""
Converte um modelo TensorFlow Lite (.tflite) quantizado em array C/C++ (header)
para incorporação direta no firmware do ESP32-S3 / ESP32 (TensorFlow Lite for Microcontrollers).

Uso:
    python models/tflite_to_c_header.py --input models/yolox_nano_ppe_int8.tflite --output models/model_data.h
"""

import argparse
from pathlib import Path


def convert_tflite_to_header(
    tflite_path: Path, output_path: Path, var_name: str = "g_yolox_model_data"
):
  with open(tflite_path, "rb") as f:
    bytes_data = f.read()

  size = len(bytes_data)

  header_lines = [
      "// Auto-gerado a partir de " + tflite_path.name,
      "#ifndef MODEL_DATA_H_",
      "#define MODEL_DATA_H_",
      "",
      "#include <cstdint>",
      "#include <cstddef>",
      "",
      f"// Tamanho total do modelo: {size} bytes ({size / 1024:.2f} KB)",
      f"constexpr size_t {var_name}_len = {size};",
      "",
      f"alignas(16) const unsigned char {var_name}[] = {{",
  ]

  # Formatar em blocos de 12 bytes por linha
  chunk_size = 12
  for i in range(0, size, chunk_size):
    chunk = bytes_data[i : i + chunk_size]
    hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
    comma = "," if i + chunk_size < size else ""
    header_lines.append(f"    {hex_str}{comma}")

  header_lines.extend(["};", "", "#endif // MODEL_DATA_H_", ""])

  output_path.parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(header_lines))

  print(
      f"Modelo C exportado com sucesso: {output_path} ({size / 1024:.2f} KB)"
  )


def main():
  parser = argparse.ArgumentParser(
      description="Converte arquivo .tflite em header C"
  )
  parser.add_argument(
      "--input",
      type=str,
      default="models/yolox_nano_ppe_int8.tflite",
      help="Caminho do arquivo .tflite",
  )
  parser.add_argument(
      "--output",
      type=str,
      default="models/model_data.h",
      help="Caminho do arquivo .h de saída",
  )
  args = parser.parse_args()

  convert_tflite_to_header(Path(args.input), Path(args.output))


if __name__ == "__main__":
  main()

