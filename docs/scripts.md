> **Nota v1.5 (2026-10-22):** o treino deve seguir `experimentacao.md` §4.2 (sem número fixo de épocas; parada pela validação) e a calibração §4.3 (750 imagens **BGR**; o script atual converte para RGB e usa 100 imagens).

## Fine-tunning Yolox
```bash
# cd ~/Projetos/Python/Vis_Comp_Disp_Borda/Git_Borda/YOLOX
# com venv ativado
python tools/train.py \
  -f exps/custom/yolox_nano_ppe.py \
  -c yolox_nano.pth \
  -b 16 \
  -d 1 \
  --fp16 \
  --cache ram \
  -o
```

Detalhes dos Parâmetros Utilizados:
-f exps/custom/yolox_nano_ppe.py: Utiliza a configuração customizada de 11 classes e resolução 416×416.
-c yolox_nano.pth: Carrega os pesos pré-treinados da Megvii para realizar o fine-tuning.
-b 16: Tamanho do lote (batch size), ideal para os 6 GB de VRAM da sua RTX 4050.
-d 1: Utiliza 1 GPU dedicada para treinamento.
--fp16: Ativa precisão mista (Mixed Precision / Tensor Cores), acelerando o treino em ≈2x e reduzindo o consumo de memória.
--cache ram: Como o dataset de EPIs tem ≈180MB, carregá-lo na memória RAM elimina gargalos de leitura em disco entre as épocas.
-o: Pré-aloca a memória de GPU para evitar fragmentação.

## Testar inferência do modelo
```bash
python tools/demo.py image \
  -f exps/custom/yolox_nano_ppe.py \
  -c YOLOX_outputs/yolox_nano_ppe/best_ckpt.pth \
  --path ../construction-ppe-dataset/images/val/ \
  --conf 0.3 \
  --nms 0.45 \
  --tsize 416 \
  --save_result \
  --device gpu
```

## Exportação para ONNX
```bash
# A partir da pasta YOLOX:
python tools/export_onnx.py \
  --output-name ../models/yolox_nano_ppe_raw.onnx \
  -f exps/custom/yolox_nano_ppe.py \
  -c YOLOX_outputs/yolox_nano_ppe/best_ckpt.pth \
  --opset 13
```

## Preparação de Dados de Calibração (100 imagens de EPIs)
```bash
# A partir da raiz do repositório:
python models/prepare_calibration_data.py
python -c "import numpy as np; np.save('models/calibration_data_nhwc.npy', np.transpose(np.load('models/calibration_data.npy'), (0, 2, 3, 1)))"
```

## Quantização INT8 (Full Integer Quantization) via onnx2tf
```bash
# Converte ONNX -> TensorFlow Lite com quantização completa INT8
onnx2tf \
  -i models/yolox_nano_ppe_raw.onnx \
  -o models/tf_int8_nhwc \
  -tb tf_converter \
  -oiqt \
  -cind images models/calibration_data_nhwc.npy 0 1 \
  -coion

# Copiar os modelos para a pasta models/
cp models/tf_int8_nhwc/yolox_nano_ppe_raw_full_integer_quant.tflite models/yolox_nano_ppe_int8.tflite
cp models/tf_int8_nhwc/yolox_nano_ppe_raw_float32.tflite models/yolox_nano_ppe_float32.tflite
```

## Conversão para Header C (ESP32 / ESP32-S3)
```bash
# Gera o arquivo C/C++ alignas(16) para TensorFlow Lite Micro no ESP32
python models/tflite_to_c_header.py \
  --input models/yolox_nano_ppe_int8.tflite \
  --output models/model_data.h
```

