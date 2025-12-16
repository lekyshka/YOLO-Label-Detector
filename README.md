# YOLO Label Detector


## Структура

- `configs/` — конфиги Hydra (`data`, `preprocess`, `train`, `infer`, `postprocess`, `logging`, `export`).
- `src/yolo_label_detector/train.py` — обучение.
- `src/yolo_label_detector/infer.py` — инференс.
- `src/yolo_label_detector/export.py` — экспорт в ONNX и TensorRT.
- `dataset/` — данные под DVC (не хранить в git).
- `plots/` — графики обучения.
- `runs/` — артефакты Ultralytics (игнорируются в git).

## Установка

```bash
poetry install
poetry run pre-commit install
```

## Данные (DVC + Google Drive)

```bash
poetry run dvc pull
```

## MLflow UI (локально)

```powershell
poetry run mlflow server --workers 1 ^
  --backend-store-uri sqlite:///mlflow.db ^
  --default-artifact-root file:///%cd%/runs/mlflow ^
  --host 127.0.0.1 ^
  --port 8080
```

UI: http://127.0.0.1:8080

## Обучение

```bash
poetry run python src/yolo_label_detector/train.py
```

Быстрый прогон: `poetry run python src/yolo_label_detector/train.py train.epochs=1 train.batch=4`.
Гиперпараметры — `configs/train/yolov8n.yaml`, датасет — `configs/data/dataset.yaml`. Графики — `plots/`, метрики/параметры — в MLflow.

## Инференс (скрипт)

```bash
poetry run python src/yolo_label_detector/infer.py ^
  infer.model_path=runs/train/exp1/weights/best.pt ^
  infer.source=path/to/images
```

## Экспорт ONNX / TensorRT

```bash
# ONNX (CPU ок)
poetry run python src/yolo_label_detector/export.py `
  export.model_path=runs/train/exp1/weights/best.pt `
  export.onnx.output_path=outputs/export/onnx/model.onnx

# TensorRT (нужны GPU + установленный TensorRT, device=0)
poetry run python src/yolo_label_detector/export.py `
  train.device=0 `
  export.tensorrt.enabled=true `
  export.tensorrt.use_onnx_input=outputs/export/onnx/model.onnx `
  export.tensorrt.output_path=outputs/export/tensorrt/model.engine
```

ONNX автоматически логируется в MLflow (artifact `onnx_model`), если логирование включено.

## Инференс-сервер

### MLflow Serving (простой вариант)

1. Экспортируйте ONNX (см. выше) и убедитесь, что он залогирован в MLflow (`onnx_model`).
2. Найдите `run_id` в MLflow UI.
3. Запустите сервер:

```bash
mlflow models serve -m runs:/<run_id>/onnx_model --host 0.0.0.0 --port 5001
```

4. Пример запроса (Python, onnxruntime готовит вход):

```python
import requests, json, numpy as np
payload = {"inputs": np.zeros((1, 3, 640, 640), dtype="float32").tolist()}
r = requests.post("http://127.0.0.1:5001/invocations",
                  headers={"Content-Type": "application/json"},
                  data=json.dumps(payload))
print(r.json())
```

### Triton Inference Server (более продвинутый)

- Используйте экспортированный `model.onnx`.
- Создайте репозиторий моделей:
  - `model_repository/yolov8/1/model.onnx`
  - `model_repository/yolov8/config.pbtxt` (указать max_batch_size, input name/shape, output).
- Запустите Triton: `tritonserver --model-repository=/path/to/model_repository`.
- Запросы можно слать через tritonclient или HTTP/gRPC. Этот вариант требует GPU и установку Triton.

## Качество кода

```bash
poetry run pre-commit run -a
```
