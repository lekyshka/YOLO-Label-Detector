# YOLO Label Detector

## Описание проекта

Данный проект является набором инструментов для обучения и использования детектора производственных этикеток на изображении на основе модели `YOLO (You Only Look Once) `.

## Структура

```
.
├── configs/                # конфиги hydra
│
├── src/
│   └── yolo_label_detector/
│       ├── train.py
│       ├── infer.py
│       └── export.py
├── dataset/                # датасет, управление через DVC
├── plots/                  # графики обучения
├── runs/                   # артефакты Ultralytics (игнорируются в .gitignore)
└── examples/               # примеры использования


```

## Инициализация окружения

```bash
poetry install
poetry run pre-commit install
```

## Подгрузка данных

```bash
poetry run dvc pull
```

## Запуск Mlflow

### Windows powershell

```powershell
poetry run mlflow server --workers 1 ^
  --backend-store-uri sqlite:///mlflow.db ^
  --default-artifact-root file:///%cd%/runs/mlflow ^
  --host 127.0.0.1 ^
  --port 8080
```

### Linux terminal

```bash
poetry run mlflow server --workers 1 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root file://$(pwd)/runs/mlflow \
  --host 127.0.0.1 \
  --port 8080
```

UI: http://127.0.0.1:8080

## Обучение

```bash
poetry run python src/yolo_label_detector/train.py
```

Быстрый прогон:

```
poetry run python src/yolo_label_detector/train.py train.epochs=1 train.batch=4
```

.
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
poetry run python src/yolo_label_detector/export.py export.model_path=runs/train/exp1/weights/best.pt export.onnx.output_path=outputs/export/onnx/model.onnx
```

ONNX автоматически логируется в MLflow (artifact `onnx_model`), если логирование включено.

## Инференс-сервер

### MLflow Serving ()

1. Экспортируйте ONNX (см. выше) и убедитесь, что он залогирован в MLflow (`onnx_model`).
2. Найдите `run_id` в MLflow UI.
3. Запустите сервер:

```bash
mlflow models serve -m runs:/<run_id>/onnx_model --host 0.0.0.0 --port 5001   --env-manager local
```

4. Пример запроса (см. `prod_inference_example.py`):

```python
poetry run python3 prod_inference_example.py
```

## Качество кода

```bash
poetry run pre-commit run -a
```

## Авторы

Банникова А. Шевцов М. Щербаков К.
