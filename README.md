# YOLO Label Detector

Простой пайплайн для обучения и инференса YOLOv8 с Hydra-конфигами, логированием в MLflow и управлением данными через DVC.

## Структура
- `configs/` — иерархические конфиги Hydra (`data`, `preprocess`, `train`, `infer`, `postprocess`, `logging`).
- `src/yolo_label_detector/train.py` — обучение модели.
- `src/yolo_label_detector/infer.py` — инференс.
- `dataset/` — данные под управлением DVC.
- `plots/` — графики обучения.
- `runs/` — артефакты Ultralytics YOLO.

## Установка
```bash
poetry install
poetry run pre-commit install
```

## Данные с Google Drive

```bash
poetry install  
poetry run dvc pull
```

## Запуск MLflow UI (локально)
```bash
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
быстрое обучение (1 эпоха)
```bash
poetry run python src/yolo_label_detector/train.py train.epochs=1 train.batch=4 
```

Основные гиперпараметры: `configs/train/yolov8n.yaml`, датасет: `configs/data/dataset.yaml`. Графики — в `plots/`, метрики/параметры пишутся в MLflow.

## Инференс
```bash
poetry run python src/yolo_label_detector/infer.py ^
  infer.model_path=runs/train/exp1/weights/best.pt ^
  infer.source=path/to/images
```

## Качество кода
```bash
poetry run pre-commit run -a
```
