from __future__ import annotations

import shutil
from pathlib import Path

import hydra
import mlflow
import onnx
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _export_onnx(model: YOLO, cfg: DictConfig) -> Path:
    if not cfg.export.onnx.enabled:
        return Path(cfg.export.onnx.output_path)

    print("Экспорт в ONNX...")
    onnx_path = Path(
        model.export(
            format="onnx",
            opset=cfg.export.onnx.opset,
            dynamic=cfg.export.onnx.dynamic,
            simplify=cfg.export.onnx.simplify,
            half=cfg.export.onnx.half,
            imgsz=cfg.train.img_size,
            device=cfg.train.device,
        )
    )
    target_path = Path(to_absolute_path(cfg.export.onnx.output_path))
    _ensure_parent(target_path)
    shutil.move(str(onnx_path), target_path)
    print(f"ONNX сохранён в: {target_path}")
    return target_path


def _export_tensorrt(model: YOLO, cfg: DictConfig, onnx_path: Path) -> Path:
    if not cfg.export.tensorrt.enabled:
        return Path(cfg.export.tensorrt.output_path)

    print("Экспорт в TensorRT (потребуется GPU + TensorRT)...")
    preferred_onnx = Path(to_absolute_path(cfg.export.tensorrt.use_onnx_input))
    onnx_source = preferred_onnx if preferred_onnx.exists() else onnx_path

    engine_path = Path(
        model.export(
            format="engine",
            half=cfg.export.tensorrt.precision.lower() == "fp16",
            dynamic=cfg.export.onnx.dynamic,
            simplify=cfg.export.onnx.simplify,
            imgsz=cfg.train.img_size,
            device=cfg.train.device,
            opset=cfg.export.onnx.opset,
            workspace=cfg.export.tensorrt.workspace,
            source=str(onnx_source) if onnx_source.exists() else None,
        )
    )
    target_path = Path(to_absolute_path(cfg.export.tensorrt.output_path))
    _ensure_parent(target_path)
    shutil.move(str(engine_path), target_path)
    print(f"TensorRT engine сохранён в: {target_path}")
    return target_path


def _log_onnx_mlflow(cfg: DictConfig, onnx_path: Path) -> None:
    if not cfg.logging.mlflow.enabled or not onnx_path.exists():
        return

    mlflow.set_tracking_uri(cfg.logging.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.logging.mlflow.experiment_name)

    with mlflow.start_run(run_name="export-onnx"):
        mlflow.log_params(
            {"source_checkpoint": cfg.export.model_path, "opset": cfg.export.onnx.opset}
        )
        mlflow.onnx.log_model(
            onnx_model=onnx.load(onnx_path),
            artifact_path="onnx_model",
        )
        print("ONNX модель залогирована в MLflow (artifact_path=onnx_model).")


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print("Конфиг экспорта:")
    print(OmegaConf.to_yaml(cfg.export))

    model_path = Path(to_absolute_path(cfg.export.model_path))
    if not model_path.exists():
        raise FileNotFoundError(f"Не найден checkpoint: {model_path}")

    model = YOLO(str(model_path))

    onnx_path = _export_onnx(model, cfg)

    _log_onnx_mlflow(cfg, onnx_path)

    if onnx_path.exists():
        print(f"ONNX готов: {onnx_path}")


if __name__ == "__main__":
    main()
