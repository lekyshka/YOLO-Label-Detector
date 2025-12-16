from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from ultralytics import YOLO


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    model_path = Path(to_absolute_path(cfg.infer.model_path))
    source_path = Path(to_absolute_path(cfg.infer.source))
    save_dir = Path(to_absolute_path(cfg.infer.save_dir))

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Source for inference not found: {source_path}")

    save_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))

    model.predict(
        source=str(source_path),
        conf=cfg.infer.conf,
        iou=cfg.infer.iou,
        imgsz=cfg.infer.imgsz,
        max_det=cfg.infer.max_det,
        device=cfg.train.device,
        project=str(save_dir),
        name="predictions",
        exist_ok=True,
        save=True,
        save_txt=True,
    )

    print(f"Inference finished. Results saved to: {save_dir/'predictions'}")


if __name__ == "__main__":
    main()
