from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO


def _flatten_dict(data: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, str]:
    items: List[Tuple[str, str]] = []
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            items.extend(_flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def _get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=get_original_cwd())
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _resolve_data_yaml(data_yaml: Path) -> Path:
    data_cfg = OmegaConf.to_container(OmegaConf.load(str(data_yaml)), resolve=True)
    base_dir = data_yaml.parent.resolve()
    root = (base_dir / data_cfg.get("path", ".")).resolve()
    train_path = (root / data_cfg["train"]).resolve()
    val_path = (root / data_cfg["val"]).resolve()

    temp_yaml_path = Path.cwd() / "data_resolved.yaml"
    temp_yaml_path.write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                f"train: {train_path.as_posix()}",
                f"val: {val_path.as_posix()}",
                "names:",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    names = data_cfg.get("names", {})
    with temp_yaml_path.open("a", encoding="utf-8") as file:
        iterable = names.items() if isinstance(names, dict) else enumerate(names)
        for key, value in iterable:
            file.write(f"  {key}: {value}\n")

    return temp_yaml_path


def _read_results(results_csv: Path) -> pd.DataFrame:
    if not results_csv.exists():
        print(
            f"[WARN] results.csv not found at {results_csv}. Skipping plots and metric extraction."
        )
        return pd.DataFrame()
    return pd.read_csv(results_csv)


def _plot_curves(
    df: pd.DataFrame,
    run_name: str,
    plots_dir: Path,
    plot_format: str,
) -> Dict[str, Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}
    if df.empty:
        return saved

    df_plot = df.copy()
    if "epoch" in df_plot.columns:
        df_plot = df_plot.set_index("epoch")

    # Loss curves
    loss_cols = [col for col in ["train/box_loss", "val/box_loss"] if col in df_plot.columns]
    if loss_cols:
        ax = df_plot[loss_cols].plot(title=f"{run_name} - Box loss", marker="o")
        ax.set_xlabel("epoch" if "epoch" in df.columns else "index")
        ax.set_ylabel("loss")
        loss_path = plots_dir / f"{run_name}_loss.{plot_format}"
        plt.tight_layout()
        plt.savefig(loss_path)
        plt.close()
        saved["loss"] = loss_path

    # Precision / recall
    pr_cols = [
        col for col in ["metrics/precision(B)", "metrics/recall(B)"] if col in df_plot.columns
    ]
    if pr_cols:
        ax = df_plot[pr_cols].plot(title=f"{run_name} - Precision/Recall", marker="o")
        ax.set_xlabel("epoch" if "epoch" in df.columns else "index")
        ax.set_ylabel("score")
        pr_path = plots_dir / f"{run_name}_precision_recall.{plot_format}"
        plt.tight_layout()
        plt.savefig(pr_path)
        plt.close()
        saved["precision_recall"] = pr_path

    # mAP
    map_cols = [
        col for col in ["metrics/mAP50(B)", "metrics/mAP50-95(B)"] if col in df_plot.columns
    ]
    if map_cols:
        ax = df_plot[map_cols].plot(title=f"{run_name} - mAP", marker="o")
        ax.set_xlabel("epoch" if "epoch" in df.columns else "index")
        ax.set_ylabel("score")
        map_path = plots_dir / f"{run_name}_map.{plot_format}"
        plt.tight_layout()
        plt.savefig(map_path)
        plt.close()
        saved["map"] = map_path

    return saved


def _log_with_mlflow(
    cfg: DictConfig,
    run_dir: Path,
    df: pd.DataFrame,
    plots: Dict[str, Path],
    resolved_data_yaml: Path,
    best_checkpoint: Path,
) -> None:
    if not cfg.logging.mlflow.enabled:
        print("MLflow logging disabled in config.")
        return

    mlflow.set_tracking_uri(cfg.logging.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.logging.mlflow.experiment_name)

    run_name = cfg.logging.mlflow.run_name or cfg.train.run_name
    params = _flatten_dict(
        {k: v for k, v in OmegaConf.to_container(cfg, resolve=True).items() if k != "hydra"}
    )
    params["git_commit"] = _get_git_commit()
    params["resolved_data_yaml"] = str(resolved_data_yaml)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({key: str(value) for key, value in params.items()})
        if not df.empty:
            last_row = df.iloc[-1]
            for key, value in last_row.items():
                if isinstance(value, (float, int)):
                    safe_key = re.sub(r"[^0-9A-Za-z_\\-\\. /]", "_", str(key))
                    mlflow.log_metric(safe_key, float(value))

        if resolved_data_yaml.exists():
            mlflow.log_artifact(str(resolved_data_yaml), artifact_path="data")

        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            mlflow.log_artifact(str(results_csv), artifact_path="training_logs")

        for name, plot_path in plots.items():
            mlflow.log_artifact(str(plot_path), artifact_path=f"plots/{name}")

        if cfg.logging.mlflow.log_best_checkpoint and best_checkpoint.exists():
            mlflow.log_artifact(str(best_checkpoint), artifact_path="checkpoints")


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    original_cwd = Path(get_original_cwd())
    print(f"Project root: {original_cwd}")
    print(OmegaConf.to_yaml(cfg))

    plots_dir = Path(to_absolute_path(cfg.paths.plots_dir))
    plots_dir.mkdir(parents=True, exist_ok=True)

    project_dir = Path(to_absolute_path(cfg.train.project))
    project_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = Path(to_absolute_path(cfg.data.data_yaml))
    resolved_data_yaml = _resolve_data_yaml(data_yaml)

    model_path = Path(to_absolute_path(cfg.train.model))
    model = YOLO(str(model_path) if model_path.exists() else cfg.train.model)

    model.train(
        data=str(resolved_data_yaml),
        epochs=cfg.train.epochs,
        imgsz=cfg.train.img_size,
        batch=cfg.train.batch,
        project=str(project_dir),
        name=cfg.train.run_name,
        exist_ok=True,
        device=cfg.train.device,
        workers=cfg.train.workers,
        patience=cfg.train.patience,
    )

    run_dir = project_dir / cfg.train.run_name
    results_csv = run_dir / "results.csv"
    results_df = _read_results(results_csv)
    plot_paths = _plot_curves(
        results_df,
        cfg.train.run_name,
        plots_dir,
        cfg.postprocess.plot_format,
    )

    best_checkpoint = run_dir / "weights" / "best.pt"
    _log_with_mlflow(cfg, run_dir, results_df, plot_paths, resolved_data_yaml, best_checkpoint)

    print(f"Training finished. Best checkpoint: {best_checkpoint}")
    if plot_paths:
        for name, path in plot_paths.items():
            print(f"Saved {name} plot -> {path}")


if __name__ == "__main__":
    main()
