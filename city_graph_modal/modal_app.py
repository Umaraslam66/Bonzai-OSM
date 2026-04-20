"""Modal app for Luxembourg city-graph preparation, training, and evaluation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import modal

ROOT = Path(__file__).resolve().parents[1]

APP_NAME = "bonzai-city-graph-smoke"
DATA_VOLUME_NAME = "bonzai-city-graph-data"
RUNS_VOLUME_NAME = "bonzai-city-graph-runs"

RAW_PBF_REL = "raw/luxembourg-260419.osm.pbf"
DATASET_REL = "processed/luxembourg_graph_v1"
RUN_REL = "runs/luxembourg_graph_v1"

DATA_MOUNT = "/vol/data"
RUNS_MOUNT = "/vol/runs"


def _configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    return logging.getLogger("city_graph_modal.modal_app")


def _read_local_env(name: str) -> Optional[str]:
    if os.environ.get(name):
        return os.environ[name]
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        value = value.strip().strip("'").strip('"')
        return value or None
    return None


def _wandb_secret() -> Optional[modal.Secret]:
    key = _read_local_env("WANDB_API_KEY")
    if not key:
        return None
    return modal.Secret.from_dict({"WANDB_API_KEY": key})


wandb_secret = _wandb_secret()
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
runs_volume = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26,<3",
        "pyproj>=3.6,<4",
        "shapely>=2.0,<3",
        "osmium>=3.7,<4",
        "torch>=2.6,<3",
        "wandb>=0.18,<1",
    )
    .add_local_python_source("city_graph_modal")
)

app = modal.App(APP_NAME)


def _data_path(rel_path: str) -> str:
    return str(Path(DATA_MOUNT) / rel_path)


def _runs_path(rel_path: str) -> str:
    return str(Path(RUNS_MOUNT) / rel_path)


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=60 * 60 * 4,
    volumes={DATA_MOUNT: data_volume},
)
def prepare_luxembourg_dataset(
    pbf_rel_path: str = RAW_PBF_REL,
    dataset_rel_dir: str = DATASET_REL,
    tile_size_m: float = 1000.0,
    tile_overlap_m: float = 192.0,
) -> dict:
    logger = _configure_logging()
    try:
        from .prepare_dataset import prepare_dataset
    except ImportError:  # pragma: no cover
        from city_graph_modal.prepare_dataset import prepare_dataset

    logger.info(
        "starting dataset prep pbf=%s output=%s tile_size_m=%.1f overlap_m=%.1f",
        _data_path(pbf_rel_path),
        _data_path(dataset_rel_dir),
        tile_size_m,
        tile_overlap_m,
    )
    summary = prepare_dataset(
        input_pbf=_data_path(pbf_rel_path),
        output_dir=_data_path(dataset_rel_dir),
        tile_size_m=tile_size_m,
        tile_overlap_m=tile_overlap_m,
    )
    data_volume.commit()
    logger.info("finished dataset prep summary=%s", summary)
    return summary


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 8,
    volumes={DATA_MOUNT: data_volume.read_only(), RUNS_MOUNT: runs_volume},
    secrets=[wandb_secret] if wandb_secret is not None else [],
)
def train_city_graph_remote(
    dataset_rel_dir: str = DATASET_REL,
    run_rel_dir: str = RUN_REL,
    batch_size: int = 8,
    epochs: int = 12,
    learning_rate: float = 3e-4,
    hidden_dim: int = 256,
    num_layers: int = 5,
    num_heads: int = 4,
    dropout: float = 0.1,
    attribute_mask_prob: float = 0.30,
    edge_mask_prob: float = 0.15,
    wandb_project: str = "bonzai-city-graph",
    wandb_run_name: Optional[str] = None,
) -> dict:
    logger = _configure_logging()
    try:
        from .train import train_model
    except ImportError:  # pragma: no cover
        from city_graph_modal.train import train_model

    logger.info(
        "starting training dataset=%s run_dir=%s epochs=%d batch_size=%d lr=%s",
        _data_path(dataset_rel_dir),
        _runs_path(run_rel_dir),
        epochs,
        batch_size,
        learning_rate,
    )
    result = train_model(
        dataset_root=_data_path(dataset_rel_dir),
        output_dir=_runs_path(run_rel_dir),
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        attribute_mask_prob=attribute_mask_prob,
        edge_mask_prob=edge_mask_prob,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )
    runs_volume.commit()
    logger.info("finished training result=%s", result)
    return result


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 2,
    volumes={DATA_MOUNT: data_volume.read_only(), RUNS_MOUNT: runs_volume},
    secrets=[wandb_secret] if wandb_secret is not None else [],
)
def evaluate_city_graph_remote(
    dataset_rel_dir: str = DATASET_REL,
    run_rel_dir: str = RUN_REL,
    split: str = "test",
    checkpoint_name: str = "best.pt",
) -> dict:
    logger = _configure_logging()
    try:
        from .evaluate import evaluate_checkpoint
    except ImportError:  # pragma: no cover
        from city_graph_modal.evaluate import evaluate_checkpoint

    checkpoint_path = Path(_runs_path(run_rel_dir)) / "checkpoints" / checkpoint_name
    output_path = Path(_runs_path(run_rel_dir)) / f"evaluation_{split}.json"
    logger.info(
        "starting evaluation dataset=%s checkpoint=%s split=%s",
        _data_path(dataset_rel_dir),
        checkpoint_path,
        split,
    )
    result = evaluate_checkpoint(
        dataset_root=_data_path(dataset_rel_dir),
        checkpoint_path=str(checkpoint_path),
        output_path=str(output_path),
        split=split,
    )
    runs_volume.commit()
    logger.info("finished evaluation result=%s", result)
    return result


def upload_local_pbf(local_path: str, remote_rel_path: str = RAW_PBF_REL) -> None:
    source = Path(local_path)
    if not source.is_file():
        raise FileNotFoundError(f"local PBF not found: {local_path}")
    volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
    try:
        with volume.batch_upload() as batch:
            batch.put_file(str(source), remote_rel_path)
    except FileExistsError:
        print(f"reusing existing Modal volume file {DATA_VOLUME_NAME}:{remote_rel_path}")


@app.local_entrypoint()
def pipeline(
    local_pbf: str = "data/luxembourg-260419.osm.pbf",
    upload: bool = True,
    prepare: bool = True,
    train: bool = True,
    evaluate: bool = True,
    dataset_rel_dir: str = DATASET_REL,
    run_rel_dir: str = RUN_REL,
    epochs: int = 12,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
) -> None:
    if upload:
        upload_local_pbf(local_pbf, RAW_PBF_REL)
        print(f"uploaded {local_pbf} -> {DATA_VOLUME_NAME}:{RAW_PBF_REL}")

    if prepare:
        prepare_result = prepare_luxembourg_dataset.remote(
            pbf_rel_path=RAW_PBF_REL,
            dataset_rel_dir=dataset_rel_dir,
        )
        print("prepare_result:", prepare_result)

    if train:
        train_result = train_city_graph_remote.remote(
            dataset_rel_dir=dataset_rel_dir,
            run_rel_dir=run_rel_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        print("train_result:", train_result)

    if evaluate:
        eval_result = evaluate_city_graph_remote.remote(
            dataset_rel_dir=dataset_rel_dir,
            run_rel_dir=run_rel_dir,
            split="test",
            checkpoint_name="best.pt",
        )
        print("eval_result:", eval_result)
