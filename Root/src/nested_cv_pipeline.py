"""Leakage-safe nested patient-level cross-validation for CME classification.

Statistical contract
--------------------
* Five outer patient-level folds estimate generalization.
* A patient is CME-positive when at least one source eye is CME-positive.
* All available eye images are pooled into one prediction and loss per patient.
* Hyperparameters and benchmark models are compared only on inner folds.
* Each outer test fold is loaded only after the winning configuration is
  selected and the final validation threshold is frozen.
* Outer predictions are written once and are never consumed by selection code.

The script is resumable at the inner-fit and outer-fold levels. It intentionally
keeps development (inner CV) artifacts separate from final (outer test)
artifacts so that the boundary can be audited from the output directory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import io
import itertools
import json
import math
import multiprocessing
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageOps

from utils.misc_utils import (
    build_classification_dataset,
    build_resnet_for_grayscale,
    build_simple_cnn_for_grayscale,
)


LABEL_MAP = {"non_CME": 0, "CME": 1}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "RV_CME_ML_FINAL" / "allpatients_resized"
METRICS = [
    "accuracy",
    "auroc",
    "auprc",
    "sensitivity",
    "specificity",
    "precision",
    "recall",
    "f1",
]
_NORMALIZATION_CACHE: dict[tuple[str, int], tuple[float, float]] = {}
PATIENT_SUFFIX = re.compile(r"_(?:OD|OS|OU)$", flags=re.IGNORECASE)
ANALYSIS_SCHEMA_VERSION = "patient_mil_max_or_v1"


@dataclass(frozen=True)
class ModelConfig:
    learning_rate: float
    scheduler: str
    image_size: int
    fine_tuning: str
    loss: str
    architecture: str = "resnet18"
    initialization: str = "imagenet"

    @property
    def config_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
        return (
            f"lr_{self.learning_rate:.0e}_sched_{self.scheduler}_"
            f"img_{self.image_size}_ft_{self.fine_tuning}_loss_{self.loss}_"
            f"{digest}"
        ).replace("-", "m")


class FocalLoss(nn.Module):
    """Multiclass focal loss with optional training-fold class weights."""

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        row = torch.arange(targets.shape[0], device=targets.device)
        log_pt = log_probs[row, targets]
        pt = probs[row, targets]
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.class_weights is not None:
            loss = loss * self.class_weights[targets]
        return loss.mean()


class Tee:
    def __init__(self, *files: Any) -> None:
        self.files = files

    def write(self, obj: str) -> None:
        for file_obj in self.files:
            file_obj.write(obj)
            file_obj.flush()

    def flush(self) -> None:
        for file_obj in self.files:
            file_obj.flush()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(payload, stream, indent=2, default=json_default)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def empty_accelerator_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def query_nvidia_gpus(min_free_memory_mb: int = 0) -> list[dict[str, Any]]:
    """Return visible NVIDIA devices ordered by free memory.

    nvidia-smi is preferred because it does not create a CUDA context. A
    torch-based fallback supports environments where nvidia-smi is not on PATH.
    """
    if not torch.cuda.is_available():
        return []
    devices: list[dict[str, Any]] = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            index, name, total, free = [part.strip() for part in line.split(",", 3)]
            if int(free) >= min_free_memory_mb:
                devices.append(
                    {
                        "index": int(index),
                        "name": name,
                        "memory_total_mb": int(total),
                        "memory_free_mb": int(free),
                    }
                )
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                    free_mb = int(free_bytes / 1024**2)
                    total_mb = int(total_bytes / 1024**2)
                except (RuntimeError, TypeError):
                    properties = torch.cuda.get_device_properties(index)
                    total_mb = int(properties.total_memory / 1024**2)
                    free_mb = total_mb
                if free_mb >= min_free_memory_mb:
                    devices.append(
                        {
                            "index": index,
                            "name": torch.cuda.get_device_name(index),
                            "memory_total_mb": total_mb,
                            "memory_free_mb": free_mb,
                        }
                    )
    visible_count = torch.cuda.device_count()
    devices = [device for device in devices if device["index"] < visible_count]
    return sorted(devices, key=lambda item: (-item["memory_free_mb"], item["index"]))


def resolve_gpu_ids(args: argparse.Namespace) -> tuple[list[int], list[dict[str, Any]]]:
    available = query_nvidia_gpus(args.min_free_memory_mb)
    available_by_id = {device["index"]: device for device in available}
    request = args.gpus.strip().lower()
    if request in {"cpu", "mps"}:
        return [], available
    if request == "auto":
        selected = [device["index"] for device in available[: args.max_gpus]]
        return selected, available
    requested = [int(value.strip()) for value in request.split(",") if value.strip()]
    missing = [gpu_id for gpu_id in requested if gpu_id not in available_by_id]
    if missing:
        raise RuntimeError(
            f"Requested GPUs are unavailable or below the free-memory threshold: {missing}. "
            f"Eligible GPUs: {sorted(available_by_id)}"
        )
    return requested[: args.max_gpus], available


def resolve_primary_device(
    request: str,
    gpu_ids: list[int],
) -> torch.device:
    """Select CUDA, Apple Metal, or CPU without silently ignoring requests."""
    normalized = request.strip().lower()
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if normalized == "mps":
        if not mps_available:
            raise RuntimeError(
                "Apple MPS was explicitly requested but is unavailable in this "
                "Python process. Run the MPS verification command from a normal "
                "macOS Terminal and confirm an arm64, MPS-enabled PyTorch build."
            )
        return torch.device("mps")
    if gpu_ids:
        return torch.device(f"cuda:{gpu_ids[0]}")
    if normalized == "auto" and mps_available:
        return torch.device("mps")
    return torch.device("cpu")


def derive_patient_id(folder_id: str) -> str:
    """Collapse eye-specific folder IDs to the person-level identifier."""
    patient_id = PATIENT_SUFFIX.sub("", str(folder_id).strip())
    if not patient_id:
        raise ValueError(f"Cannot derive a patient ID from folder: {folder_id!r}")
    return patient_id


def load_dataset(data_root: str) -> pd.DataFrame:
    """Load image records and apply the prespecified patient-level OR label.

    The source directory label remains available as ``eye_label``. The analytic
    ``label`` is one for every image belonging to a patient when at least one of
    that patient's source eye images is stored in the CME directory.
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root.resolve()}")
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_classification_dataset(data_root).copy()
    if df.empty:
        raise ValueError(f"No OD/OS image files found under dataset root: {root.resolve()}")
    df = df.rename(columns={"patient_id": "folder_id"})
    df["patient_id"] = df["folder_id"].map(derive_patient_id)
    if df["label"].dtype == object:
        df["label"] = df["label"].map(LABEL_MAP)
    if df["label"].isna().any():
        raise ValueError("Dataset contains unmapped labels.")
    df["eye_label"] = df["label"].astype(int)
    patient_labels = df.groupby("patient_id")["eye_label"].max()
    df["label"] = df["patient_id"].map(patient_labels).astype(int)
    if df["patient_id"].isna().any():
        raise ValueError("patient_id may not be missing.")
    if not set(df["eye"].unique()).issubset({"OD", "OS"}):
        raise ValueError("Every image must have OD or OS laterality.")
    counts = patient_labels.value_counts()
    print("\nAnalytic patient-level dataset")
    print(f"Images   : {len(df)}")
    print(f"Patients : {df['patient_id'].nunique()}")
    print(f"CME-positive patients : {int(counts.get(1, 0))}")
    print(f"CME-negative patients : {int(counts.get(0, 0))}")
    return df.reset_index(drop=True)


def dataset_manifest(df: pd.DataFrame, data_root: str) -> dict[str, Any]:
    """Describe the exact analytic cohort without exposing image contents."""
    root = Path(data_root).resolve()
    records = []
    for row in df.sort_values(["patient_id", "eye", "image_path"]).itertuples():
        path = Path(row.image_path).resolve()
        try:
            relative_path = path.relative_to(root).as_posix()
        except ValueError:
            relative_path = path.as_posix()
        records.append(
            {
                "relative_path": relative_path,
                "patient_id": str(row.patient_id),
                "folder_id": str(row.folder_id),
                "eye": str(row.eye),
                "eye_label": int(row.eye_label),
                "patient_label": int(row.label),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"))
    patient_labels = patient_table(df)["label"].value_counts()
    eye_labels = df["eye_label"].value_counts()
    patient_assigned_image_labels = df["label"].value_counts()
    return {
        "data_root": str(root),
        "cohort_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "images": int(len(df)),
        "patients": int(df["patient_id"].nunique()),
        "source_eye_image_counts": {
            "CME": int(eye_labels.get(1, 0)),
            "non_CME": int(eye_labels.get(0, 0)),
        },
        "patient_label_assigned_image_counts": {
            "CME": int(patient_assigned_image_labels.get(1, 0)),
            "non_CME": int(patient_assigned_image_labels.get(0, 0)),
        },
        "patient_counts": {
            "CME": int(patient_labels.get(1, 0)),
            "non_CME": int(patient_labels.get(0, 0)),
        },
        "files": records,
    }


def source_manifest() -> dict[str, Any]:
    """Fingerprint the source files that define the analysis path."""
    source_root = Path(__file__).resolve().parent
    source_paths = [
        Path(__file__).resolve(),
        source_root / "dataset" / "dataset.py",
        source_root / "utils" / "misc_utils.py",
        source_root / "utils" / "xai_utils.py",
    ]
    files = []
    for path in source_paths:
        files.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    return {"git_commit": git_commit, "files": files}


def runtime_environment() -> dict[str, Any]:
    """Capture the software and accelerator context for a new run."""
    import PIL
    import sklearn
    import torchvision

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pillow": PIL.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "mps_built": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
        ),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }


def patient_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("patient_id", as_index=False)["label"]
        .first()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )


def patient_splits(
    df: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return stratified patient-ID splits using StratifiedGroupKFold."""
    patients = patient_table(df)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    splits = []
    for train_index, test_index in splitter.split(
        patients,
        y=patients["label"],
        groups=patients["patient_id"],
    ):
        train_ids = patients.iloc[train_index]["patient_id"].to_numpy()
        test_ids = patients.iloc[test_index]["patient_id"].to_numpy()
        if set(train_ids) & set(test_ids):
            raise AssertionError("Patient leakage detected by split construction.")
        splits.append((train_ids, test_ids))
    return splits


def subset_patients(df: pd.DataFrame, ids: Iterable[Any]) -> pd.DataFrame:
    return df[df["patient_id"].isin(set(ids))].copy().reset_index(drop=True)


def assert_disjoint(**partitions: pd.DataFrame) -> None:
    names = list(partitions)
    sets = {
        name: set(frame["patient_id"].unique())
        for name, frame in partitions.items()
    }
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise AssertionError(
                    f"Patient leakage between {left} and {right}: {sorted(overlap)}"
                )


def split_counts(frame: pd.DataFrame, prefix: str) -> dict[str, int]:
    patients = patient_table(frame)
    counts = patients["label"].value_counts()
    return {
        f"{prefix}_patients": int(len(patients)),
        f"{prefix}_cme_positive_patients": int(counts.get(1, 0)),
        f"{prefix}_cme_negative_patients": int(counts.get(0, 0)),
        f"{prefix}_images": int(len(frame)),
    }


def normalization_key(frame: pd.DataFrame, image_size: int) -> tuple[str, int]:
    paths = "\n".join(sorted(frame["image_path"].astype(str)))
    return hashlib.sha1(paths.encode("utf-8")).hexdigest(), image_size


def training_normalization(
    frame: pd.DataFrame,
    image_size: int,
) -> tuple[float, float]:
    """Compute normalization exclusively from the current training images."""
    key = normalization_key(frame, image_size)
    if key in _NORMALIZATION_CACHE:
        return _NORMALIZATION_CACHE[key]
    pixel_sum = 0.0
    pixel_squared_sum = 0.0
    pixel_count = 0
    for image_path in frame["image_path"]:
        image = Image.open(image_path).convert("L").resize((image_size, image_size))
        values = np.asarray(image, dtype=np.float64) / 255.0
        pixel_sum += float(values.sum())
        pixel_squared_sum += float(np.square(values).sum())
        pixel_count += int(values.size)
    mean = pixel_sum / pixel_count
    variance = max(pixel_squared_sum / pixel_count - mean**2, 1e-12)
    result = (float(mean), float(math.sqrt(variance)))
    _NORMALIZATION_CACHE[key] = result
    return result


class PatientBagDataset(Dataset):
    """One training example per patient, containing every available eye image."""

    def __init__(
        self,
        frame: pd.DataFrame,
        transform: Any = None,
        align_laterality: bool = True,
    ) -> None:
        self.transform = transform
        self.align_laterality = align_laterality
        self.groups: list[tuple[str, int, pd.DataFrame]] = []
        for patient_id, group in frame.groupby("patient_id", sort=True):
            labels = group["label"].unique()
            if len(labels) != 1:
                raise ValueError(f"Patient {patient_id} has inconsistent patient labels.")
            ordered = group.sort_values(["eye", "image_path"]).reset_index(drop=True)
            self.groups.append((str(patient_id), int(labels[0]), ordered))

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        patient_id, label, rows = self.groups[index]
        images = []
        for row in rows.itertuples():
            with Image.open(row.image_path) as source:
                image = source.convert("L")
                if self.align_laterality and str(row.eye).upper() == "OS":
                    image = ImageOps.mirror(image)
                if self.transform is not None:
                    image = self.transform(image)
                else:
                    image = transforms.ToTensor()(image)
            images.append(image)
        return torch.stack(images), label, patient_id


def patient_bag_collate(
    batch: list[tuple[torch.Tensor, int, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Pad variable-size patient bags and return a validity mask."""
    if not batch:
        raise ValueError("Cannot collate an empty patient batch.")
    if any(images.shape[0] == 0 for images, _, _ in batch):
        raise ValueError("Every patient bag must contain at least one image.")
    max_images = max(images.shape[0] for images, _, _ in batch)
    batch_size = len(batch)
    channels, height, width = batch[0][0].shape[1:]
    padded = batch[0][0].new_zeros(
        (batch_size, max_images, channels, height, width)
    )
    mask = torch.zeros((batch_size, max_images), dtype=torch.bool)
    labels = torch.empty(batch_size, dtype=torch.long)
    patient_ids = []
    for index, (images, label, patient_id) in enumerate(batch):
        count = images.shape[0]
        padded[index, :count] = images
        mask[index, :count] = True
        labels[index] = int(label)
        patient_ids.append(patient_id)
    return padded, mask, labels, patient_ids


class PatientMILClassifier(nn.Module):
    """Shared eye-image encoder with OR-compatible max-evidence pooling.

    Each image produces a CME log-odds score. A patient's score is the maximum
    across available images, matching the clinical rule that evidence in any
    eye makes the patient CME-positive.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    @property
    def layer4(self) -> nn.Module:
        return self.backbone.layer4

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5:
            raise ValueError("Expected [batch, images, channels, height, width].")
        batch_size, image_count, channels, height, width = images.shape
        if mask is None:
            mask = torch.ones(
                (batch_size, image_count), dtype=torch.bool, device=images.device
            )
        if mask.shape != (batch_size, image_count):
            raise ValueError("Patient bag mask shape does not match the image tensor.")
        instance_logits = self.backbone(
            images.reshape(batch_size * image_count, channels, height, width)
        ).reshape(batch_size, image_count, 2)
        instance_log_odds = instance_logits[..., 1] - instance_logits[..., 0]
        patient_log_odds = instance_log_odds.masked_fill(~mask, -torch.inf).max(dim=1).values
        return torch.stack((-0.5 * patient_log_odds, 0.5 * patient_log_odds), dim=1)


def make_transforms(
    image_size: int,
    mean: float,
    std: float,
) -> tuple[Any, Any]:
    train_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ]
    )
    return train_transform, eval_transform


def make_loader(
    frame: pd.DataFrame,
    transform: Any,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        PatientBagDataset(frame, transform=transform, align_laterality=True),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=patient_bag_collate,
    )


def configure_fine_tuning(model: nn.Module, strategy: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if strategy == "fc_only":
        modules = [model.fc]
    elif strategy == "layer4_fc":
        modules = [model.layer4, model.fc]
    elif strategy == "layer3_layer4_fc":
        modules = [model.layer3, model.layer4, model.fc]
    elif strategy == "full":
        modules = [model]
    else:
        raise ValueError(f"Unknown fine-tuning strategy: {strategy}")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True


def build_model(config: ModelConfig) -> nn.Module:
    if config.architecture == "simple_cnn":
        if config.initialization != "random":
            raise ValueError("Simple CNN only supports random initialization.")
        backbone = build_simple_cnn_for_grayscale(num_classes=2)
        return PatientMILClassifier(backbone)
    if config.architecture != "resnet18":
        raise ValueError(f"Unknown architecture: {config.architecture}")
    backbone = build_resnet_for_grayscale(
        num_classes=2,
        pretrained=config.initialization == "imagenet",
    )
    configure_fine_tuning(backbone, config.fine_tuning)
    return PatientMILClassifier(backbone)


def class_weights(frame: pd.DataFrame, device: torch.device) -> torch.Tensor:
    patient_labels = patient_table(frame)["label"]
    counts = patient_labels.value_counts()
    if counts.get(0, 0) == 0 or counts.get(1, 0) == 0:
        raise ValueError("Both classes must occur in every training partition.")
    weights = torch.tensor(
        [
            len(patient_labels) / (2.0 * counts[0]),
            len(patient_labels) / (2.0 * counts[1]),
        ],
        dtype=torch.float32,
        device=device,
    )
    return weights


def build_loss(
    config: ModelConfig,
    train_frame: pd.DataFrame,
    device: torch.device,
    focal_gamma: float,
) -> nn.Module:
    weights = class_weights(train_frame, device)
    if config.loss == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=weights)
    if config.loss == "focal":
        return FocalLoss(class_weights=weights, gamma=focal_gamma)
    raise ValueError(f"Unknown loss: {config.loss}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    epochs: int,
    step_size: int,
    gamma: float,
    eta_min: float,
) -> Any:
    if config.scheduler == "none":
        return None
    if config.scheduler == "step":
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    if config.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    raise ValueError(f"Unknown scheduler: {config.scheduler}")


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for images, mask, labels, _ in loader:
            logits = model(
                images.to(device, non_blocking=True),
                mask.to(device, non_blocking=True),
            )
            probs = torch.softmax(logits, dim=1)[:, 1]
            targets.append(labels.numpy().astype(int))
            probabilities.append(probs.detach().cpu().numpy())
    return np.concatenate(targets), np.concatenate(probabilities)


def safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid = np.isfinite(thresholds)
    if not valid.any():
        return 0.5
    scores = tpr[valid] - fpr[valid]
    return float(thresholds[valid][int(np.argmax(scores))])


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    samples = 0
    for images, mask, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item()) * len(labels)
        samples += len(labels)
    return running_loss / max(samples, 1)


def fit_with_validation(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    config: ModelConfig,
    args: argparse.Namespace,
    seed: int,
    checkpoint_path: Path,
) -> tuple[nn.Module, dict[str, Any], np.ndarray, np.ndarray]:
    """Fit without access to any outer-test object."""
    seed_everything(seed, deterministic=args.deterministic)
    mean, std = training_normalization(train_frame, config.image_size)
    train_transform, eval_transform = make_transforms(config.image_size, mean, std)
    train_loader = make_loader(
        train_frame,
        train_transform,
        args.batch_size,
        args.num_workers,
        True,
        seed,
    )
    validation_loader = make_loader(
        validation_frame,
        eval_transform,
        args.batch_size,
        args.num_workers,
        False,
        seed,
    )
    model = build_model(config).to(args.device)
    criterion = build_loss(config, train_frame, args.device, args.focal_gamma)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        config,
        args.epochs,
        args.step_size,
        args.scheduler_gamma,
        args.eta_min,
    )

    best_auroc = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, args.device)
        val_targets, val_probs = predict(model, validation_loader, args.device)
        val_auroc = safe_auroc(val_targets, val_probs)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_auroc": val_auroc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        improved = np.isfinite(val_auroc) and val_auroc > best_auroc + args.min_delta
        if improved:
            best_auroc = val_auroc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
                    "prediction_unit": "patient",
                    "pooling": "maximum image-level CME log-odds",
                    "config": asdict(config),
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_validation_auroc": best_auroc,
                    "normalization_mean": mean,
                    "normalization_std": std,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step()
        if epochs_without_improvement >= args.patience:
            break

    if not checkpoint_path.exists():
        raise RuntimeError("No checkpoint was saved; validation AUROC was undefined.")
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_targets, val_probs = predict(model, validation_loader, args.device)
    fit_summary = {
        "seed": seed,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_validation_auroc": float(checkpoint["best_validation_auroc"]),
        "epochs_completed": len(history),
        "normalization_mean": float(checkpoint["normalization_mean"]),
        "normalization_std": float(checkpoint["normalization_std"]),
        "history": history,
    }
    return model, fit_summary, val_targets, val_probs


def prediction_frame(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    **metadata: Any,
) -> pd.DataFrame:
    patient_rows = []
    for patient_id, group in frame.groupby("patient_id", sort=True):
        labels = group["label"].unique()
        if len(labels) != 1:
            raise ValueError(f"Patient {patient_id} has inconsistent patient labels.")
        patient_rows.append(
            {
                "patient_id": patient_id,
                "label": int(labels[0]),
                "image_count": int(len(group)),
                "eyes": "+".join(sorted(group["eye"].astype(str).unique())),
                "source_eye_labels": "+".join(
                    str(value) for value in sorted(group["eye_label"].astype(int).unique())
                ),
            }
        )
    result = pd.DataFrame(patient_rows)
    if len(result) != len(y_true):
        raise AssertionError("Prediction order does not match patient dataset length.")
    for key, value in reversed(list(metadata.items())):
        result.insert(0, key, value)
    result["true_label"] = y_true.astype(int)
    result["prob_CME"] = y_prob.astype(float)
    result["prob_nonCME"] = 1.0 - result["prob_CME"]
    result["threshold"] = float(threshold)
    result["pred_label"] = (result["prob_CME"] >= threshold).astype(int)
    return result


def hyperparameter_grid() -> list[ModelConfig]:
    return [
        ModelConfig(lr, scheduler, image_size, fine_tuning, loss)
        for lr, scheduler, image_size, fine_tuning, loss in itertools.product(
            [1e-5, 1e-4, 1e-3],
            ["step", "cosine", "none"],
            [224, 320],
            ["fc_only", "layer4_fc", "layer3_layer4_fc", "full"],
            ["weighted_cross_entropy", "focal"],
        )
    ]


def run_inner_configuration(
    outer_fold: int,
    outer_development: pd.DataFrame,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    config: ModelConfig,
    run_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    config_dir = (
        run_dir
        / "model_selection_inner_only"
        / f"outer_{outer_fold}"
        / config.config_id
    )
    fold_rows = []
    prediction_frames = []
    for inner_fold, (train_ids, val_ids) in enumerate(inner_splits, start=1):
        inner_dir = config_dir / f"inner_{inner_fold}"
        metrics_path = inner_dir / "metrics.json"
        predictions_path = inner_dir / "validation_predictions.csv"
        if args.resume and metrics_path.exists() and predictions_path.exists():
            with metrics_path.open() as stream:
                fold_rows.append(json.load(stream))
            prediction_frames.append(pd.read_csv(predictions_path))
            continue

        inner_train = subset_patients(outer_development, train_ids)
        inner_validation = subset_patients(outer_development, val_ids)
        assert_disjoint(train=inner_train, validation=inner_validation)
        inner_dir.mkdir(parents=True, exist_ok=True)
        inner_train[["patient_id", "label"]].drop_duplicates().to_csv(
            inner_dir / "train_patients.csv", index=False
        )
        inner_validation[["patient_id", "label"]].drop_duplicates().to_csv(
            inner_dir / "validation_patients.csv", index=False
        )
        seed = args.seed + outer_fold * 10_000 + inner_fold * 100
        checkpoint_path = inner_dir / "temporary_best_model.pth"
        model, fit_summary, targets, probs = fit_with_validation(
            inner_train,
            inner_validation,
            config,
            args,
            seed,
            checkpoint_path,
        )
        threshold = youden_threshold(targets, probs)
        metrics = calculate_metrics(targets, probs, threshold)
        fold_row = {
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "config_id": config.config_id,
            **asdict(config),
            **split_counts(inner_train, "train"),
            **split_counts(inner_validation, "validation"),
            **metrics,
            "best_epoch": fit_summary["best_epoch"],
            "seed": seed,
        }
        predictions = prediction_frame(
            inner_validation,
            targets,
            probs,
            threshold,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            config_id=config.config_id,
            source="inner_validation_only",
        )
        write_json(metrics_path, fold_row)
        write_json(inner_dir / "training_history.json", fit_summary)
        predictions.to_csv(predictions_path, index=False)
        checkpoint_path.unlink(missing_ok=True)
        del model
        empty_accelerator_cache(args.device)
        fold_rows.append(fold_row)
        prediction_frames.append(predictions)

    folds_df = pd.DataFrame(fold_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    pooled_threshold = youden_threshold(
        predictions_df["true_label"].to_numpy(),
        predictions_df["prob_CME"].to_numpy(),
    )
    pooled = calculate_metrics(
        predictions_df["true_label"].to_numpy(),
        predictions_df["prob_CME"].to_numpy(),
        pooled_threshold,
    )
    summary = {
        "outer_fold": outer_fold,
        "config_id": config.config_id,
        **asdict(config),
        "selection_metric": "mean_inner_validation_auroc",
        "inner_auroc_mean": float(folds_df["auroc"].mean()),
        "inner_auroc_sd": float(folds_df["auroc"].std(ddof=1)),
        **{f"inner_{metric}_mean": float(folds_df[metric].mean()) for metric in METRICS},
        **{f"inner_{metric}_sd": float(folds_df[metric].std(ddof=1)) for metric in METRICS},
        **{f"inner_pooled_{key}": value for key, value in pooled.items()},
    }
    folds_df.to_csv(config_dir / "inner_fold_metrics.csv", index=False)
    write_json(config_dir / "inner_summary.json", summary)
    return summary, predictions_df


def inner_configuration_worker(
    outer_fold: int,
    outer_development: pd.DataFrame,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    config: ModelConfig,
    run_dir: str,
    args_dict: dict[str, Any],
    gpu_id: int,
) -> tuple[dict[str, Any], pd.DataFrame, int]:
    """Spawn-safe worker that owns one GPU for one complete configuration."""
    worker_args = argparse.Namespace(**args_dict)
    worker_args.device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)
    summary, predictions = run_inner_configuration(
        outer_fold,
        outer_development,
        inner_splits,
        config,
        Path(run_dir),
        worker_args,
    )
    return summary, predictions, gpu_id


def run_configuration_batch(
    outer_fold: int,
    outer_development: pd.DataFrame,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    configs: list[ModelConfig],
    run_dir: Path,
    args: argparse.Namespace,
) -> list[tuple[dict[str, Any], pd.DataFrame]]:
    """Run configurations sequentially or with one persistent worker per GPU."""
    if len(args.gpu_ids) <= 1:
        results = []
        for config_index, config in enumerate(configs, start=1):
            print(
                f"Outer {outer_fold}/{args.outer_folds}, "
                f"configuration {config_index}/{len(configs)} on {args.device}: "
                f"{config.config_id}"
            )
            results.append(
                run_inner_configuration(
                    outer_fold,
                    outer_development,
                    inner_splits,
                    config,
                    run_dir,
                    args,
                )
            )
        return results

    context = multiprocessing.get_context("spawn")
    executors = {
        gpu_id: concurrent.futures.ProcessPoolExecutor(
            max_workers=1,
            mp_context=context,
        )
        for gpu_id in args.gpu_ids
    }
    future_to_config: dict[concurrent.futures.Future, ModelConfig] = {}
    args_dict = {
        key: value
        for key, value in vars(args).items()
        if key not in {"device", "gpu_ids", "available_gpus"}
    }
    try:
        for config_index, config in enumerate(configs):
            gpu_id = args.gpu_ids[config_index % len(args.gpu_ids)]
            print(
                f"Queueing outer {outer_fold}, configuration "
                f"{config_index + 1}/{len(configs)} on cuda:{gpu_id}: "
                f"{config.config_id}"
            )
            future = executors[gpu_id].submit(
                inner_configuration_worker,
                outer_fold,
                outer_development,
                inner_splits,
                config,
                str(run_dir),
                args_dict,
                gpu_id,
            )
            future_to_config[future] = config

        completed = []
        for future in concurrent.futures.as_completed(future_to_config):
            config = future_to_config[future]
            summary, predictions, gpu_id = future.result()
            print(f"Completed {config.config_id} on cuda:{gpu_id}")
            completed.append((summary, predictions))
        return completed
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=False)


def benchmark_configs(selected: ModelConfig) -> list[tuple[str, ModelConfig]]:
    return [
        ("ImageNet ResNet18", selected),
        (
            "Random ResNet18",
            ModelConfig(
                selected.learning_rate,
                selected.scheduler,
                selected.image_size,
                "full",
                selected.loss,
                architecture="resnet18",
                initialization="random",
            ),
        ),
        (
            "Simple CNN",
            ModelConfig(
                selected.learning_rate,
                selected.scheduler,
                selected.image_size,
                "full",
                selected.loss,
                architecture="simple_cnn",
                initialization="random",
            ),
        ),
    ]


def run_inner_benchmark(
    outer_fold: int,
    outer_development: pd.DataFrame,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    selected: ModelConfig,
    run_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    named_configs = benchmark_configs(selected)
    model_names = {config.config_id: model_name for model_name, config in named_configs}
    batch_results = run_configuration_batch(
        outer_fold,
        outer_development,
        inner_splits,
        [config for _, config in named_configs],
        run_dir / "benchmark",
        args,
    )
    rows, predictions = [], []
    for summary, pred in batch_results:
        model_name = model_names[summary["config_id"]]
        summary = dict(summary)
        summary["benchmark_model"] = model_name
        pred = pred.copy()
        pred["benchmark_model"] = model_name
        rows.append(summary)
        predictions.append(pred)
    return rows, predictions


def final_validation_split(
    outer_development: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_ids, validation_ids = patient_splits(
        outer_development,
        n_splits=n_splits,
        seed=seed,
    )[0]
    train_frame = subset_patients(outer_development, train_ids)
    validation_frame = subset_patients(outer_development, validation_ids)
    assert_disjoint(train=train_frame, validation=validation_frame)
    return train_frame, validation_frame


def evaluate_outer_fold(
    outer_fold: int,
    outer_development: pd.DataFrame,
    outer_test: pd.DataFrame,
    selected: ModelConfig,
    selection_summary: dict[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """The only function that receives outer_test and runs inference on it."""
    outer_dir = run_dir / "final_outer_evaluation" / f"outer_{outer_fold}"
    metrics_path = outer_dir / "outer_test_metrics.json"
    predictions_path = outer_dir / "outer_test_predictions.csv"
    if args.resume and metrics_path.exists() and predictions_path.exists():
        with metrics_path.open() as stream:
            saved_metrics = json.load(stream)
        saved_predictions = pd.read_csv(predictions_path)
        gradcam_status_path = outer_dir / "gradcam" / "status.json"
        gradcam_complete = False
        if gradcam_status_path.exists():
            with gradcam_status_path.open() as stream:
                gradcam_complete = json.load(stream).get("status") == "complete"
        if not args.skip_gradcam and not gradcam_complete:
            checkpoint = torch.load(
                outer_dir / "best_model.pth",
                map_location=args.device,
                weights_only=False,
            )
            model = build_model(selected).to(args.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            generate_gradcam(
                model,
                outer_test,
                selected,
                float(saved_metrics["threshold"]),
                outer_fold,
                outer_dir,
                args,
            )
            del model
        return saved_metrics, saved_predictions

    train_frame, validation_frame = final_validation_split(
        outer_development,
        n_splits=args.final_validation_splits,
        seed=args.seed + outer_fold * 1_000 + 77,
    )
    assert_disjoint(
        train=train_frame,
        validation=validation_frame,
        test=outer_test,
    )
    outer_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [
        ("train", train_frame),
        ("validation", validation_frame),
        ("test", outer_test),
    ]:
        frame[["patient_id", "label"]].drop_duplicates().to_csv(
            outer_dir / f"{name}_patients.csv", index=False
        )
        frame.to_csv(outer_dir / f"{name}_images.csv", index=False)

    seed = args.seed + outer_fold * 100_000 + 999
    checkpoint_path = outer_dir / "best_model.pth"
    model, fit_summary, val_targets, val_probs = fit_with_validation(
        train_frame,
        validation_frame,
        selected,
        args,
        seed,
        checkpoint_path,
    )
    threshold = youden_threshold(val_targets, val_probs)
    validation_metrics = calculate_metrics(val_targets, val_probs, threshold)
    write_json(
        outer_dir / "validation_threshold.json",
        {
            "selection": "Youden J on final validation patients only",
            "threshold": threshold,
            "metrics": validation_metrics,
        },
    )

    _, eval_transform = make_transforms(
        selected.image_size,
        fit_summary["normalization_mean"],
        fit_summary["normalization_std"],
    )
    test_loader = make_loader(
        outer_test,
        eval_transform,
        args.batch_size,
        args.num_workers,
        False,
        seed,
    )
    test_targets, test_probs = predict(model, test_loader, args.device)
    test_metrics = calculate_metrics(test_targets, test_probs, threshold)
    predictions = prediction_frame(
        outer_test,
        test_targets,
        test_probs,
        threshold,
        outer_fold=outer_fold,
        source="outer_test_single_evaluation",
    )
    metrics = {
        "outer_fold": outer_fold,
        "config_id": selected.config_id,
        **asdict(selected),
        **split_counts(train_frame, "train"),
        **split_counts(validation_frame, "validation"),
        **split_counts(outer_test, "test"),
        **test_metrics,
        "best_epoch": fit_summary["best_epoch"],
        "seed": seed,
        "selection_metric": selection_summary["selection_metric"],
        "selected_inner_validation_auroc": selection_summary["inner_auroc_mean"],
    }
    predictions.to_csv(predictions_path, index=False)
    write_json(metrics_path, metrics)
    write_json(outer_dir / "training_history.json", fit_summary)
    write_json(
        outer_dir / "selected_hyperparameters.json",
        {
            **asdict(selected),
            "config_id": selected.config_id,
            "selected_without_outer_test_access": True,
            "inner_selection_summary": selection_summary,
        },
    )
    if not args.skip_gradcam:
        generate_gradcam(
            model,
            outer_test,
            selected,
            threshold,
            outer_fold,
            outer_dir,
            args,
        )
    del model
    empty_accelerator_cache(args.device)
    return metrics, predictions


def generate_gradcam(
    model: nn.Module,
    outer_test: pd.DataFrame,
    config: ModelConfig,
    threshold: float,
    outer_fold: int,
    outer_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Generate patient-category CAMs for each bag's max-evidence image."""
    status_path = outer_dir / "gradcam" / "status.json"
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        checkpoint = torch.load(
            outer_dir / "best_model.pth",
            map_location="cpu",
            weights_only=False,
        )
        mean = float(checkpoint["normalization_mean"])
        std = float(checkpoint["normalization_std"])
        _, eval_transform = make_transforms(config.image_size, mean, std)
        loader = make_loader(
            outer_test,
            eval_transform,
            args.batch_size,
            args.num_workers,
            False,
            args.seed + outer_fold,
        )
        targets, probabilities = predict(model, loader, args.device)
        patient_predictions = prediction_frame(
            outer_test, targets, probabilities, threshold
        )
        patient_predictions["category"] = np.select(
            [
                (patient_predictions["true_label"] == 1) & (patient_predictions["pred_label"] == 1),
                (patient_predictions["true_label"] == 0) & (patient_predictions["pred_label"] == 0),
                (patient_predictions["true_label"] == 0) & (patient_predictions["pred_label"] == 1),
                (patient_predictions["true_label"] == 1) & (patient_predictions["pred_label"] == 0),
            ],
            ["TP", "TN", "FP", "FN"],
            default="unknown",
        )

        gradcam_root = outer_dir / "gradcam"
        reports: dict[str, int] = {}
        rng = np.random.default_rng(args.seed + outer_fold)
        model.eval()
        for category in ["TP", "TN", "FP", "FN"]:
            candidates = patient_predictions[
                patient_predictions["category"] == category
            ]
            if len(candidates) > args.gradcam_samples_per_category:
                chosen_indices = rng.choice(
                    candidates.index.to_numpy(),
                    size=args.gradcam_samples_per_category,
                    replace=False,
                )
                candidates = candidates.loc[chosen_indices]
            category_dir = gradcam_root / "xai_plots" / category
            category_dir.mkdir(parents=True, exist_ok=True)
            records = []
            for patient_row in candidates.itertuples():
                patient_images = outer_test[
                    outer_test["patient_id"] == patient_row.patient_id
                ].sort_values(["eye", "image_path"])
                tensors = []
                rows = []
                for image_row in patient_images.itertuples():
                    with Image.open(image_row.image_path) as source:
                        image = source.convert("L")
                        if str(image_row.eye).upper() == "OS":
                            image = ImageOps.mirror(image)
                        tensors.append(eval_transform(image))
                    rows.append(image_row)
                image_batch = torch.stack(tensors).to(args.device)
                with torch.no_grad():
                    image_logits = model.backbone(image_batch)
                    image_evidence = image_logits[:, 1] - image_logits[:, 0]
                    evidence_index = int(torch.argmax(image_evidence).item())
                selected_tensor = image_batch[evidence_index : evidence_index + 1]
                selected_row = rows[evidence_index]
                target_layer = model.backbone.layer4[-1]
                cam = GradCAM(model=model, target_layers=[target_layer])
                cam_map = cam(
                    input_tensor=selected_tensor,
                    targets=[ClassifierOutputTarget(1)],
                )[0]
                visible = selected_tensor[0, 0].detach().cpu().numpy() * std + mean
                visible = np.clip(visible, 0.0, 1.0)
                rgb = np.repeat(visible[..., None], 3, axis=2)
                heat = plt.get_cmap("jet")(cam_map)[..., :3]
                overlay = np.clip(0.55 * rgb + 0.45 * heat, 0.0, 1.0)
                stem = (
                    f"{category}_{patient_row.patient_id}_{selected_row.eye}_"
                    f"pCME_{patient_row.prob_CME:.3f}"
                )
                png_path = category_dir / f"{stem}_pair.png"
                pdf_path = category_dir / f"{stem}_triple.pdf"
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(rgb)
                axes[0].set_title("Selected max-evidence eye")
                axes[1].imshow(overlay)
                axes[1].set_title("Patient-level Grad-CAM overlay")
                for axis in axes:
                    axis.axis("off")
                fig.tight_layout()
                fig.savefig(png_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(cam_map, cmap="jet")
                axes[0].set_title("Raw CAM")
                axes[1].imshow(rgb)
                axes[1].set_title("Selected eye")
                axes[2].imshow(overlay)
                axes[2].set_title("Overlay")
                for axis in axes:
                    axis.axis("off")
                fig.tight_layout()
                fig.savefig(pdf_path, bbox_inches="tight")
                plt.close(fig)
                records.append(
                    {
                        "patient_id": patient_row.patient_id,
                        "category": category,
                        "true_patient_label": int(patient_row.true_label),
                        "predicted_patient_label": int(patient_row.pred_label),
                        "prob_CME": float(patient_row.prob_CME),
                        "selected_eye": selected_row.eye,
                        "selected_source_eye_label": int(selected_row.eye_label),
                        "selected_image_path": selected_row.image_path,
                        "selection_rule": "maximum image-level CME log-odds within patient bag",
                        "pair_png_path": str(png_path),
                        "triple_pdf_path": str(pdf_path),
                    }
                )
            pd.DataFrame(records).to_csv(
                category_dir / f"xai_report_{category}.csv", index=False
            )
            reports[category] = len(records)
        write_json(
            status_path,
            {
                "status": "complete",
                "unit": "patient",
                "attributed_instance": "max-evidence image in each patient bag",
                "samples": reports,
            },
        )
    except Exception as exc:
        write_json(
            status_path,
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "instruction": "Install pytorch-grad-cam and rerun with --resume.",
            },
        )
        print(f"WARNING: Grad-CAM failed for outer fold {outer_fold}: {exc}")


def patient_bootstrap_ci(
    predictions: pd.DataFrame,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    patient_ids = predictions["patient_id"].unique()
    rows = []
    for iteration in range(iterations):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        pieces = []
        for draw_index, patient_id in enumerate(sampled):
            piece = predictions[predictions["patient_id"] == patient_id].copy()
            piece["bootstrap_patient"] = f"{draw_index}_{patient_id}"
            pieces.append(piece)
        boot = pd.concat(pieces, ignore_index=True)
        metrics = calculate_metrics(
            boot["true_label"].to_numpy(),
            boot["prob_CME"].to_numpy(),
            boot["threshold"].to_numpy(),
        )
        rows.append({"iteration": iteration, **metrics})
    samples = pd.DataFrame(rows)
    result = []
    for metric in METRICS:
        result.append(
            {
                "metric": metric,
                "ci_lower": float(samples[metric].quantile(0.025)),
                "ci_upper": float(samples[metric].quantile(0.975)),
            }
        )
    return pd.DataFrame(result)


def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | np.ndarray,
) -> dict[str, float | int]:
    thresholds = np.asarray(threshold, dtype=float)
    y_pred = (y_prob >= thresholds).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    threshold_value = (
        float(thresholds) if thresholds.ndim == 0 else float("nan")
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auroc": safe_auroc(y_true, y_prob),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold_value,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def mean_sd(value: pd.Series) -> str:
    return f"{value.mean():.3f} ± {value.std(ddof=1):.3f}"


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def write_latex_table(
    frame: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
) -> None:
    latex = frame.to_latex(
        index=False,
        escape=True,
        float_format=lambda value: f"{value:.3f}",
        caption=caption,
        label=label,
        position="ht",
    )
    path.write_text(latex)


def make_table_1(
    outer_metrics: pd.DataFrame,
    pooled: dict[str, Any],
    ci: pd.DataFrame,
    tables_dir: Path,
) -> pd.DataFrame:
    ci_lookup = ci.set_index("metric")
    rows = []
    for metric in METRICS:
        rows.append(
            {
                "Metric": metric.upper() if metric in {"auroc", "auprc"} else metric.title(),
                "Mean ± SD": mean_sd(outer_metrics[metric]),
                "Pooled": pooled[metric],
                "95% CI": (
                    f"{ci_lookup.loc[metric, 'ci_lower']:.3f}--"
                    f"{ci_lookup.loc[metric, 'ci_upper']:.3f}"
                ),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(tables_dir / "Table_1_classification_performance.csv", index=False)
    write_latex_table(
        table,
        tables_dir / "Table_1_classification_performance.tex",
        "Nested patient-level cross-validation performance from untouched outer test folds.",
        "tab:nested_performance",
    )
    return table


def summarize_development_table(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    rows = []
    for group, group_metrics in metrics.groupby(group_column):
        group_predictions = predictions[predictions[group_column] == group]
        threshold = youden_threshold(
            group_predictions["true_label"].to_numpy(),
            group_predictions["prob_CME"].to_numpy(),
        )
        pooled = calculate_metrics(
            group_predictions["true_label"].to_numpy(),
            group_predictions["prob_CME"].to_numpy(),
            threshold,
        )
        row = {group_column: group}
        for metric in METRICS:
            source = (
                f"inner_{metric}_mean"
                if f"inner_{metric}_mean" in group_metrics
                else metric
            )
            row[f"{metric}_mean_sd"] = mean_sd(group_metrics[source])
            row[f"pooled_{metric}"] = pooled[metric]
        row["data_role"] = "inner_validation_model_development_only"
        rows.append(row)
    return pd.DataFrame(rows)


def make_ablation_table(
    inner_fold_metrics: pd.DataFrame,
    inner_predictions: pd.DataFrame,
    tables_dir: Path,
) -> pd.DataFrame:
    factors = ["learning_rate", "scheduler", "image_size", "fine_tuning", "loss"]
    rows = []
    for factor in factors:
        for level, metrics in inner_fold_metrics.groupby(factor):
            config_ids = set(metrics["config_id"])
            predictions = inner_predictions[
                inner_predictions["config_id"].isin(config_ids)
            ]
            threshold = youden_threshold(
                predictions["true_label"].to_numpy(),
                predictions["prob_CME"].to_numpy(),
            )
            pooled = calculate_metrics(
                predictions["true_label"].to_numpy(),
                predictions["prob_CME"].to_numpy(),
                threshold,
            )
            row = {
                "Factor": factor,
                "Level": level,
                "Inner folds": len(metrics),
                "Inner AUROC mean ± SD": mean_sd(metrics["auroc"]),
                "Inner AUPRC mean ± SD": mean_sd(metrics["auprc"]),
                "Inner accuracy mean ± SD": mean_sd(metrics["accuracy"]),
                "Pooled inner AUROC": pooled["auroc"],
                "Pooled inner AUPRC": pooled["auprc"],
                "Data role": "model development only; no outer-test comparisons",
            }
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(tables_dir / "Table_3_inner_cv_ablation_summary.csv", index=False)
    write_latex_table(
        table,
        tables_dir / "Table_3_inner_cv_ablation_summary.tex",
        "Ablation summary calculated exclusively from inner validation folds.",
        "tab:inner_ablation",
    )
    return table


def plot_workflow(figures_dir: Path) -> None:
    labels = [
        "Outer patient-level fold",
        "Development pool (80%)",
        "Inner stratified group CV",
        "Hyperparameter selection\n(mean inner validation AUROC)",
        "Patient MIL fit\n(all images; max CME evidence)",
        "Validation patients\n(early stopping + threshold)",
        "Untouched outer test fold\n(one evaluation only)",
    ]
    colors = ["#34495e", "#2980b9", "#16a085", "#8e44ad", "#d35400", "#f39c12", "#c0392b"]
    fig, axis = plt.subplots(figsize=(9, 13))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    y_positions = np.linspace(0.93, 0.08, len(labels))
    for index, (label, color, y_pos) in enumerate(zip(labels, colors, y_positions)):
        axis.text(
            0.5,
            y_pos,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.7", "facecolor": color, "edgecolor": "white"},
        )
        if index < len(labels) - 1:
            axis.annotate(
                "",
                xy=(0.5, y_positions[index + 1] + 0.045),
                xytext=(0.5, y_pos - 0.045),
                arrowprops={"arrowstyle": "->", "lw": 2, "color": "#2c3e50"},
            )
    axis.text(
        0.98,
        0.08,
        "Outer test data are unavailable to all selection functions",
        rotation=90,
        va="bottom",
        ha="right",
        fontsize=10,
        color="#c0392b",
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "Figure_1_nested_cross_validation.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "Figure_1_nested_cross_validation.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_outer_performance(predictions: pd.DataFrame, figures_dir: Path) -> None:
    y_true = predictions["true_label"].to_numpy()
    y_prob = predictions["prob_CME"].to_numpy()
    y_pred = predictions["pred_label"].to_numpy()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, axis = plt.subplots(figsize=(7, 7))
    axis.plot(fpr, tpr, lw=2, label=f"Nested CV AUROC = {roc_auc_score(y_true, y_prob):.3f}")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Pooled outer-test ROC curve")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figures_dir / "pooled_outer_test_roc.png", dpi=300)
    fig.savefig(figures_dir / "pooled_outer_test_roc.pdf")
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, axis = plt.subplots(figsize=(7, 7))
    axis.plot(recall, precision, lw=2, label=f"Nested CV AUPRC = {average_precision_score(y_true, y_prob):.3f}")
    axis.axhline(y_true.mean(), linestyle="--", color="gray", label=f"Prevalence = {y_true.mean():.3f}")
    axis.set(xlabel="Recall", ylabel="Precision", title="Pooled outer-test precision-recall curve")
    axis.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(figures_dir / "pooled_outer_test_pr.png", dpi=300)
    fig.savefig(figures_dir / "pooled_outer_test_pr.pdf")
    plt.close(fig)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(6.5, 6))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, matrix[row, column], ha="center", va="center", fontsize=16)
    axis.set_xticks([0, 1], ["non-CME", "CME"])
    axis.set_yticks([0, 1], ["non-CME", "CME"])
    axis.set(xlabel="Predicted", ylabel="True", title="Pooled outer-test confusion matrix")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(figures_dir / "pooled_outer_test_confusion_matrix.png", dpi=300)
    fig.savefig(figures_dir / "pooled_outer_test_confusion_matrix.pdf")
    plt.close(fig)


def make_gradcam_composite(run_dir: Path, figures_dir: Path) -> None:
    """Combine final-model Grad-CAM examples across outcome categories."""
    category_paths: dict[str, Path] = {}
    for category in ["TP", "TN", "FP", "FN"]:
        candidates = sorted(
            (
                run_dir / "final_outer_evaluation"
            ).glob(f"outer_*/gradcam/xai_plots/{category}/*_pair.png")
        )
        if candidates:
            category_paths[category] = candidates[0]
    if not category_paths:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for axis, category in zip(axes.ravel(), ["TP", "TN", "FP", "FN"]):
        path = category_paths.get(category)
        if path is None:
            axis.text(0.5, 0.5, f"No {category} sample", ha="center", va="center")
        else:
            axis.imshow(plt.imread(path))
            axis.set_title(
                f"{category}: final outer-fold model",
                fontsize=12,
                fontweight="bold",
            )
        axis.axis("off")
    fig.suptitle(
        "Grad-CAM examples generated from nested-CV final models",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "gradcam_final_outer_models.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "gradcam_final_outer_models.pdf", bbox_inches="tight")
    plt.close(fig)


def write_manuscript_results(
    outer_metrics: pd.DataFrame,
    pooled: dict[str, Any],
    ci: pd.DataFrame,
    run_dir: Path,
) -> None:
    """Write data-driven prose and LaTeX macros for manuscript revision."""
    ci_lookup = ci.set_index("metric")
    publication_dir = run_dir / "publication"
    publication_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "NESTED CROSS-VALIDATION RESULTS",
        "",
        (
            f"Across pooled untouched outer-test predictions, AUROC was "
            f"{pooled['auroc']:.3f} (95% CI "
            f"{ci_lookup.loc['auroc', 'ci_lower']:.3f}--"
            f"{ci_lookup.loc['auroc', 'ci_upper']:.3f}) and AUPRC was "
            f"{pooled['auprc']:.3f} (95% CI "
            f"{ci_lookup.loc['auprc', 'ci_lower']:.3f}--"
            f"{ci_lookup.loc['auprc', 'ci_upper']:.3f})."
        ),
        (
            f"Using fold-specific thresholds selected exclusively on final "
            f"validation patients, pooled accuracy was {pooled['accuracy']:.3f}, "
            f"sensitivity was {pooled['sensitivity']:.3f}, specificity was "
            f"{pooled['specificity']:.3f}, precision was {pooled['precision']:.3f}, "
            f"recall was {pooled['recall']:.3f}, and F1 was {pooled['f1']:.3f}."
        ),
        "",
        "FOLD-LEVEL MEAN +/- SD",
    ]
    for metric in METRICS:
        lines.append(f"{metric}: {mean_sd(outer_metrics[metric])}")
    (publication_dir / "manuscript_results_summary.txt").write_text(
        "\n".join(lines) + "\n"
    )

    macro_names = {
        "accuracy": "NestedAccuracy",
        "auroc": "NestedAUROC",
        "auprc": "NestedAUPRC",
        "sensitivity": "NestedSensitivity",
        "specificity": "NestedSpecificity",
        "precision": "NestedPrecision",
        "recall": "NestedRecall",
        "f1": "NestedFOne",
    }
    macro_lines = [
        "% Automatically generated by nested_cv_pipeline.py.",
        "% Values come only from untouched outer-test predictions.",
    ]
    for metric, macro in macro_names.items():
        macro_lines.extend(
            [
                rf"\newcommand{{\{macro}}}{{{pooled[metric]:.3f}}}",
                rf"\newcommand{{\{macro}CI}}{{"
                f"{ci_lookup.loc[metric, 'ci_lower']:.3f}--"
                f"{ci_lookup.loc[metric, 'ci_upper']:.3f}"
                "}",
                rf"\newcommand{{\{macro}MeanSD}}{{{mean_sd(outer_metrics[metric])}}}",
            ]
        )
    (publication_dir / "nested_cv_results_macros.tex").write_text(
        "\n".join(macro_lines) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, choices=[4, 5], default=4)
    parser.add_argument("--final-validation-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gpus",
        default="auto",
        help=(
            "Accelerator allocation: 'auto' prefers eligible NVIDIA CUDA GPUs, "
            "then Apple MPS, then CPU; '0,1,2' selects CUDA device IDs; "
            "'mps' requests an Apple Silicon GPU; and 'cpu' disables acceleration."
        ),
    )
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=3,
        help="Maximum GPUs used when --gpus auto is selected.",
    )
    parser.add_argument(
        "--min-free-memory-mb",
        type=int,
        default=2000,
        help="Ignore GPUs with less than this much free memory.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--gradcam-samples-per-category", type=int, default=5)
    parser.add_argument("--skip-gradcam", action="store_true")
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Smoke-test only: omit the development-only architecture benchmark.",
    )
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--non-deterministic", dest="deterministic", action="store_false")
    parser.add_argument(
        "--config-start",
        type=int,
        default=0,
        help=(
            "Zero-based first configuration in the prespecified grid. Use with "
            "--max-configs for hardware preflight or deliberate multi-machine sharding."
        ),
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Smoke-test only: truncate the prespecified grid.",
    )
    parser.set_defaults(resume=True, deterministic=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.gpu_ids, args.available_gpus = resolve_gpu_ids(args)
    args.device = resolve_primary_device(args.gpus, args.gpu_ids)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(
        args.output_dir
        or Path("experiment_logs") / f"NESTED_CV_{timestamp}"
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = (run_dir / "console_output.txt").open("a")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)
    started = time.time()

    try:
        seed_everything(args.seed, deterministic=args.deterministic)
        df = load_dataset(args.data_root)
        cohort_manifest = dataset_manifest(df, args.data_root)
        existing_run_manifest_path = run_dir / "run_manifest.json"
        if args.resume and existing_run_manifest_path.exists():
            with existing_run_manifest_path.open() as stream:
                existing_run_manifest = json.load(stream)
            existing_schema = existing_run_manifest.get("analysis_schema_version")
            existing_cohort = existing_run_manifest.get("dataset", {}).get("cohort_sha256")
            if existing_schema != ANALYSIS_SCHEMA_VERSION:
                raise RuntimeError(
                    "Refusing to resume an incompatible run: expected analysis schema "
                    f"{ANALYSIS_SCHEMA_VERSION!r}, found {existing_schema!r}. Use a new "
                    "output directory."
                )
            if existing_cohort != cohort_manifest["cohort_sha256"]:
                raise RuntimeError(
                    "Refusing to resume because the dataset fingerprint changed. "
                    "Use a new output directory."
                )
        write_json(run_dir / "dataset_manifest.json", cohort_manifest)
        write_json(run_dir / "source_manifest.json", source_manifest())
        full_grid = hyperparameter_grid()
        if args.config_start < 0 or args.config_start >= len(full_grid):
            raise ValueError(
                f"--config-start must be between 0 and {len(full_grid) - 1}."
            )
        configs = full_grid[args.config_start :]
        if args.max_configs is not None:
            if args.max_configs < 1:
                raise ValueError("--max-configs must be positive.")
            configs = configs[: args.max_configs]
        if len(configs) != 144 and args.max_configs is None and args.config_start == 0:
            raise AssertionError(f"Expected 144 configurations, found {len(configs)}")

        write_json(
            run_dir / "run_manifest.json",
            {
                "created": datetime.now().isoformat(),
                "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
                "statistical_contract": {
                    "endpoint": "patient-level CME",
                    "label_rule": "CME-positive if any available eye image is CME-positive",
                    "model_input": "all available patient images as a variable-size bag",
                    "patient_pooling": "maximum image-level CME log-odds (multiple-instance learning)",
                    "laterality_preprocessing": "mirror OS images before shared encoding",
                    "outer_split": "StratifiedGroupKFold by patient_id",
                    "outer_test_use": "single final evaluation only",
                    "inner_split": "StratifiedGroupKFold by patient_id",
                    "selection_criterion": "mean inner validation AUROC",
                    "final_threshold": "Youden J on final validation subset only",
                    "benchmark_and_ablation_data": "inner validation only",
                },
                "arguments": {
                    key: str(value) if key == "device" else value
                    for key, value in vars(args).items()
                },
                "gpu_allocation": {
                    "selected_accelerator": str(args.device),
                    "selected_gpu_ids": args.gpu_ids,
                    "available_devices": args.available_gpus,
                    "strategy": (
                        "one persistent process per CUDA GPU; one configuration per GPU at a time"
                        if len(args.gpu_ids) > 1
                        else f"single-process sequential execution on {args.device}"
                    ),
                },
                "runtime_environment": runtime_environment(),
                "grid": [asdict(config) | {"config_id": config.config_id} for config in configs],
                "grid_selection": {
                    "prespecified_total": len(full_grid),
                    "start_index": args.config_start,
                    "selected_count": len(configs),
                    "complete_grid": args.config_start == 0 and len(configs) == len(full_grid),
                },
                "dataset": {
                    key: value
                    for key, value in cohort_manifest.items()
                    if key != "files"
                },
            },
        )
        figures_dir = run_dir / "publication" / "figures"
        tables_dir = run_dir / "publication" / "tables"
        figures_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        plot_workflow(figures_dir)
        if args.gpu_ids:
            print(f"Selected GPUs: {args.gpu_ids}")
            for device in args.available_gpus:
                marker = "*" if device["index"] in args.gpu_ids else " "
                print(
                    f"{marker} cuda:{device['index']} {device['name']} "
                    f"free={device['memory_free_mb']} MiB/"
                    f"{device['memory_total_mb']} MiB"
                )
        elif args.device.type == "mps":
            print("Selected Apple Metal Performance Shaders device: mps")
            print("MPS configurations run sequentially in one process.")
        else:
            print("No eligible NVIDIA GPU selected; using CPU sequential execution.")

        outer_splits = patient_splits(df, args.outer_folds, args.seed)
        outer_metrics_rows = []
        outer_predictions_frames = []
        all_inner_fold_metrics = []
        all_inner_predictions = []
        benchmark_rows = []
        benchmark_predictions = []
        fold_assignment_rows = []

        for outer_fold, (development_ids, test_ids) in enumerate(outer_splits, start=1):
            outer_development = subset_patients(df, development_ids)
            outer_test = subset_patients(df, test_ids)
            assert_disjoint(development=outer_development, test=outer_test)
            for patient_id in development_ids:
                fold_assignment_rows.append(
                    {"outer_fold": outer_fold, "patient_id": patient_id, "role": "development"}
                )
            for patient_id in test_ids:
                fold_assignment_rows.append(
                    {"outer_fold": outer_fold, "patient_id": patient_id, "role": "test"}
                )

            # Outer test is deliberately not passed to either selection function.
            inner_splits = patient_splits(
                outer_development,
                args.inner_folds,
                args.seed + outer_fold * 1_000,
            )
            summaries = []
            config_predictions = {}
            batch_results = run_configuration_batch(
                outer_fold,
                outer_development,
                inner_splits,
                configs,
                run_dir,
                args,
            )
            for summary, predictions in batch_results:
                summaries.append(summary)
                config_predictions[summary["config_id"]] = predictions
                inner_metrics_path = (
                    run_dir
                    / "model_selection_inner_only"
                    / f"outer_{outer_fold}"
                    / summary["config_id"]
                    / "inner_fold_metrics.csv"
                )
                all_inner_fold_metrics.append(pd.read_csv(inner_metrics_path))
                all_inner_predictions.append(predictions)

            summary_df = pd.DataFrame(summaries).sort_values(
                ["inner_auroc_mean", "config_id"],
                ascending=[False, True],
            )
            summary_df.to_csv(
                run_dir
                / "model_selection_inner_only"
                / f"outer_{outer_fold}"
                / "configuration_ranking_inner_only.csv",
                index=False,
            )
            selected_row = summary_df.iloc[0].to_dict()
            selected = next(
                config for config in configs if config.config_id == selected_row["config_id"]
            )
            write_json(
                run_dir
                / "model_selection_inner_only"
                / f"outer_{outer_fold}"
                / "selected_configuration.json",
                {
                    "selection_criterion": "maximum mean inner validation AUROC",
                    "outer_test_consulted": False,
                    "selected": selected_row,
                },
            )

            if not args.skip_benchmark:
                fold_benchmark_rows, fold_benchmark_predictions = run_inner_benchmark(
                    outer_fold,
                    outer_development,
                    inner_splits,
                    selected,
                    run_dir,
                    args,
                )
                benchmark_rows.extend(fold_benchmark_rows)
                benchmark_predictions.extend(fold_benchmark_predictions)

            outer_metrics, outer_predictions = evaluate_outer_fold(
                outer_fold,
                outer_development,
                outer_test,
                selected,
                selected_row,
                run_dir,
                args,
            )
            outer_metrics_rows.append(outer_metrics)
            outer_predictions_frames.append(outer_predictions)

        assignments = pd.DataFrame(fold_assignment_rows)
        assignments.to_csv(run_dir / "outer_fold_patient_assignments.csv", index=False)
        outer_metrics_df = pd.DataFrame(outer_metrics_rows).sort_values("outer_fold")
        outer_predictions_df = pd.concat(outer_predictions_frames, ignore_index=True)
        outer_metrics_df.to_csv(run_dir / "outer_fold_metrics.csv", index=False)
        outer_predictions_df.to_csv(run_dir / "all_outer_test_predictions.csv", index=False)

        pooled = calculate_metrics(
            outer_predictions_df["true_label"].to_numpy(),
            outer_predictions_df["prob_CME"].to_numpy(),
            outer_predictions_df["threshold"].to_numpy(),
        )
        write_json(run_dir / "pooled_outer_test_metrics.json", pooled)
        ci = patient_bootstrap_ci(
            outer_predictions_df,
            args.bootstrap_iterations,
            args.seed + 909,
        )
        ci.to_csv(run_dir / "patient_bootstrap_95ci.csv", index=False)

        inner_fold_metrics_df = pd.concat(all_inner_fold_metrics, ignore_index=True)
        inner_predictions_df = pd.concat(all_inner_predictions, ignore_index=True)
        inner_fold_metrics_df.to_csv(
            run_dir / "all_inner_fold_metrics.csv", index=False
        )
        inner_predictions_df.to_csv(
            run_dir / "all_inner_validation_predictions.csv", index=False
        )

        if benchmark_rows:
            benchmark_df = pd.DataFrame(benchmark_rows)
            benchmark_predictions_df = pd.concat(benchmark_predictions, ignore_index=True)
            benchmark_df.to_csv(run_dir / "all_inner_benchmark_summaries.csv", index=False)
            benchmark_predictions_df.to_csv(
                run_dir / "all_inner_benchmark_predictions.csv", index=False
            )
            benchmark_table = summarize_development_table(
                benchmark_df,
                benchmark_predictions_df,
                "benchmark_model",
            )
            benchmark_table.to_csv(
                tables_dir / "Table_2_internal_benchmark_inner_cv_only.csv",
                index=False,
            )
            write_latex_table(
                benchmark_table,
                tables_dir / "Table_2_internal_benchmark_inner_cv_only.tex",
                "Internal benchmark calculated exclusively from inner validation folds.",
                "tab:inner_benchmark",
            )

        make_table_1(outer_metrics_df, pooled, ci, tables_dir)
        make_ablation_table(
            inner_fold_metrics_df,
            inner_predictions_df,
            tables_dir,
        )
        plot_outer_performance(outer_predictions_df, figures_dir)
        make_gradcam_composite(run_dir, figures_dir)
        write_manuscript_results(outer_metrics_df, pooled, ci, run_dir)
        write_json(
            run_dir / "completion_manifest.json",
            {
                "status": "complete",
                "completed": datetime.now().isoformat(),
                "elapsed_seconds": time.time() - started,
                "outer_test_predictions": len(outer_predictions_df),
                "outer_test_patients": outer_predictions_df["patient_id"].nunique(),
                "outer_folds": args.outer_folds,
                "inner_folds": args.inner_folds,
                "configurations": len(configs),
                "outer_test_used_for_selection": False,
            },
        )
        print(f"Nested cross-validation complete: {run_dir}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    main()
