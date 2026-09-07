import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from nested_cv_pipeline import (
    PatientMILClassifier,
    assert_disjoint,
    calculate_metrics,
    dataset_manifest,
    derive_patient_id,
    hyperparameter_grid,
    load_dataset,
    make_loader,
    patient_splits,
    patient_table,
    resolve_primary_device,
    resolve_gpu_ids,
    subset_patients,
    youden_threshold,
)


def synthetic_frame():
    rows = []
    for patient in range(40):
        label = patient % 2
        for image in range(2):
            rows.append(
                {
                    "patient_id": f"p{patient:03d}",
                    "label": label,
                    "image_path": f"/unused/p{patient:03d}_{image}.png",
                }
            )
    return pd.DataFrame(rows)


def test_full_grid_is_prespecified_144_configurations():
    configs = hyperparameter_grid()
    assert len(configs) == 144
    assert len({config.config_id for config in configs}) == 144


def test_outer_and_inner_splits_are_patient_disjoint_and_stratified():
    frame = synthetic_frame()
    outer_splits = patient_splits(frame, n_splits=5, seed=42)
    test_patients = []
    for outer_fold, (development_ids, test_ids) in enumerate(outer_splits, start=1):
        development = subset_patients(frame, development_ids)
        test = subset_patients(frame, test_ids)
        assert_disjoint(development=development, test=test)
        assert set(test.groupby("patient_id")["label"].first()) == {0, 1}
        test_patients.extend(test_ids.tolist())
        inner_splits = patient_splits(development, n_splits=4, seed=outer_fold)
        for train_ids, validation_ids in inner_splits:
            train = subset_patients(development, train_ids)
            validation = subset_patients(development, validation_ids)
            assert_disjoint(train=train, validation=validation, outer_test=test)
    assert sorted(test_patients) == sorted(frame["patient_id"].unique())


def test_threshold_and_metrics_support_fold_specific_thresholds():
    targets = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])
    threshold = youden_threshold(targets, probabilities)
    metrics = calculate_metrics(targets, probabilities, threshold)
    assert metrics["auroc"] == 1.0
    assert metrics["accuracy"] == 1.0
    fold_thresholds = np.array([0.2, 0.5, 0.5, 0.8])
    pooled = calculate_metrics(targets, probabilities, fold_thresholds)
    assert pooled["accuracy"] == 1.0
    assert np.isnan(pooled["threshold"])


def test_cpu_gpu_request_always_selects_no_gpu():
    args = argparse.Namespace(
        gpus="cpu",
        max_gpus=3,
        min_free_memory_mb=2000,
    )
    selected, _ = resolve_gpu_ids(args)
    assert selected == []
    assert resolve_primary_device("cpu", selected).type == "cpu"


def test_mps_request_is_not_parsed_as_a_cuda_device_id():
    args = argparse.Namespace(
        gpus="mps",
        max_gpus=3,
        min_free_memory_mb=2000,
    )
    selected, _ = resolve_gpu_ids(args)
    assert selected == []


def test_dataset_manifest_fingerprints_exact_cohort(tmp_path):
    first = tmp_path / "CME" / "p001" / "p001_OD.png"
    second = tmp_path / "non_CME" / "p002" / "p002_OS.png"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"first-image")
    second.write_bytes(b"second-image")
    frame = pd.DataFrame(
        [
            {"patient_id": "p001", "folder_id": "p001", "eye": "OD", "image_path": str(first), "eye_label": 1, "label": 1},
            {"patient_id": "p002", "folder_id": "p002", "eye": "OS", "image_path": str(second), "eye_label": 0, "label": 0},
        ]
    )
    manifest = dataset_manifest(frame, str(tmp_path))
    assert manifest["images"] == 2
    assert manifest["patients"] == 2
    assert manifest["source_eye_image_counts"] == {"CME": 1, "non_CME": 1}
    assert manifest["patient_counts"] == {"CME": 1, "non_CME": 1}
    assert len(manifest["cohort_sha256"]) == 64
    assert len(manifest["files"]) == 2


def test_patient_label_is_positive_when_either_eye_has_cme(tmp_path):
    from PIL import Image

    cme = tmp_path / "CME" / "SyntheticPatient_OS" / "SyntheticPatient_OS.png"
    non_cme = tmp_path / "non_CME" / "SyntheticPatient_OD" / "SyntheticPatient_OD.png"
    cme.parent.mkdir(parents=True)
    non_cme.parent.mkdir(parents=True)
    Image.new("L", (8, 8), color=200).save(cme)
    Image.new("L", (8, 8), color=20).save(non_cme)

    frame = load_dataset(str(tmp_path))
    assert derive_patient_id("SyntheticPatient_OS") == "SyntheticPatient"
    assert frame["patient_id"].nunique() == 1
    assert set(frame["eye_label"]) == {0, 1}
    assert set(frame["label"]) == {1}
    assert patient_table(frame).iloc[0].to_dict() == {
        "patient_id": "SyntheticPatient",
        "label": 1,
    }


def test_mil_pooling_uses_maximum_eye_level_cme_evidence():
    class PixelBackbone(nn.Module):
        def forward(self, images):
            score = images[:, 0, 0, 0]
            return torch.stack((-score, score), dim=1)

    model = PatientMILClassifier(PixelBackbone())
    images = torch.tensor([[[[[-2.0]]], [[[3.0]]]], [[[[-2.0]]], [[[-3.0]]]]])
    mask = torch.tensor([[True, True], [True, True]])
    probabilities = torch.softmax(model(images, mask), dim=1)[:, 1]
    assert probabilities[0] > 0.99
    assert probabilities[1] < 0.05


def test_patient_loader_returns_one_variable_size_bag_per_patient(tmp_path):
    from PIL import Image

    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for patient_id, eyes, label in [("a", ["OD", "OS"], 1), ("b", ["OD"], 0)]:
        for eye in eyes:
            path = tmp_path / f"{patient_id}_{eye}.png"
            Image.new("L", (8, 8), color=100).save(path)
            rows.append(
                {
                    "patient_id": patient_id,
                    "folder_id": f"{patient_id}_{eye}",
                    "eye": eye,
                    "image_path": str(path),
                    "eye_label": label,
                    "label": label,
                }
            )
    frame = pd.DataFrame(rows)
    transform = lambda image: torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0)
    loader = make_loader(frame, transform, 2, 0, False, 42)
    images, mask, labels, patient_ids = next(iter(loader))
    assert images.shape == (2, 2, 1, 8, 8)
    assert mask.sum(dim=1).tolist() == [2, 1]
    assert labels.tolist() == [1, 0]
    assert patient_ids == ["a", "b"]
