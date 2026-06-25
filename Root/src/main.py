import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import ConfusionMatrixDisplay
from torch.optim.lr_scheduler import StepLR
from model.model import Net , RNN_Net
from utils.train import train, validate, plot_roc_curve, get_probs_and_targets
from utils.test import test
from utils.model_utils import *
import matplotlib.pyplot as plt
from utils.misc_utils import *
from dataset.dataset import EyeFFEDataset
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from utils.xai_utils import (
    run_gradcam_on_category,
    make_category_grid_figure,
    analyze_cam_similarity,
    analyze_mean_cam_centers,
)
import torch
import sys
from torchsummary import summary
from PIL import Image
import pandas as pd
import numpy as np
import time
import os
from PIL import Image
import re
import glob
import json
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    
)
from datetime import datetime



class Tee:

    def __init__(self, *files):

        self.files = files

    def write(self, obj):

        for f in self.files:

            f.write(obj)
            f.flush()

    def flush(self):

        for f in self.files:

            f.flush()


EXPERIMENT_ROOT = "experiment_logs"

EXPERIMENTS = [
    {
        "name": "FINAL_MODEL",
        "learning_rate": 1e-4,
        "image_size": 224,
        "augmentation": "basic",
    }
]


def reset_random_seed(seed: int = 0):
    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return g


def make_transforms(image_size, augmentation):
    if augmentation == "none":
        train_transforms = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.289], std=[0.146]),
        ]
    elif augmentation == "basic":
        train_transforms = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.289], std=[0.146]),
        ]
    elif augmentation == "advanced":
        train_transforms = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.289], std=[0.146]),
        ]
    else:
        raise ValueError(f"Unsupported augmentation: {augmentation}")

    eval_transforms = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.289], std=[0.146]),
    ]

    return transforms.Compose(train_transforms), transforms.Compose(eval_transforms)


def save_hyperparameters(run_dir, config):
    hparam_path = os.path.join(run_dir, "hyperparameters.json")
    with open(hparam_path, "w") as f:
        json.dump(config, f, indent=4)


print(torch.__version__)
gpu = 0
# Check if specified GPU is available, else default to CPU
if torch.cuda.is_available():
    try:
        device = torch.device(f"cuda:{gpu}")
        # Test if the specified GPU index is valid
        _ = torch.cuda.get_device_name(device)
    except AssertionError:
        print(f"GPU {gpu} is not available. Falling back to GPU 0.")
        device = torch.device("cuda:0")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")


def run_experiment(config):
    reset_random_seed(0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
    RUN_DIR = os.path.join(EXPERIMENT_ROOT, f"{config['name']}_{timestamp}")
    os.makedirs(RUN_DIR, exist_ok=True)

    save_hyperparameters(RUN_DIR, config)

    log_path = os.path.join(RUN_DIR, "console_output.txt")
    log_file = open(log_path, "w")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    try:
        print("=" * 60)
        print(f"Experiment folder: {RUN_DIR}")
        print(f"Experiment config: {config}")
        print(f"Started at: {timestamp}")
        print(f"PyTorch version: {torch.__version__}")
        print("=" * 60)

        root_dir = "/data2/users/koushani/chbmit/data/allpatients_resized"

        num_patient_folders = 0
        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            if not os.path.isdir(label_path):
                continue
            for patient in os.listdir(label_path):
                patient_path = os.path.join(label_path, patient)
                if os.path.isdir(patient_path):
                    num_patient_folders += 1

        print("Patient folders:", num_patient_folders)

        df_final = build_classification_dataset(root_dir)

        print("\n==================================================")
        print("FULL DATASET SUMMARY")
        print("==================================================")

        print(f"Total images: {len(df_final)}")
        print(f"Total patients: {df_final['patient_id'].nunique()}")
        print("\nRaw labels:")
        print(df_final["label"].value_counts())
        print(df_final.head())

        only_od = 0
        only_os = 0
        for patient, group in df_final.groupby("patient_id"):
            eyes = set(group["eye"])
            if eyes == {"OD"}:
                only_od += 1
            elif eyes == {"OS"}:
                only_os += 1

        print("Only OD:", only_od)
        print("Only OS:", only_os)
        print("\n===================================")
        print("PATIENT IMAGE COUNT ANALYSIS")
        print("===================================")

        patient_image_counts = df_final.groupby("patient_id").size()
        print(patient_image_counts.value_counts().sort_index())

        print("\nPatients not having exactly 2 images:")
        for patient, count in patient_image_counts.items():
            if count != 2:
                print(patient, count)

        total_images = len(df_final)
        print(f"Total number of data points (images): {total_images}")

        label_mapping = {"non_CME": 0, "CME": 1}
        df_final["label"] = df_final["label"].map(label_mapping)

        print("\nEncoded labels:")
        print(df_final["label"].value_counts())
        print("\nUnique label values:", sorted(df_final["label"].unique()))

        folds = run_5_fold_cross_validation(
            df_final,
            patient_col="patient_id",
            label_col="label",
            n_splits=5,
            val_fraction=0.15,
            random_state=42,
        )

        print(f"\nCreated {len(folds)} folds")

        summary_rows = []
        for fold in folds:
            fold_dir = os.path.join(RUN_DIR, f"fold_{fold['fold']}")
            os.makedirs(fold_dir, exist_ok=True)
            fold["fold_dir"] = fold_dir

            fold["train_df"].to_csv(os.path.join(fold_dir, "train.csv"), index=False)
            fold["val_df"].to_csv(os.path.join(fold_dir, "val.csv"), index=False)
            fold["test_df"].to_csv(os.path.join(fold_dir, "test.csv"), index=False)

            summary_rows.append({
                "fold": fold["fold"],
                "train_images": len(fold["train_df"]),
                "val_images": len(fold["val_df"]),
                "test_images": len(fold["test_df"]),
                "train_patients": fold["train_df"]["patient_id"].nunique(),
                "val_patients": fold["val_df"]["patient_id"].nunique(),
                "test_patients": fold["test_df"]["patient_id"].nunique(),
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(RUN_DIR, "fold_summary.csv"), index=False)
        print("\nSaved fold_summary.csv")

        cv_results = []
        all_fold_probs = []
        all_fold_targets = []

        train_transform, eval_transform = make_transforms(config["image_size"], config["augmentation"])

        for fold in folds:
            fold_num = fold["fold"]
            fold_dir = fold["fold_dir"]

            print("\n" + "=" * 60)
            print(f"TRAINING FOLD {fold_num}")
            print("=" * 60)

            df_train = fold["train_df"]
            df_val = fold["val_df"]
            df_test = fold["test_df"]

            train_dataset = EyeFFEDataset(df_train, transform=train_transform)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

            test_dataset = EyeFFEDataset(df_test, transform=eval_transform)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

            val_dataset = EyeFFEDataset(df_val, transform=eval_transform)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

            model = build_resnet_for_grayscale(num_classes=2)

            for param in model.parameters():
                param.requires_grad = False
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.fc.parameters():
                param.requires_grad = True

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())

            print("\n=== TRAINABLE PARAMETERS ===")
            for name, param in model.named_parameters():
                print(f"{name:50s} {param.requires_grad}")
            print(f"Trainable params: {trainable:,}")
            print(f"Total params: {total:,}")
            print(f"Percent trainable: {100 * trainable / total:.2f}%")

            model = model.to(device)
            if fold_num == 1:
                summary(
                    model,
                    input_size=(1, config["image_size"], config["image_size"]),
                    batch_size=1,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
            num_epochs = 30

            train_counts = df_train["label"].value_counts()
            num_non_cme = train_counts[0]
            num_cme = train_counts[1]
            class_weights = torch.tensor([
                len(df_train) / (2 * num_non_cme),
                len(df_train) / (2 * num_cme),
            ], dtype=torch.float32).to(device)

            print("Class weights:", class_weights)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config["learning_rate"],
                weight_decay=1e-3,
            )

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Trainable params: {trainable_params:,} / {total_params:,}")

            out_dir = os.path.join(fold_dir, "plots")
            os.makedirs(out_dir, exist_ok=True)
            best_ckpt_path = os.path.join(fold_dir, "best_model.pth")

            print(
                f"\nFold {fold_num}: {len(df_train)} train, {len(df_val)} val, {len(df_test)} test images"
            )
            for epoch in range(1, num_epochs + 1):
                train_loss, train_acc = train(
                    model,
                    device,
                    train_loader,
                    optimizer,
                    criterion,
                    epoch,
                    log_interval=10,
                    val_loader=val_loader,
                    validate_every_epochs=1,
                    val_threshold=0.5,
                    early_stopping_on_loss=True,
                    es_patience=10,
                    es_min_delta=1e-4,
                    checkpoint_path=best_ckpt_path,
                    monitor_metric="auroc",
                )
                if hasattr(model, "_stop_training") and model._stop_training:
                    print("Early stopping triggered — breaking training loop.")
                    break

            if os.path.exists(best_ckpt_path):
                print(f"\nLoading best model from {best_ckpt_path} for final evaluation...")
                ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                best_threshold = 0.5
                print(f"Using validation threshold {best_threshold:.4f}")
                final_metrics = validate(
                    model,
                    device,
                    criterion,
                    test_loader,
                    threshold=best_threshold,
                    optimize_threshold=False,
                )
                targets, probs = get_probs_and_targets(model, device, test_loader)

                print(f"Fold {fold_num} Results:")
                print(f"Accuracy    : {final_metrics['accuracy']:.4f}")
                print(f"AUROC       : {final_metrics['auroc']:.4f}")
                print(f"AUPRC       : {final_metrics['auprc']:.4f}")

                fold_metrics = {"fold": fold_num, **final_metrics}
                pd.DataFrame([fold_metrics]).to_csv(os.path.join(fold_dir, "test_metrics.csv"), index=False)

                cv_results.append(
                    {
                        "fold": fold_num,
                        "accuracy": final_metrics["accuracy"],
                        "auroc": final_metrics["auroc"],
                        "auprc": final_metrics["auprc"],
                        "sensitivity": final_metrics["sensitivity"],
                        "specificity": final_metrics["specificity"],
                        "f1": final_metrics["f1"],
                    }
                )
                all_fold_probs.append(probs)
                all_fold_targets.append(targets)

                pd.DataFrame(cv_results).to_csv(os.path.join(RUN_DIR, "cross_validation_results.csv"), index=False)

                print("\n=== FINAL METRICS (Best AUROC Model) ===")
                for k in ["loss", "accuracy", "auroc", "auprc", "sensitivity", "specificity", "f1"]:
                    if k in final_metrics:
                        print(f"  {k:12s}: {final_metrics[k]:.4f}")
                print("=====================================================\n")
            else:
                print("No best checkpoint found — was validation ever run?")

        print("\nCreating pooled CV ROC/PR curves...")

        pooled_metrics = evaluate_cross_validation_results(
            all_fold_targets=all_fold_targets,
            all_fold_probs=all_fold_probs,
            run_dir=RUN_DIR,
            threshold=0.5,
        )

        results_df = pd.DataFrame(cv_results)

        print("\n")
        print("=" * 60)
        print("5-FOLD CROSS VALIDATION RESULTS")
        print("=" * 60)

        for metric in ["accuracy", "auroc", "auprc", "sensitivity", "specificity", "f1"]:
            mean = results_df[metric].mean()
            std = results_df[metric].std()
            print(f"{metric}: {mean:.4f} ± {std:.4f}")

        results_df.to_csv(os.path.join(RUN_DIR, "cross_validation_results.csv"), index=False)

        cv_summary = {}
        for metric in ["accuracy", "auroc", "auprc", "sensitivity", "specificity", "f1"]:
            cv_summary[metric] = {"mean": float(results_df[metric].mean()), "std": float(results_df[metric].std())}

        with open(os.path.join(RUN_DIR, "cv_summary.json"), "w") as f:
            json.dump(cv_summary, f, indent=4)

        return {
            "experiment_name": config["name"],
            "experiment_dir": RUN_DIR,
            "learning_rate": config["learning_rate"],
            "image_size": config["image_size"],
            "augmentation": config["augmentation"],
            "accuracy_mean": float(results_df["accuracy"].mean()),
            "accuracy_std": float(results_df["accuracy"].std()),
            "auroc_mean": float(results_df["auroc"].mean()),
            "auroc_std": float(results_df["auroc"].std()),
            "auprc_mean": float(results_df["auprc"].mean()),
            "auprc_std": float(results_df["auprc"].std()),
            "sensitivity_mean": float(results_df["sensitivity"].mean()),
            "sensitivity_std": float(results_df["sensitivity"].std()),
            "specificity_mean": float(results_df["specificity"].mean()),
            "specificity_std": float(results_df["specificity"].std()),
            "f1_mean": float(results_df["f1"].mean()),
            "f1_std": float(results_df["f1"].std()),
        }
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def main():
    experiment_summaries = []
    for config in EXPERIMENTS:
        summary = run_experiment(config)
        if summary is not None:
            experiment_summaries.append(summary)

    if experiment_summaries:
        os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
        ablation_path = os.path.join(EXPERIMENT_ROOT, "ablation_summary_final.csv")
        pd.DataFrame(experiment_summaries).to_csv(ablation_path, index=False)
        print("\nAblation summary saved to:", ablation_path)


if __name__ == "__main__":
    main()