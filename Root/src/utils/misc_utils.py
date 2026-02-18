import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from model.model import Net , RNN_Net
from utils.train import train
from utils.test import test
import matplotlib.pyplot as plt
import cv2
from torchvision.models import ResNet18_Weights
from collections import Counter
import torchvision.models as models
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score,
)

import os
import re


from sklearn.model_selection import train_test_split

def stratified_patient_split(df, patient_col="patient_id", label_col="label",
                             train_size=0.8, val_size=0.1, test_size=0.1,
                             random_state=42):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9

    # ---- patient-level table ----
    patient_df = (
        df.groupby(patient_col)[label_col]
          .max()  # patient is positive if ANY eye is positive
          .reset_index()
          .rename(columns={label_col: "patient_label"})
    )

    # ---- split patients: train vs temp ----
    train_patients, temp_patients = train_test_split(
        patient_df,
        test_size=(1.0 - train_size),
        stratify=patient_df["patient_label"],
        random_state=random_state
    )

    # ---- split temp into val and test ----
    # temp is (val + test) fraction; we want val:test = val_size:test_size
    val_frac_of_temp = val_size / (val_size + test_size)

    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=(1.0 - val_frac_of_temp),
        stratify=temp_patients["patient_label"],
        random_state=random_state
    )

    # ---- map back to image-level rows ----
    df_train = df[df[patient_col].isin(train_patients[patient_col])]
    df_val   = df[df[patient_col].isin(val_patients[patient_col])]
    df_test  = df[df[patient_col].isin(test_patients[patient_col])]

    return df_train, df_val, df_test


def rename_patient_folders_with_underscore(root_dir, subdirs=["CME", "no CME"]):
    """
    Rename patient folders inside given subdirectories (e.g., CME, no CME)
    by replacing spaces in folder names with underscores.

    Args:
        root_dir (str): Path to the RV_images_final directory
        subdirs (list): List of subdirectory names (e.g., ["CME", "no CME"])
    """
    for label in subdirs:
        label_path = os.path.join(root_dir, label)

        if not os.path.isdir(label_path):
            print(f"[SKIP] Not a directory: {label_path}")
            continue

        for folder_name in os.listdir(label_path):
            old_path = os.path.join(label_path, folder_name)

            if not os.path.isdir(old_path):
                continue

            # Replace space with underscore
            if " " in folder_name:
                new_name = folder_name.replace(" ", "_")
                new_path = os.path.join(label_path, new_name)

                # Rename folder
                os.rename(old_path, new_path)
                print(f"[RENAME] {folder_name} → {new_name}")
            else:
                print(f"[OK] {folder_name} (no change)")


def rename_image_files_with_underscores(root_dir, subdirs=["CME", "no CME"]):
    """
    Go through each patient folder inside CME/no CME and rename image files
    by replacing spaces with underscores.

    Args:
        root_dir (str): Root path to RV_images_final
        subdirs (list): Subdirectories to process (e.g., ["CME", "no CME"])
    """
    for label in subdirs:
        label_path = os.path.join(root_dir, label)
        if not os.path.isdir(label_path):
            print(f"[SKIP] Not a directory: {label_path}")
            continue

        for patient_folder in os.listdir(label_path):
            patient_path = os.path.join(label_path, patient_folder)
            if not os.path.isdir(patient_path):
                continue

            for filename in os.listdir(patient_path):
                file_path = os.path.join(patient_path, filename)

                # Only process image files
                if not os.path.isfile(file_path) or not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                # Replace spaces with underscores
                if " " in filename:
                    new_name = filename.replace(" ", "_")
                    new_path = os.path.join(patient_path, new_name)

                    os.rename(file_path, new_path)
                    print(f"[RENAME] {filename} → {new_name}")
                else:
                    print(f"[OK] {filename} (no change)")


def show_eye_pair(patient_folder_path, save_dir=None):
    """
    Visualizes and optionally saves a pair of images (OD and OS) from a single patient folder.
    
    Args:
        patient_folder_path (str): Path to a patient's folder (e.g., ACB_OU)
        save_dir (str or None): Folder to save the output plot. If None, does not save.
    """
    # Find OD and OS images
    images = os.listdir(patient_folder_path)
    od_image = next((f for f in images if "OD" in f.upper()), None)
    os_image = next((f for f in images if "OS" in f.upper()), None)

    if not od_image or not os_image:
        print(f"[ERROR] Could not find both OD and OS images in {patient_folder_path}")
        return

    # Load images
    od_path = os.path.join(patient_folder_path, od_image)
    os_path = os.path.join(patient_folder_path, os_image)
    od_img = cv2.imread(od_path)
    os_img = cv2.imread(os_path)

    # Convert BGR to RGB
    od_img = cv2.cvtColor(od_img, cv2.COLOR_BGR2RGB)
    os_img = cv2.cvtColor(os_img, cv2.COLOR_BGR2RGB)

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(od_img)
    axs[0].set_title(f"Right Eye (OD)\n{od_image}")
    axs[0].axis('off')

    axs[1].imshow(os_img)
    axs[1].set_title(f"Left Eye (OS)\n{os_image}")
    axs[1].axis('off')

    patient_name = os.path.basename(patient_folder_path)
    plt.suptitle(f"Patient: {patient_name}", fontsize=14)
    plt.tight_layout()

    # Save figure if path is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{patient_name}_OD_OS_pair.png")
        plt.savefig(save_path, dpi=300)
        print(f"[SAVED] Plot saved to: {save_path}")

    plt.show()




def build_classification_dataset(root_dir):
    data = []

    for label in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label)
        if not os.path.isdir(label_path):
            continue

        for patient_folder in os.listdir(label_path):
            patient_path = os.path.join(label_path, patient_folder)
            if not os.path.isdir(patient_path):
                
                continue

            for filename in os.listdir(patient_path):
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                eye = None
                if "OD" in filename.upper():
                    eye = "OD"
                elif "OS" in filename.upper():
                    eye = "OS"
                else:
                    continue

                file_path = os.path.join(patient_path, filename)

                data.append({
                    "patient_id": patient_folder,
                    "eye": eye,
                    "image_path": file_path,
                    "label": label
                })

    return pd.DataFrame(data)


def resize_images(df, output_root, target_size=(350, 350)):
    os.makedirs(output_root, exist_ok=True)

    for _, row in df.iterrows():
        img = cv2.imread(row["image_path"])
        if img is None:
            print(f"[SKIP] Could not load {row['image_path']}")
            continue

        # Resize image
        resized_img = cv2.resize(img, target_size)

        # Create mirrored output path
        label = row["label"]
        patient_id = row["patient_id"]
        filename = os.path.basename(row["image_path"])

        output_folder = os.path.join(output_root, label, patient_id)
        os.makedirs(output_folder, exist_ok=True)

        save_path = os.path.join(output_folder, filename)

        # Check if file has valid image extension
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"[ERROR] Invalid file extension: {filename}")
            continue

        # Save resized image
        cv2.imwrite(save_path, resized_img)

    print(f"All images resized to {target_size} and saved to {output_root}")



def count_labels_in_dataloader(dataloader):
        label_counter = Counter()
        for _, labels in dataloader:
            label_counter.update(labels.tolist())
        return label_counter



def build_resnet_for_grayscale(num_classes=2):
    # Load pretrained ImageNet weights (new API)
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Modify first conv layer to accept 1 grayscale channel
    # NOTE: We copy and average RGB weights so we preserve pretrained knowledge
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # Initialize grayscale conv weights by averaging pretrained RGB weights
    new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)

    model.conv1 = new_conv

    # Replace the final FC layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model



def get_probs_and_targets_binary(model, device, dataloader):
    """
    Returns:
      probs: ndarray shape (N,)  -> P(class=1) (CME)
      targets: ndarray shape (N,) -> ground-truth labels in {0,1}
    """
    model.eval()
    probs_list = []
    targets_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)  # shape (B,2)
            prob_pos = torch.softmax(logits, dim=1)[:, 1]  # P(CME)

            probs_list.append(prob_pos.detach().cpu().numpy())
            targets_list.append(labels.detach().cpu().numpy())

    probs = np.concatenate(probs_list, axis=0)
    targets = np.concatenate(targets_list, axis=0).astype(int)
    return probs, targets


# -----------------------------
# 2) Compute AAO-style metrics at a fixed threshold
# -----------------------------
def compute_metrics_at_threshold(targets, probs, threshold=0.5):
    """
    Metrics aligned with your AAO abstract:
      - Accuracy
      - AUROC
      - AUPRC
      - Sensitivity (Recall for positive class)
      - Specificity (Recall for negative class)
      - F1
    """
    preds = (probs >= threshold).astype(int)

    # Confusion matrix: [[TN, FP],
    #                    [FN, TP]]
    cm = confusion_matrix(targets, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, zero_division=0)

    # Sensitivity = TP / (TP + FN)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUROC + AUPRC use probabilities (threshold-free)
    # Guard against edge cases where only one class exists in targets
    if len(np.unique(targets)) == 2:
        auroc = roc_auc_score(targets, probs)
        auprc = average_precision_score(targets, probs)
    else:
        auroc = float("nan")
        auprc = float("nan")

    return {
        "accuracy": accuracy,
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "cm": cm,
        "threshold": threshold,
    }


# -----------------------------
# 3) Plot Precision-Recall curve
# -----------------------------
def plot_pr_curve(targets, probs, save_path):
    precision, recall, _ = precision_recall_curve(targets, probs)

    ap = (
        average_precision_score(targets, probs)
        if len(np.unique(targets)) == 2
        else float("nan")
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        f"Precision–Recall Curve (AP = {ap:.3f})"
        if not np.isnan(ap)
        else "Precision–Recall Curve"
    )
    ax.grid(True, alpha=0.3)

    # Tight layout but controlled
    fig.tight_layout(pad=0.5)

    # Save as PDF with very small margins
    fig.savefig(
        save_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,   # 🔥 small margin
        dpi=300
    )

    plt.close(fig)
    return ap


# -----------------------------
# 4) Plot Confusion Matrix
# -----------------------------
def plot_confusion_matrix(cm, save_path, title="Confusion Matrix", labels=("non-CME", "CME")):
    fig, ax = plt.subplots(figsize=(6, 5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(labels)
    )

    disp.plot(
        ax=ax,
        cmap="Blues",          # 🔵 BLUE SHADES
        values_format="d",
        colorbar=True
    )

    # 🔑 Ensure the heatmap renders in PDF
    for artist in ax.get_children():
        if hasattr(artist, "set_rasterized"):
            artist.set_rasterized(True)

    ax.set_title(title)
    fig.tight_layout()

    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# 5) End-to-end: load checkpoint, compute metrics, save plots
# -----------------------------
def generate_prc_and_confusion(
    model,
    device,
    test_loader,
    checkpoint_path,
    out_dir="/data2/users/koushani/chbmit/Root/plots",
    threshold=0.5,
):
    os.makedirs(out_dir, exist_ok=True)

    # Load trained weights (NO retraining required)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Make sure test_loader is deterministic (recommended)
    # If you want reproducible plots/metrics, set shuffle=False in test_loader.
    probs, targets = get_probs_and_targets_binary(model, device, test_loader)

    # Metrics aligned to AAO abstract
    metrics = compute_metrics_at_threshold(targets, probs, threshold=threshold)

    # Save PR curve
    pr_path = os.path.join(out_dir, "pr_curve_test.pdf")
    ap = plot_pr_curve(targets, probs, pr_path)

    # Save Confusion Matrix
    cm_path = os.path.join(out_dir, "confusion_matrix_test.pdf")
    plot_confusion_matrix(
        metrics["cm"],
        cm_path,
        title=f"Confusion Matrix (threshold = {threshold})",
        labels=("non-CME", "CME"),
    )

    # Print metrics in the same style as your abstract
    print("\n=== METRICS (Aligned with AAO Abstract) ===")
    print(f"Accuracy     : {metrics['accuracy']:.4f}")
    print(f"AUROC        : {metrics['auroc']:.4f}")
    print(f"AUPRC (AP)   : {metrics['auprc']:.4f}")
    print(f"Sensitivity  : {metrics['sensitivity']:.4f}")
    print(f"Specificity  : {metrics['specificity']:.4f}")
    print(f"F1           : {metrics['f1']:.4f}")
    print("Confusion Matrix [[TN, FP],[FN, TP]]:")
    print(metrics["cm"])
    print("=========================================\n")

    print(f"Saved PR curve to: {pr_path}")
    print(f"Saved confusion matrix to: {cm_path}")

    return metrics