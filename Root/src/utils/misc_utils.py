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
import os
import re

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split
)
from PIL import Image


def evaluate_cross_validation_results(
    all_fold_targets,
    all_fold_probs,
    run_dir,
    threshold=0.5
):
    """
    all_fold_targets : list of arrays
    all_fold_probs   : list of arrays (N,2)
    run_dir          : output directory
    """

    # =====================================
    # CONCATENATE ALL FOLDS
    # =====================================

    targets = np.concatenate(
        all_fold_targets
    )

    probs = np.concatenate(
        all_fold_probs
    )

    print("targets:", targets.shape)
    print("probs:", probs.shape)

    probs_pos = probs[:, 1]

    if targets.ndim == 2:
        targets = np.argmax(
            targets,
            axis=1
        )

    # =====================================
    # ROC CURVE
    # =====================================

    fpr, tpr, _ = roc_curve(
        targets,
        probs_pos
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(figsize=(6, 6))

    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"AUC = {roc_auc:.3f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "5-Fold Cross-Validated ROC"
    )

    plt.legend()

    plt.savefig(
        os.path.join(
            run_dir,
            "cv_roc_curve.pdf"
        ),
        bbox_inches="tight"
    )

    plt.close()

    # =====================================
    # PR CURVE
    # =====================================

    precision, recall, _ = precision_recall_curve(
        targets,
        probs_pos
    )

    ap = average_precision_score(
        targets,
        probs_pos
    )

    plt.figure(figsize=(6, 6))

    plt.plot(
        recall,
        precision,
        lw=2,
        label=f"AP = {ap:.3f}"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title(
        "5-Fold Cross-Validated PR Curve"
    )

    plt.legend()

    plt.savefig(
        os.path.join(
            run_dir,
            "cv_pr_curve.pdf"
        ),
        bbox_inches="tight"
    )

    plt.close()

    # =====================================
    # CONFUSION MATRIX
    # =====================================

    preds = (
        probs_pos >= threshold
    ).astype(int)

    cm = confusion_matrix(
        targets,
        preds
    )

    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    im = ax.imshow(
        cm,
        cmap=plt.cm.Blues
    )

    plt.colorbar(
        im,
        ax=ax
    )

    classes = [
        "non-CME",
        "CME"
    ]

    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Confusion Matrix (threshold = {threshold})"
    )

    thresh = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color=(
                    "white"
                    if cm[i, j] > thresh
                    else "black"
                )
            )

    fig.tight_layout()

    png_path = os.path.join(
        run_dir,
        "cv_confusion_matrix.png"
    )

    pdf_path = os.path.join(
        run_dir,
        "cv_confusion_matrix.pdf"
    )

    plt.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

    img = Image.open(
        png_path
    ).convert("RGB")

    img.save(
        pdf_path,
        "PDF",
        resolution=600
    )

    # =====================================
    # POOLED METRICS
    # =====================================

    pooled_metrics = {

        "accuracy":
            accuracy_score(
                targets,
                preds
            ),

        "sensitivity":
            recall_score(
                targets,
                preds
            ),

        "specificity":
            tn / (tn + fp),

        "f1":
            f1_score(
                targets,
                preds
            ),

        "auroc":
            roc_auc,

        "auprc":
            ap,

        "tp":
            int(tp),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn)
    }

    print("\n=== POOLED CV METRICS ===")

    for k, v in pooled_metrics.items():

        print(
            f"{k}: {v}"
        )

    print(
        "=========================\n"
    )

    return pooled_metrics



def run_5_fold_cross_validation(
    df,
    patient_col="patient_id",
    label_col="label",
    n_splits=5,
    val_fraction=0.15,
    random_state=42
):
    """
    Patient-level stratified 5-fold CV.

    Returns
    -------
    folds : list

        folds[i] = {

            "train_df": ...,

            "val_df": ...,

            "test_df": ...,

            "train_patients": ...,

            "val_patients": ...,

            "test_patients": ...

        }
    """

    # ------------------------------------------
    # One row per patient
    # ------------------------------------------

    patient_df = (
        df
        .groupby(patient_col)[label_col]
        .max()
        .reset_index()
        .rename(
            columns={
                label_col:
                "patient_label"
            }
        )
    )

    print("\n====================================")
    print("5-FOLD CROSS VALIDATION")
    print("====================================")

    print(
        f"Patients: "
        f"{len(patient_df)}"
    )

    print(
        patient_df[
            "patient_label"
        ].value_counts()
    )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    folds = []

    # ------------------------------------------
    # Fold loop
    # ------------------------------------------

    for fold_idx, (
        train_val_idx,
        test_idx
    ) in enumerate(

        skf.split(
            patient_df[patient_col],
            patient_df["patient_label"]
        ),

        start=1
    ):

        print("\n")
        print("=" * 60)
        print(f"FOLD {fold_idx}")
        print("=" * 60)

        # ----------------------------------
        # Fold test patients
        # ----------------------------------

        test_patients = (
            patient_df
            .iloc[test_idx]
        )

        train_val_patients = (
            patient_df
            .iloc[train_val_idx]
        )

        # ----------------------------------
        # Validation split
        # ----------------------------------

        train_patients, val_patients = (
            train_test_split(
                train_val_patients,
                test_size=val_fraction,
                stratify=train_val_patients[
                    "patient_label"
                ],
                random_state=random_state
            )
        )

        # ----------------------------------
        # Convert patients -> images
        # ----------------------------------

        df_train = df[
            df[patient_col].isin(
                train_patients[
                    patient_col
                ]
            )
        ]

        df_val = df[
            df[patient_col].isin(
                val_patients[
                    patient_col
                ]
            )
        ]

        df_test = df[
            df[patient_col].isin(
                test_patients[
                    patient_col
                ]
            )
        ]

        # ----------------------------------
        # Leakage checks
        # ----------------------------------

        train_set = set(
            train_patients[
                patient_col
            ]
        )

        val_set = set(
            val_patients[
                patient_col
            ]
        )

        test_set = set(
            test_patients[
                patient_col
            ]
        )

        assert len(
            train_set & val_set
        ) == 0

        assert len(
            train_set & test_set
        ) == 0

        assert len(
            val_set & test_set
        ) == 0

        print(
            f"Train patients: "
            f"{len(train_set)}"
        )

        print(
            f"Val patients: "
            f"{len(val_set)}"
        )

        print(
            f"Test patients: "
            f"{len(test_set)}"
        )

        print()

        print(
            f"Train images: "
            f"{len(df_train)}"
        )

        print(
            f"Val images: "
            f"{len(df_val)}"
        )

        print(
            f"Test images: "
            f"{len(df_test)}"
        )

        print()

        print(
            "Train labels:"
        )

        print(
            df_train[
                label_col
            ].value_counts()
        )

        print(
            "Val labels:"
        )

        print(
            df_val[
                label_col
            ].value_counts()
        )

        print(
            "Test labels:"
        )

        print(
            df_test[
                label_col
            ].value_counts()
        )

        folds.append(
            {
                "fold": fold_idx,

                "train_df":
                    df_train,

                "val_df":
                    df_val,

                "test_df":
                    df_test,

                "train_patients":
                    train_patients,

                "val_patients":
                    val_patients,

                "test_patients":
                    test_patients
            }
        )

    return folds


def stratified_patient_split(
    df,
    patient_col="patient_id",
    label_col="label",
    train_size=0.70,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
):
    """
    Patient-level stratified split.

    Ensures:

    - No patient leakage
    - Both eyes remain together
    - CME/non-CME balance preserved
    - Separate train/val/test sets

    Returns
    -------
    df_train
    df_val
    df_test
    """

    assert abs(
        train_size + val_size + test_size - 1.0
    ) < 1e-9

    # ----------------------------------
    # Create patient-level table
    # ----------------------------------

    patient_df = (
        df.groupby(patient_col)[label_col]
        .max()
        .reset_index()
        .rename(
            columns={
                label_col: "patient_label"
            }
        )
    )

    print(
        f"\nUnique patients: "
        f"{len(patient_df)}"
    )

    print(
        "\nPatient-level labels:"
    )

    print(
        patient_df["patient_label"]
        .value_counts()
    )

    # ----------------------------------
    # Train split
    # ----------------------------------

    train_patients, temp_patients = train_test_split(
        patient_df,
        test_size=(1.0 - train_size),
        stratify=patient_df["patient_label"],
        random_state=random_state,
    )

    # ----------------------------------
    # Validation / Test split
    # ----------------------------------

    val_fraction_of_temp = (
        val_size / (val_size + test_size)
    )

    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=(1.0 - val_fraction_of_temp),
        stratify=temp_patients["patient_label"],
        random_state=random_state,
    )

    # ----------------------------------
    # Map patients back to image rows
    # ----------------------------------

    df_train = df[
        df[patient_col].isin(
            train_patients[patient_col]
        )
    ].copy()

    df_val = df[
        df[patient_col].isin(
            val_patients[patient_col]
        )
    ].copy()

    df_test = df[
        df[patient_col].isin(
            test_patients[patient_col]
        )
    ].copy()

    # ----------------------------------
    # Leakage checks
    # ----------------------------------

    train_ids = set(df_train[patient_col])
    val_ids = set(df_val[patient_col])
    test_ids = set(df_test[patient_col])

    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0

    print(
        "\nLeakage check passed."
    )

    print(
        "No patient overlap between "
        "train / val / test."
    )

    # ----------------------------------
    # Summary statistics
    # ----------------------------------

    print("\nImage Counts")
    print(
        f"Train: {len(df_train)}"
    )
    print(
        f"Val  : {len(df_val)}"
    )
    print(
        f"Test : {len(df_test)}"
    )

    print("\nPatient Counts")
    print(
        f"Train: {len(train_ids)}"
    )
    print(
        f"Val  : {len(val_ids)}"
    )
    print(
        f"Test : {len(test_ids)}"
    )

    print("\nTrain Labels")
    print(
        df_train[label_col]
        .value_counts()
    )

    print("\nValidation Labels")
    print(
        df_val[label_col]
        .value_counts()
    )

    print("\nTest Labels")
    print(
        df_test[label_col]
        .value_counts()
    )

    return (
        df_train,
        df_val,
        df_test,
    )


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
    """
    Build image-level dataframe from directory structure.

    Expected structure:

    root_dir/
        CME/
            Patient_1/
                xxx_OD.jpg
                xxx_OS.jpg

        non_CME/
            Patient_2/
                xxx_OD.jpg
                xxx_OS.jpg

    Returns
    -------
    DataFrame with columns:

        patient_id
        eye
        image_path
        label
    """

    data = []

    for label in sorted(os.listdir(root_dir)):

        label_path = os.path.join(root_dir, label)

        if not os.path.isdir(label_path):
            continue

        for patient_folder in sorted(os.listdir(label_path)):

            patient_path = os.path.join(
                label_path,
                patient_folder
            )

            if not os.path.isdir(patient_path):
                continue

            for filename in sorted(os.listdir(patient_path)):

                if not filename.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):
                    continue

                eye = None

                if "OD" in filename.upper():
                    eye = "OD"

                elif "OS" in filename.upper():
                    eye = "OS"

                else:
                    continue

                data.append(
                    {
                        "patient_id": patient_folder,
                        "eye": eye,
                        "image_path": os.path.join(
                            patient_path,
                            filename
                        ),
                        "label": label,
                    }
                )

    df = pd.DataFrame(data)

    print("\nDataset Summary")
    print(f"Images   : {len(df)}")
    print(
        f"Patients : "
        f"{df['patient_id'].nunique()}"
    )

    print("\nLabel Distribution")
    print(df["label"].value_counts())

    return df


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

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )

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


def plot_roc_curve(targets, probs, save_path):

    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, _ = roc_curve(targets, probs)

    auc = roc_auc_score(targets, probs)

    fig, ax = plt.subplots(figsize=(6,5))

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {auc:.3f}"
    )

    ax.plot(
        [0,1],
        [0,1],
        linestyle="--"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")

    ax.legend()

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        save_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
        dpi=300
    )

    plt.close(fig)

    return auc


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