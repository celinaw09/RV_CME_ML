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


def main():
    # --------------------------------------------------

    # Reproducibility

    # --------------------------------------------------

    g = torch.Generator()

    g.manual_seed(0)

    torch.manual_seed(0)

    np.random.seed(0)

    # --------------------------------------------------

    # Experiment Folder

    # --------------------------------------------------

    timestamp = datetime.now().strftime(

        "%Y%m%d_%H%M%S"

    )

    RUN_DIR = (

        f"experiment_logs/run_{timestamp}"

    )

    os.makedirs(

        RUN_DIR,

        exist_ok=True

    )

    # --------------------------------------------------

    # Log ALL console output

    # --------------------------------------------------

    log_path = os.path.join(

        RUN_DIR,

        "console_output.txt"

    )

    log_file = open(

        log_path,

        "w"

    )

    sys.stdout = Tee(

        sys.stdout,

        log_file

    )

    sys.stderr = Tee(

        sys.stderr,

        log_file

    )

    print(

        "=" * 60

    )

    print(

        f"Experiment folder: {RUN_DIR}"

    )

    print(

        f"Started at: {timestamp}"

    )

    print(

        f"PyTorch version: {torch.__version__}"

    )

    print(

        "=" * 60

    )




    # --------------------------------------------------
    # Build dataset
    # --------------------------------------------------


    root_dir = "/data2/users/koushani/chbmit/data/allpatients_resized"

    num_patient_folders = 0

    for label in os.listdir(root_dir):

        label_path = os.path.join(root_dir, label)

        if not os.path.isdir(label_path):

            continue

        for patient in os.listdir(label_path):

            patient_path = os.path.join(

                label_path,

                patient

            )

            if os.path.isdir(patient_path):

                num_patient_folders += 1

    print(

        "Patient folders:",

        num_patient_folders

    )

    

    df_final = build_classification_dataset(
        "/data2/users/koushani/chbmit/data/allpatients_resized"
    )

    print("\n==================================================")
    print("FULL DATASET SUMMARY")
    print("==================================================")

    print(
        f"Total images: {len(df_final)}"
    )

    print(
        f"Total patients: "
        f"{df_final['patient_id'].nunique()}"
    )

    print("\nRaw labels:")

    print(
        df_final["label"].value_counts()
    )



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

    patient_image_counts = (
        df_final
        .groupby("patient_id")
        .size()
    )

    print(
        patient_image_counts
        .value_counts()
        .sort_index()
    )

    print("\nPatients not having exactly 2 images:")

    for patient, count in patient_image_counts.items():

        if count != 2:

            print(
                patient,
                count
            )

    total_images = len(df_final)

    print(
        f"Total number of data points (images): "
        f"{total_images}"
    )


    # --------------------------------------------------
    # Convert labels to integers
    # --------------------------------------------------

    label_mapping = {
        "non_CME": 0,
        "CME": 1
    }

    df_final["label"] = (
        df_final["label"]
        .map(label_mapping)
    )

    print("\nEncoded labels:")

    print(
        df_final["label"]
        .value_counts()
    )

    print(
        "\nUnique label values:",
        sorted(df_final["label"].unique())
    )

    # --------------------------------------------------
    # Patient-level stratified split
    # --------------------------------------------------

    folds = run_5_fold_cross_validation(

    df_final,

    patient_col="patient_id",

    label_col="label",

    n_splits=5,

    val_fraction=0.15,

    random_state=42

    )

    print(

        f"\nCreated {len(folds)} folds"

    )


    summary_rows = []

    for fold in folds:

        fold_dir = os.path.join(
            RUN_DIR,
            f"fold_{fold['fold']}"
        )

        os.makedirs(
            fold_dir,
            exist_ok=True
        )

        fold["fold_dir"] = fold_dir

        # -------------------
        # Save splits
        # -------------------

        fold["train_df"].to_csv(
            os.path.join(
                fold_dir,
                "train.csv"
            ),
            index=False
        )

        fold["val_df"].to_csv(
            os.path.join(
                fold_dir,
                "val.csv"
            ),
            index=False
        )

        fold["test_df"].to_csv(
            os.path.join(
                fold_dir,
                "test.csv"
            ),
            index=False
        )

        # -------------------
        # Summary
        # -------------------

        summary_rows.append({

            "fold":
                fold["fold"],

            "train_images":
                len(fold["train_df"]),

            "val_images":
                len(fold["val_df"]),

            "test_images":
                len(fold["test_df"]),

            "train_patients":
                fold["train_df"][
                    "patient_id"
                ].nunique(),

            "val_patients":
                fold["val_df"][
                    "patient_id"
                ].nunique(),

            "test_patients":
                fold["test_df"][
                    "patient_id"
                ].nunique()
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        os.path.join(
            RUN_DIR,
            "fold_summary.csv"
        ),
        index=False
    )

    print(

        "\nSaved fold_summary.csv"

    )

    cv_results = []
    all_fold_probs = []
    all_fold_targets = []

    for fold in folds:

        fold_num = fold["fold"]

        fold_dir = fold["fold_dir"]

        print("\n" + "=" * 60)
        print(f"TRAINING FOLD {fold_num}")
        print("=" * 60)

        df_train = fold["train_df"]
        df_val = fold["val_df"]
        df_test = fold["test_df"]
   

        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((320,320)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.289],
                std=[0.146]
            )
        ])

        eval_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.289],
                std=[0.146]
            )
        ])


        train_dataset = EyeFFEDataset(df_train,transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

        test_dataset = EyeFFEDataset(df_test, transform=eval_transform)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

        val_dataset = EyeFFEDataset(df_val, transform=eval_transform)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)


        model = build_resnet_for_grayscale(num_classes=2)  # or use unmodified ResNet if using Grayscale(3)


        # Freeze everything
        # Freeze everything
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze layer4
        for param in model.layer4.parameters():
            param.requires_grad = True

        # Unfreeze classifier
        for param in model.fc.parameters():
            param.requires_grad = True

        # Count trainable params
        trainable = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        total = sum(
            p.numel()
            for p in model.parameters()
        )



        print("\n=== TRAINABLE PARAMETERS ===")

        for name, param in model.named_parameters():

            print(f"{name:50s} {param.requires_grad}")

        print(f"Trainable params: {trainable:,}")
        print(f"Total params: {total:,}")
        print(f"Percent trainable: {100*trainable/total:.2f}%")

        # ckpt = load_checkpoint(CHECKPOINT_PATH, model, optimizer=None, map_location=device)

        model = model.to(device)
        if fold_num == 1:
            summary(
                model,
                input_size=(1,320,320),
                batch_size=1,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        num_epochs = 30


        train_counts = df_train["label"].value_counts()

        num_non_cme = train_counts[0]

        num_cme = train_counts[1]

        class_weights = torch.tensor(

            [

                len(df_train)/(2*num_non_cme),

                len(df_train)/(2*num_cme)

            ],

            dtype=torch.float32

        ).to(device)

        print("Class weights:", class_weights)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )




        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=5e-5,
            weight_decay=1e-3
        )

        trainable_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        total_params = sum(
            p.numel()
            for p in model.parameters()
        )

        print(
            f"Trainable params: "
            f"{trainable_params:,} / {total_params:,}"
        )

        out_dir = os.path.join(
            fold_dir,
            "plots"
        )

        os.makedirs(
            out_dir,
            exist_ok=True
        )
        save_dir = fold_dir

        best_ckpt_path = os.path.join(
            fold_dir,
            "best_model.pth"
        )


        print(
            f"\nFold {fold_num}: "
            f"{len(df_train)} train, "
            f"{len(df_val)} val, "
            f"{len(df_test)} test images"
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
            validate_every_epochs=1,       # e.g., validate every 5 epochs
            val_threshold=0.5,
            early_stopping_on_loss=True,   # now actually “early stopping on monitor_metric”
            es_patience=10,
            es_min_delta=1e-4,
            checkpoint_path=best_ckpt_path,
            monitor_metric="auroc",     # or "auroc" if you want to chase AUROC instead
        )

            if hasattr(model, "_stop_training") and model._stop_training:
                print("Early stopping triggered — breaking training loop.")
                break

        # ===== Load BEST model and compute final stats =====
        if os.path.exists(best_ckpt_path):
            print(f"\nLoading best model from {best_ckpt_path} for final evaluation...")
            ckpt = torch.load(
                best_ckpt_path,
                map_location=device,
                weights_only=False
            )

            model.load_state_dict(

                ckpt["model_state_dict"]

            )

            best_threshold = 0.5

            print(

                f"Using validation threshold "

                f"{best_threshold:.4f}"

            )

            

            print(
                f"Using validation threshold "
                f"{best_threshold:.4f}"
            )

            final_metrics = validate(
                model,
                device,
                criterion,
                test_loader,
                threshold=best_threshold,
                optimize_threshold=False
            )

            # ADD BELOW

            targets, probs = get_probs_and_targets(

                model,

                device,

                test_loader

            )


            print(
                f"Fold {fold_num} Results:"
            )

            print(
                f"Accuracy    : {final_metrics['accuracy']:.4f}"
            )

            print(
                f"AUROC       : {final_metrics['auroc']:.4f}"
            )

            print(
                f"AUPRC       : {final_metrics['auprc']:.4f}"
            )


            fold_metrics = {
                "fold": fold_num,
                **final_metrics
            }

            pd.DataFrame([fold_metrics]).to_csv(
                os.path.join(
                    fold_dir,
                    "test_metrics.csv"
                ),
                index=False
            )


            cv_results.append({
                "fold": fold_num,
                "accuracy": final_metrics["accuracy"],
                "auroc": final_metrics["auroc"],
                "auprc": final_metrics["auprc"],
                "sensitivity": final_metrics["sensitivity"],
                "specificity": final_metrics["specificity"],
                "f1": final_metrics["f1"]
            })

            all_fold_probs.append(probs)
            all_fold_targets.append(targets)

            pd.DataFrame(cv_results).to_csv(
                os.path.join(
                    RUN_DIR,
                    "cross_validation_results.csv"
                ),
                index=False
            )

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
        threshold=0.5
    )






    results_df = pd.DataFrame(cv_results)

    print("\n")
    print("=" * 60)
    print("5-FOLD CROSS VALIDATION RESULTS")
    print("=" * 60)

    for metric in [
        "accuracy",
        "auroc",
        "auprc",
        "sensitivity",
        "specificity",
        "f1"
    ]:
        mean = results_df[metric].mean()
        std = results_df[metric].std()

        print(
            f"{metric}: "
            f"{mean:.4f} ± {std:.4f}"
        )

    results_df.to_csv(
        os.path.join(
            RUN_DIR,
            "cross_validation_results.csv"
        ),
        index=False
    )

    cv_summary = {}

    for metric in [
        "accuracy",
        "auroc",
        "auprc",
        "sensitivity",
        "specificity",
        "f1"
    ]:
        cv_summary[metric] = {
            "mean": float(results_df[metric].mean()),
            "std": float(results_df[metric].std())
        }

    with open(
        os.path.join(
            RUN_DIR,
            "cv_summary.json"
        ),
        "w"
    ) as f:
        json.dump(
            cv_summary,
            f,
            indent=4
        )

    
    # XAI_ROOT = "/data2/users/koushani/chbmit/repo_reset/RV_CME_ML/Root"

    # make sure model is loaded with best weights and eval mode
#     model.eval()

#     for category in ["TP", "TN", "FP", "FN"]:
#         report = run_gradcam_on_category(
#             df=df_test,
#             model=model,
#             device=device,
#             transform=transform,
#             xai_root_dir=XAI_ROOT,
#             category=category,
#             n_samples=5,
#             seed=42,
#             threshold=0.5
#         )

#         if isinstance(report, pd.DataFrame) and len(report) > 0:
#             print(report[[
#                 "sample_id",
#                 "category",
#                 "true_label",
#                 "pred_idx_thresh",
#                 "prob_CME",
#                 "pair_png_path",
#                 "triple_pdf_path"
#             ]])

#             report_csv = os.path.join(XAI_ROOT, "xai_plots", category, f"xai_report_{category}.csv")
#             out_pdf = os.path.join(XAI_ROOT, "xai_plots", category, f"{category}_grid.pdf")

#             make_category_grid_figure(
#                 report_csv=report_csv,
#                 out_pdf=out_pdf,
#                 n_rows=5,
#                 seed=42,
#                 category_name=category
#             )

#     scores_df, summary_df = analyze_cam_similarity(
#     xai_root_dir=XAI_ROOT,
#     categories=("TP", "TN", "FP", "FN"),
#     out_filename_prefix="cam_similarity",
#     make_boxplot=True,
# )

#     print("\n=== CAM SIMILARITY SUMMARY ===")
#     print(summary_df)
#     print("================================\n")


    


    


if __name__ == "__main__":
    main()