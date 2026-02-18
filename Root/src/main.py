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
from sklearn.model_selection import train_test_split
from utils.xai_utils import run_gradcam_on_balanced_samples
import torch
from torchsummary import summary
from PIL import Image
import pandas as pd
import numpy as np
import time
import os
import re





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
    g = torch.Generator()
    g.manual_seed(0)
    # Load the CSVs
    df_final = build_classification_dataset("/data2/users/koushani/chbmit/data/allpatients_resized")
    total_images = len(df_final)
    print(f"Total number of data points (images): {total_images}")
    print(df_final.columns)
    label_mapping = {'non_CME': 0, 'CME': 1}
    df_final['label'] = df_final['label'].map(label_mapping)
    print("Label column type:", df_final['label'].dtype)
    print("Unique values in 'label':", df_final['label'].unique())
    print("Value counts:\n", df_final['label'].value_counts())


    
    # row = df_final.iloc[idx]
    # img_path = row["image_path"]
    # label = row["label"]

    # # === Step 1: Load original image (no transform)
    # original_img = Image.open(img_path)

    # # Show original
    # plt.imshow(original_img)
    # plt.title(f"Original Image — Label: {label}")
    # plt.axis("off")
    # plt.savefig(f"sample_{idx}_original_image.png", dpi=300)
    # plt.close()
    
    # Split by label
    df_label_0 = df_final[df_final['label'] == 0]
    df_label_1 = df_final[df_final['label'] == 1]

    print("Label 0 count:", len(df_label_0))
    print("Label 1 count:", len(df_label_1))

    # Sample 15 from label 0 and 5 from label 1
    df_test_0 = df_label_0.sample(n=40, random_state=42)
    df_test_1 = df_label_1.sample(n=20, random_state=42)


    # Combine and form test set
    df_test = pd.concat([df_test_0, df_test_1])
    df_train = df_final.drop(df_test.index)

    print("Train:", len(df_train), df_train["label"].value_counts(normalize=True))
    print("Test: ", len(df_test),  df_test["label"].value_counts(normalize=True))

    # Save to CSV
    df_train.to_csv("train.csv", index=False)
    df_test.to_csv("test.csv", index=False)
    

    # Print label distributions
    print("Train label distribution:")
    print(df_train["label"].value_counts())

    print("\n Test label distribution:")
    print(df_test["label"].value_counts())
    print("\n Validation label distribution:")
   

    # === Step 2: Define transform
    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.289], std=[0.146])
])


    train_dataset = EyeFFEDataset(df_train, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    test_dataset = EyeFFEDataset(df_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=4)


    # Usage
    train_label_distribution = count_labels_in_dataloader(train_loader)
    print("Label distribution in training dataloader:")
    for label, count in sorted(train_label_distribution.items()):
        print(f"Label {label}: {count} samples")

        images_train, labels_train = next(iter(train_loader))
        print(f"Image batch shape: {images_train.shape}")
        print(f"Label batch shape: {labels_train.shape}")


    test_label_distribution = count_labels_in_dataloader(test_loader)
    print("Label distribution in testing dataloader:")
    for label, count in sorted(test_label_distribution.items()):
        print(f"Label {label}: {count} samples")

        images_test, labels_test = next(iter(test_loader))
        print(f"Image batch shape: {images_test.shape}")
        print(f"Label batch shape: {labels_test.shape}")

    


    # images_test, labels_test = next(iter(dataloader))
    # print(f"Image batch shape: {images_test.shape}")
    # print(f"Label batch shape: {labels_test.shape}")

   # Define your target batch and sample index
    target_batch_idx = 3     # e.g., 3rd batch
    sample_idx_within_batch = 5 # e.g., 6th sample in that batch

    # Create the iterator
    dataloader_iter = iter(train_loader)

    # Iterate up to the target batch
    for current_batch_idx in range(target_batch_idx + 1):
        images, labels = next(dataloader_iter)

    # Now pick the specific sample
    image = images[sample_idx_within_batch]
    label = labels[sample_idx_within_batch]

    print(f"Selected from batch {target_batch_idx}, sample {sample_idx_within_batch}")
    print(f"Image shape: {image.shape}, Label: {label}")
    print(f"Min: {image.min().item():.4f}, Max: {image.max().item():.4f}")

    # CHECKPOINT_PATH = "/data2/users/koushani/chbmit/Root/src/checkpoints/best_model.pth"

    model = build_resnet_for_grayscale(num_classes=2)  # or use unmodified ResNet if using Grayscale(3)

    # ckpt = load_checkpoint(CHECKPOINT_PATH, model, optimizer=None, map_location=device)

    model = model.to(device)
    channels = 1
    H = 320
    W = 320
    # Now log it
    input_size=(channels, H, W)
    summary(model, input_size=(1, 320, 320), batch_size=1, device="cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    out_dir = "/data2/users/koushani/chbmit/Root/plots"
    os.makedirs(out_dir, exist_ok=True)
    save_dir = "/data2/users/koushani/chbmit/github/RV_CME_ML/Root/src/checkpoint_dir"
    os.makedirs(save_dir, exist_ok=True)

    best_ckpt_path = os.path.join(save_dir, "best_model.pth")  # <-- file inside dir

    # for epoch in range(1, num_epochs + 1):
    #     train_loss, train_acc = train(
    #     model,
    #     device,
    #     train_loader,
    #     optimizer,
    #     epoch,
    #     log_interval=10,
    #     val_loader=test_loader,
    #     validate_every_epochs=5,       # e.g., validate every 5 epochs
    #     val_threshold=0.5,
    #     early_stopping_on_loss=True,   # now actually “early stopping on monitor_metric”
    #     es_patience=10,
    #     es_min_delta=1e-4,
    #     checkpoint_path=best_ckpt_path,
    #     monitor_metric="accuracy",     # or "auroc" if you want to chase AUROC instead
    # )

    #     if hasattr(model, "_stop_training") and model._stop_training:
    #         print("Early stopping triggered — breaking training loop.")
    #         break

    # ===== Load BEST model and compute final stats =====
    if os.path.exists(best_ckpt_path):
        print(f"\nLoading best model from {best_ckpt_path} for final evaluation...")
        state_dict = torch.load(best_ckpt_path, map_location=device)  # weights_only=True by default in 2.6
        model.load_state_dict(state_dict)

        # Evaluate on your validation/test loader using this best model
        final_metrics = validate(model, device, test_loader, threshold=0.5)

        print("\n=== FINAL METRICS (Best Validation Accuracy Model) ===")
        for k in ["loss", "accuracy", "auroc", "auprc", "sensitivity", "specificity", "f1"]:
            if k in final_metrics:
                print(f"  {k:12s}: {final_metrics[k]:.4f}")
        print("=====================================================\n")
    else:
        print("No best checkpoint found — was validation ever run?")

    
    XAI_ROOT = "/data2/users/koushani/chbmit/github/RV_CME_ML/Root/"

    # make sure model is loaded with best weights and eval mode
    model.eval()

    xai_report = run_gradcam_on_balanced_samples(
        df=df_test,
        model=model,
        device=device,
        transform=transform,
        xai_root_dir=XAI_ROOT,
        n_per_class=5,
        seed=42,
    )
    print(xai_report[["sample_id", "true_label", "pred_idx", "prob_CME", "overlay_path", "comparison_path"]])
# run_gradcam_single(
#     model, device, sample_path, transform,
#     out_path="/.../gradcam_overlay_force_CME.png",
#     class_idx=1  # force CME
# )



    


if __name__ == "__main__":
    main()