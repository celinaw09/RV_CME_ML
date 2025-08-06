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
import pandas as pd

import os
import re

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