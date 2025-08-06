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
from utils.misc_utils import build_classification_dataset, resize_images
from dataset.dataset import EyeFFEDataset
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch
from PIL import Image
import pandas as pd
import numpy as np

import os
import re




def main():
    g = torch.Generator()
    g.manual_seed(0)
    # Load the CSVs
    df_final = build_classification_dataset("/data2/users/koushani/chbmit/data/allpatients_resized")
    total_images = len(df_final)
    print(f"Total number of data points (images): {total_images}")
    sample_idx = 10
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

    # === Step 2: Define transform
    transform = transforms.Compose([
        # transforms.Resize((320, 320)),  # or (224, 224) for ResNet
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])


    dataset = EyeFFEDataset(df_final, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    images_train, labels_train = next(iter(dataloader))
    print(f"Image batch shape: {images_train.shape}")
    print(f"Label batch shape: {labels_train.shape}")


    # images_test, labels_test = next(iter(dataloader))
    # print(f"Image batch shape: {images_test.shape}")
    # print(f"Label batch shape: {labels_test.shape}")

   # Define your target batch and sample index
    target_batch_idx = 3     # e.g., 3rd batch
    sample_idx_within_batch = 5 # e.g., 6th sample in that batch

    # Create the iterator
    dataloader_iter = iter(dataloader)

    # Iterate up to the target batch
    for current_batch_idx in range(target_batch_idx + 1):
        images, labels = next(dataloader_iter)

    # Now pick the specific sample
    image = images[sample_idx_within_batch]
    label = labels[sample_idx_within_batch]

    print(f"Selected from batch {target_batch_idx}, sample {sample_idx_within_batch}")
    print(f"Image shape: {image.shape}, Label: {label}")
    print(f"Min: {image.min().item():.4f}, Max: {image.max().item():.4f}")

    # Convert tensor to numpy array and move channel to last dimension
    image_np = image.permute(1, 2, 0).numpy()  # Now shape: [224, 224, 3]

    # Plot using matplotlib
    plt.imshow(image.squeeze(0), cmap='gray')  # squeeze channel for display
    plt.title(f"Sample idx: {sample_idx_within_batch} | Label: {label}")
    plt.axis('off')
    plt.savefig(f'Sample idx_{sample_idx_within_batch}_image_grayscale.png', dpi=300)
    plt.close()










    # idx = 10
    # row = df_final.iloc[idx]

    # print(row)

    # # Load and display the image
    # img_path = row['image_path']
    # label = row['label']

    # image = Image.open(img_path)

    # plt.imshow(image)
    # plt.title(f"Label: {label} | Eye: {row['eye']}")
    # plt.axis('off')
    # plt.savefig(f'Sample idx_{idx}_imagefromcsv.png', dpi=300)
    # plt.close()

    # # Convert to NumPy array
    # img_array = np.array(image)

    # # Print shape
    # print(f"Image shape: {img_array.shape}")  # (H, W, C)
    # # df_final.to_csv("faa_image_classification_dataset_final.csv", index=False)
   
    # # img_size = 224  # or 320 if that's your original image shape

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    # ])
    

    # df_train, df_test = train_test_split(df_final, test_size=0.1, stratify=df_final["label"], random_state=42)

    # df_train.to_csv("train.csv", index=False)
    # df_test.to_csv("test.csv", index=False)

    # train_dataset = EyeFFEDataset(csv_file="train.csv", transform=transform)
    # test_dataset = EyeFFEDataset(csv_file="test.csv", transform=transform)

    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # images_train, labels_train = next(iter(train_loader))
    # print(f"Image batch shape: {images_train.shape}")
    # print(f"Label batch shape: {labels_train.shape}")


    # images_test, labels_test = next(iter(test_loader))
    # print(f"Image batch shape: {images_test.shape}")
    # print(f"Label batch shape: {labels_test.shape}")

    # sample_idx = 12
    # images, labels = next(iter(train_loader))

    # # Pick sample at index 12
    # image = images[sample_idx]         # Tensor: [3, 224, 224]
    # label = labels[sample_idx]

    # print(f"Sample image shape: {image.shape}")  # Should be [3, 224, 224]

    # # Convert tensor to numpy array and move channel to last dimension
    # image_np = image.permute(1, 2, 0).numpy()  # Now shape: [224, 224, 3]

    # # Plot using matplotlib
    # plt.imshow(image_np)
    # plt.title(f"Sample idx: {sample_idx} | Label: {label}")
    # plt.axis('off')
    # plt.savefig(f'Sample idx_{sample_idx}_RGB_image.png', dpi=300)
    # plt.close()
    
   # why are all sample points having the same label? Is shuffling not happening?
   # # do we need to reshape 320X320 to 224X224 to run ResNet on it?
    


if __name__ == "__main__":
    main()