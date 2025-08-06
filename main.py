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
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from sklearn.model_selection import train_test_split
import torch
from torchsummary import summary
from PIL import Image
import pandas as pd
import numpy as np

import os
import re

def count_labels_in_dataloader(dataloader):
        label_counter = Counter()
        for _, labels in dataloader:
            label_counter.update(labels.tolist())
        return label_counter

def build_resnet_for_grayscale(num_classes=2):
    model = models.resnet18(pretrained=True)  # or resnet50, resnet34, etc.

    # Modify first conv layer to accept 1-channel input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Replace the final FC layer to match your number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


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
    transforms.Grayscale(num_output_channels=3),  # duplicate grayscale to RGB
    transforms.Resize((320, 320)),  # optional
    transforms.ToTensor(),
])


    dataset = EyeFFEDataset(df_final, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # Usage
    label_distribution = count_labels_in_dataloader(dataloader)
    print("Label distribution in dataloader:")
    for label, count in sorted(label_distribution.items()):
        print(f"Label {label}: {count} samples")

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

    model = build_resnet_for_grayscale(num_classes=2)  # or use unmodified ResNet if using Grayscale(3)
    model = model.to(device)
    channels = 1
    H = 320
    W = 320
    # Now log it
    input_size=(channels, H, W)
    summary(model, input_size=(1, 320, 320), batch_size=1, device="cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Accuracy: {acc:.2f}%")









   
    
   # why are all sample points having the same label? Is shuffling not happening?
   # # do we need to reshape 320X320 to 224X224 to run ResNet on it?
    


if __name__ == "__main__":
    main()