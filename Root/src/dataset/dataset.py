from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd


class EyeFFEDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label = row["label"]  # Already numeric

        # Load image in grayscale mode ('L')
        image = Image.open(image_path).convert("L")  # Ensure grayscale

        if self.transform:
            image = self.transform(image)

        return image, label