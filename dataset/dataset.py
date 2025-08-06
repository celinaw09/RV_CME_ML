from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd


class EyeFFEDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = {"non_CME": 0, "CME": 1}  # adjust if more classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label = self.label_map[row["label"]]
        
        image = Image.open(image_path)   # Convert to Grayscales

        if self.transform:
            image = self.transform(image)

        return image, label