import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
from torchvision import transforms

training_data_transformations = transforms.Compose(
    [
        transforms.Resize(224, 224),
        transforms.Normalize(0.5, 0.5)
    ]
)

class CustomImageDataset(Dataset):
    def __init__(self, data_file, transform=None, target_transform=None):
        self.data_csv = pd.read_csv(data_file)
        self.img_labels = self.data_csv["encoded_label"]
        self.img_dir = self.data_csv["image_path"]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir.iloc[idx]
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label