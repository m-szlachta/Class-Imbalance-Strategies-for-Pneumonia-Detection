import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn
from PIL import Image

# run as a script, sys.path[0] is src/architectures, so put the repo root on it
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.evaluation import MetricTracker
from src.utils.callbacks import *

BATCH_SIZE = 64
DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {DEVICE} device")

image_transformations = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
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
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels.iloc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


train_data = CustomImageDataset(data_file="data/data_csv/train.csv", transform=image_transformations)
val_data = CustomImageDataset(data_file="data/data_csv/val.csv", transform=image_transformations)
test_data = CustomImageDataset(data_file="data/data_csv/test.csv", transform=image_transformations)


train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GroupNorm(2, 32),      # 32 ch, 2 groups (16 ch each)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64),      # 64 ch, 4 groups
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),     # 128 ch, 8 groups
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 256),    # 256 ch, 16 groups
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)

        return x


def train(dataloader, model, loss_fn, optimizer, tracker, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, train_acc = 0, 0
    model.train()

    for X, y in dataloader:
        X, y = X.to(DEVICE), y.to(DEVICE).float()

        pred = model(X).squeeze(1)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss_fn(pred, y).item()
        train_acc += ((pred > 0).float() == y).float().sum().item()
        
    train_acc /= size
    train_loss /= num_batches
    print(f"Train: \n Accuracy: {train_acc*100:.2f}%, Loss {train_loss:.4f}")
    tracker.record("train", epoch, train_loss, train_acc)

def validation(dataloader, model, loss_fn, tracker, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    vloss, vacc = 0, 0
    with torch.no_grad():
        for vinputs, vlabels in dataloader:
            vinputs, vlabels = vinputs.to(DEVICE), vlabels.to(DEVICE).float()
            voutputs = model(vinputs).squeeze(1)
            vloss += loss_fn(voutputs, vlabels).item()
            vacc += ((voutputs > 0).float() == vlabels).float().sum().item()

    vloss /= num_batches
    vacc /= size
    print(f"Validation: \n  Accuracy: {vacc*100:.2f}%, Loss: {vloss:.4f}")
    tracker.record("val", epoch, vloss, vacc)

    return vloss, vacc

def test(dataloader, model, loss_fn, tracker, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE).float()
            pred = model(X).squeeze(1)
            test_loss += loss_fn(pred, y).item()
            test_acc += ((pred > 0).float() == y).float().sum().item()
    test_loss /= num_batches
    test_acc /= size
    print(f"Test: \n Accuracy: {test_acc*100:.2f}%, Avg loss: {test_loss:.4f} \n")
    tracker.record("test", epoch, test_loss, test_acc)

model = CNN().to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr)


epochs = 20
tracker = MetricTracker()
early_stop = EarlyStopping()
reduce_lr = ReduceLROnPlateau()
model_checkpoint = ModelCheckpoint("data/models/CNN_model.pt")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, tracker, t + 1)
    vloss, _ = validation(val_dataloader, model, loss_fn, tracker, t + 1)

    if early_stop.on_epoch_end(vloss):
        break

    lr = reduce_lr.reduce(vloss, lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    model_checkpoint.save_model(model, vloss)
    
    test(test_dataloader, model, loss_fn, tracker, t + 1)

tracker.plot("reports/curves.png")