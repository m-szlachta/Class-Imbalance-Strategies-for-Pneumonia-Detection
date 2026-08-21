import sys
from pathlib import Path
import pandas as pd
import torch
from torch import nn

# run as a script, sys.path[0] is src/architectures, so put the repo root on it
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.evaluation import MetricTracker
from src.utils.evaluation_metrics import PredictionCollector, format_metrics, sweep_threshold
from src.utils.dataset import GPUDataset
from src.utils.callbacks import *

BATCH_SIZE = 64
DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {DEVICE} device")

# input size is fixed at 224x224, so let cuDNN pick and cache the best algorithm once
torch.backends.cudnn.benchmark = True

# each split variant caches under its own name, so pointing DATA_DIR at an
# oversampled or undersampled split builds that split's cache rather than reusing another's
DATA_DIR = "data/data_csv"

train_data = GPUDataset(f"{DATA_DIR}/train.csv", DEVICE)
val_data = GPUDataset(f"{DATA_DIR}/val.csv", DEVICE)
test_data = GPUDataset(f"{DATA_DIR}/test.csv", DEVICE)

n_pos = train_data.labels.sum()
class_weight = (len(train_data) - n_pos) / n_pos
CW = True

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


def train(data, model, loss_fn, optimizer, tracker, epoch):
    # the collector keeps batches on the GPU; calling .item() per batch would sync every step
    collector = PredictionCollector()
    model.train()

    for X, y in data.batches(BATCH_SIZE, shuffle=True):
        pred = model(X).squeeze(1)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        collector.update(pred.detach(), y, loss.detach())

    metrics = collector.compute()
    print(format_metrics(metrics, "Train"))
    tracker.record("train", epoch, metrics["loss"], metrics)

def validation(data, model, loss_fn, tracker, epoch):
    collector = PredictionCollector()
    model.eval()
    with torch.no_grad():
        for vinputs, vlabels in data.batches(BATCH_SIZE):
            voutputs = model(vinputs).squeeze(1)
            collector.update(voutputs, vlabels, loss_fn(voutputs, vlabels))

    metrics = collector.compute()
    print(format_metrics(metrics, "Validation"))
    tracker.record("val", epoch, metrics["loss"], metrics)

    return metrics["loss"], metrics

def test(data, model, loss_fn, tracker, epoch):
    collector = PredictionCollector()
    model.eval()
    with torch.no_grad():
        for X, y in data.batches(BATCH_SIZE):
            pred = model(X).squeeze(1)
            collector.update(pred, y, loss_fn(pred, y))

    metrics = collector.compute()
    print(format_metrics(metrics, "Test"), "\n")
    tracker.record("test", epoch, metrics["loss"], metrics)

    return metrics

model = CNN().to(DEVICE)

if CW:
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weight)
else:
    loss_fn = nn.BCEWithLogitsLoss()
    
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)

model_path = "data/models/CNN_model.pt"
epochs = 20
tracker = MetricTracker()
early_stop = EarlyStopping()
reduce_lr = ReduceLROnPlateau()
model_checkpoint = ModelCheckpoint(model_path)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_data, model, loss_fn, optimizer, tracker, t + 1)
    vloss, _ = validation(val_data, model, loss_fn, tracker, t + 1)

    model_checkpoint.save_model(model, vloss)
    test(test_data, model, loss_fn, tracker, t + 1)

    if early_stop.on_epoch_end(vloss):
        break

    lr = reduce_lr.reduce(vloss, lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

model.load_state_dict(torch.load(model_path, weights_only=True))

best = sweep_threshold(val_data, model, "reports_cw/threshold_sweep_cw")
print(f"Best validation threshold: {best['threshold']:.2f} (accuracy {best['accuracy']:.4f})")

print("TEST score:")
test(test_data, model, loss_fn, tracker, t + 1)


tracker.plot("reports_cw/curves_cw.png", metrics=["accuracy", "auroc", "auprc"])