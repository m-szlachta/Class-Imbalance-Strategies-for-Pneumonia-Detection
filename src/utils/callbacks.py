import torch

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_loss = None

    def on_epoch_end(self, current_loss: float) -> bool:
        if self.best_loss is None or current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter == self.patience:
                print("Early stopping")
                return True
        return False

class ReduceLROnPlateau:
    def __init__(self, factor: float = 0.5, patience: int = 2, min_lr: float = 1e-6):
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.counter = 0
        self.best_loss = None

    def reduce(self, current_loss: float, lr: float) -> float:
        if self.best_loss is None or current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.counter = 0
                lr = max(lr * self.factor, self.min_lr)
                print(f"Lr reduced to {lr}")
        return lr

class ModelCheckpoint:
    def __init__(self, path):
        self.path = path
        self.best_loss = None

    def save_model(self, model, current_loss):
        if self.best_loss == None or current_loss < self.best_loss:
            self.best_loss = current_loss
            torch.save(model.state_dict(), self.path)

    
