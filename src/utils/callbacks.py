class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_loss = None

    def on_epoch_end(self, current_loss):
        if self.best_loss is None or self.best_loss < current_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter == self.patience:
                print("Early stopping")
                return True
        return False