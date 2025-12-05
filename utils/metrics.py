import torch
import numpy as np
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------
# Simple AverageMeter (to track average loss, accuracy, etc.)
# ---------------------------------------------------------
class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for loss tracking during training loops.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ---------------------------------------------------------
# Accuracy computation
# ---------------------------------------------------------
def accuracy_from_logits(logits, labels):
    """
    Computes accuracy in a batch from raw logits.
    Returns: float
    """
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


# ---------------------------------------------------------
# Confusion matrix helper
# ---------------------------------------------------------
def compute_confusion_matrix(model, dataloader, device):
    """
    Computes confusion matrix on full dataset.
    Requires sklearn.
    """
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    return cm


# ---------------------------------------------------------
# Pretty print confusion matrix
# ---------------------------------------------------------
def print_confusion_matrix(cm):
    print("\nConfusion Matrix:")
    print(cm)
