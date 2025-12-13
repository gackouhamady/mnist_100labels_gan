import torch
<<<<<<< HEAD
from sklearn.metrics import confusion_matrix


def accuracy(preds, labels):
    return (preds == labels).float().mean().item()


def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def compute_confusion_matrix(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    return confusion_matrix(all_labels, all_preds)


=======
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
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)

            if logits.shape[1] == 11:
                logits = logits[:, :10]

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    return cm



# ---------------------------------------------------------
# Pretty print confusion matrix
# ---------------------------------------------------------
>>>>>>> 49fecc8 (Initial implementation of Semi-Supervised GAN for MNIST (100 labels))
def print_confusion_matrix(cm):
    print("\nConfusion Matrix:")
    print(cm)
