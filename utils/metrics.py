import torch
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


def print_confusion_matrix(cm):
    print("\nConfusion Matrix:")
    print(cm)
