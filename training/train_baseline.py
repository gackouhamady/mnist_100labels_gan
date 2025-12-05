import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import json
import os
import sys

# Import the baseline CNN
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_baseline import BaselineCNN


# ----------------------------
# Custom dataset for labeled MNIST
# ----------------------------
class LabeledMNIST(Dataset):
    def __init__(self, root, indices_file):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # Load MNIST (train split)
        self.mnist = datasets.MNIST(root=root, train=True, download=True, transform=transform)

        # Load the 100 labeled indices
        with open(indices_file, "r") as f:
            self.indices = json.load(f)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.mnist[real_idx]
        return img, label


# ----------------------------
# Training function
# ----------------------------
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ----------------------------
# Evaluation
# ----------------------------
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# ----------------------------
# MAIN SCRIPT
# ----------------------------
if __name__ == "__main__":

    # ----------------------------------
    # Config
    # ----------------------------------
    root = "./data"
    labeled_file = "./datasets/labeled_indices.json"
    batch_size = 32
    lr = 1e-3
    epochs = 50
    save_path = "./experiments/baseline_cnn.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ----------------------------------
    # Dataset & DataLoader
    # ----------------------------------
    train_dataset = LabeledMNIST(root, labeled_file)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Test set
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # ----------------------------------
    # Model, Loss, Optimizer
    # ----------------------------------
    model = BaselineCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ----------------------------------
    # Training Loop
    # ----------------------------------
    print("\nStarting training on 100 labeled samples...\n")

    for epoch in range(1, epochs + 1):
        loss = train(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)

        print(f"Epoch [{epoch}/{epochs}] | Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")

    # ----------------------------------
    # Save model
    # ----------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
