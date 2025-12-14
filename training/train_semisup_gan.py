import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import json
import os
import sys
import numpy as np
<<<<<<< HEAD
=======
import itertools
>>>>>>> 49fecc8 (Initial implementation of Semi-Supervised GAN for MNIST (100 labels))

# -------------------------------
# Import models
# -------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gan_generator import Generator
from models.gan_discriminator import DiscriminatorKPlus1


# -------------------------------
# Labeled MNIST dataset (100 samples)
# -------------------------------
class LabeledMNIST(Dataset):
    def __init__(self, root, indices_file):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # [-1, 1]
        ])

        self.mnist = datasets.MNIST(root=root, train=True, download=True, transform=transform)

        with open(indices_file, "r") as f:
            self.indices = json.load(f)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.mnist[real_idx]
        return img, label


# -------------------------------
# Unlabeled MNIST dataset
# -------------------------------
class UnlabeledMNIST(Dataset):
    def __init__(self, root, indices_file):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        mnist = datasets.MNIST(root=root, train=True, download=True, transform=transform)

        with open(indices_file, "r") as f:
            all_indices = json.load(f)

<<<<<<< HEAD
        # unlabeled = full MNIST minus labeled indices
=======
>>>>>>> 49fecc8 (Initial implementation of Semi-Supervised GAN for MNIST (100 labels))
        self.indices = all_indices
        self.mnist = mnist

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, _ = self.mnist[real_idx]
        return img


# -------------------------------
# Feature Matching Loss
# -------------------------------
def feature_matching_loss(real_features, fake_features):
    return torch.mean((real_features.mean(0) - fake_features.mean(0)) ** 2)


# -------------------------------
# Evaluate discriminator as classifier (10 classes)
# -------------------------------
def evaluate_classifier(discriminator, loader, device):
    discriminator.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
<<<<<<< HEAD
            logits = discriminator(images)[:, :10]     # only real classes
=======
            logits = discriminator(images)[:, :10]  # only real classes (0..9)
>>>>>>> 49fecc8 (Initial implementation of Semi-Supervised GAN for MNIST (100 labels))
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# -------------------------------
# MAIN TRAINING SCRIPT
# -------------------------------
if __name__ == "__main__":

    # -------------------------------
    # Config
    # -------------------------------
    root = "./data"
    labeled_file = "./datasets/labeled_indices.json"
    unlabeled_file = "./datasets/unlabeled_indices.json"

    batch_size = 64
    z_dim = 100
    lr_d = 2e-4
    lr_g = 2e-4
    epochs = 50
    lambda_unsup = 1.0

    save_path_D = "./experiments/gan_discriminator.pt"
    save_path_G = "./experiments/gan_generator.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -------------------------------
    # Datasets & Dataloaders
    # -------------------------------
    labeled_dataset = LabeledMNIST(root, labeled_file)
    unlabeled_dataset = UnlabeledMNIST(root, unlabeled_file)

    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)

<<<<<<< HEAD
    # Test set for evaluation
    test_dataset = datasets.MNIST(root=root, train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,))
                                  ]))
=======
    print("Labeled batches:", len(labeled_loader))
    print("Unlabeled batches:", len(unlabeled_loader))

    # Test set for evaluation
    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )
>>>>>>> 49fecc8 (Initial implementation of Semi-Supervised GAN for MNIST (100 labels))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # -------------------------------
    # Models
    # -------------------------------
    D = DiscriminatorKPlus1(num_classes=10).to(device)
    G = Generator(z_dim=z_dim).to(device)

    # -------------------------------
    # Optimizers
    # -------------------------------
    opt_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    opt_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))

    ce_loss = nn.CrossEntropyLoss()

    print("\nStarting Semi-Supervised GAN Training…\n")

<<<<<<< HEAD
=======
    # -------------------------------------------------
    # Track accuracy per epoch
    # -------------------------------------------------
    epoch_accuracies = []
    os.makedirs("./experiments", exist_ok=True)

    # Reset accuracy log file (so each run starts clean)
    with open("./experiments/accuracy_per_epoch.txt", "w") as f:
        f.write("epoch,accuracy\n")



>>>>>>> 49fecc8 (Initial implementation of Semi-Supervised GAN for MNIST (100 labels))
    # -------------------------------
    # Training Loop
    # -------------------------------
    for epoch in range(1, epochs + 1):
        D.train()
        G.train()

<<<<<<< HEAD
        for (x_l, y_l), x_u in zip(labeled_loader, unlabeled_loader):
=======
        # Cycle labeled batches forever; iterate over ALL unlabeled batches each epoch
        labeled_iter = itertools.cycle(labeled_loader)

        for x_u in unlabeled_loader:
            x_l, y_l = next(labeled_iter)

>>>>>>> 49fecc8 (Initial implementation of Semi-Supervised GAN for MNIST (100 labels))
            x_l, y_l = x_l.to(device), y_l.to(device)
            x_u = x_u.to(device)

            # ------------------------------------------
            # Train Discriminator (Supervised + Unsupervised)
            # ------------------------------------------
            opt_D.zero_grad()

<<<<<<< HEAD
            # SUPERVISED loss on labeled data
            logits_l = D(x_l)[:, :10]  # ignore fake class
            loss_sup = ce_loss(logits_l, y_l)

            # UNSUPERVISED: real unlabeled
            logits_u = D(x_u)
            prob_fake_u = torch.softmax(logits_u, dim=1)[:, 10]
            loss_unsup_real = -torch.log(1 - prob_fake_u + 1e-8).mean()

            # UNSUPERVISED: generated fake samples
=======
            # SUPERVISED loss on labeled data (only 10 real classes)
            logits_l = D(x_l)[:, :10]
            loss_sup = ce_loss(logits_l, y_l)

            # UNSUPERVISED: real unlabeled should be "not fake"
            logits_u = D(x_u)  # 11 classes
            prob_fake_u = torch.softmax(logits_u, dim=1)[:, 10]
            loss_unsup_real = -torch.log(1 - prob_fake_u + 1e-8).mean()

            # UNSUPERVISED: generated fake samples should be "fake"
>>>>>>> 49fecc8 (Initial implementation of Semi-Supervised GAN for MNIST (100 labels))
            z = torch.randn(x_u.size(0), z_dim, device=device)
            x_fake = G(z).detach()
            logits_fake = D(x_fake)
            prob_fake_g = torch.softmax(logits_fake, dim=1)[:, 10]
            loss_unsup_fake = -torch.log(prob_fake_g + 1e-8).mean()

            loss_unsup = loss_unsup_real + loss_unsup_fake

            # TOTAL D LOSS
            loss_D = loss_sup + lambda_unsup * loss_unsup
            loss_D.backward()
            opt_D.step()

            # ------------------------------------------
            # Train Generator (Feature Matching)
            # ------------------------------------------
            opt_G.zero_grad()
            z = torch.randn(x_u.size(0), z_dim, device=device)
            x_fake = G(z)
            _, feat_fake = D(x_fake, return_features=True)
            _, feat_real = D(x_u, return_features=True)

            loss_G = feature_matching_loss(feat_real, feat_fake)
            loss_G.backward()
            opt_G.step()

        # -------------------------------
        # Evaluate as classifier
        # -------------------------------
        acc = evaluate_classifier(D, test_loader, device)
<<<<<<< HEAD
        print(f"Epoch [{epoch}/{epochs}] | Acc: {acc:.4f} | L_sup: {loss_sup:.3f} | L_unsup: {loss_unsup:.3f} | L_G: {loss_G:.3f}")
=======
        epoch_accuracies.append(acc)

        with open("./experiments/accuracy_per_epoch.txt", "a") as f:
            f.write(f"{epoch},{acc:.4f}\n")

        print(
            f"Epoch [{epoch}/{epochs}] | Acc: {acc:.4f} | "
            f"L_sup: {loss_sup:.3f} | L_unsup: {loss_unsup:.3f} | L_G: {loss_G:.3f}"
        )
>>>>>>> 49fecc8 (Initial implementation of Semi-Supervised GAN for MNIST (100 labels))

    # -------------------------------
    # Save models
    # -------------------------------
    os.makedirs("./experiments", exist_ok=True)
    torch.save(D.state_dict(), save_path_D)
    torch.save(G.state_dict(), save_path_G)

<<<<<<< HEAD
=======
    with open("./experiments/results_sgan.txt", "w") as f:
        f.write(f"Final Test Accuracy (SGAN): {acc:.4f}\n")

>>>>>>> 49fecc8 (Initial implementation of Semi-Supervised GAN for MNIST (100 labels))
    print("\nTraining finished.")
    print(f"Discriminator saved to {save_path_D}")
    print(f"Generator saved to {save_path_G}")
