import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import sys
<<<<<<< HEAD
=======
import numpy as np
import matplotlib.pyplot as plt
>>>>>>> 49fecc8 (Initial implementation of Semi-Supervised GAN for MNIST (100 labels))

# Import model + metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gan_discriminator import DiscriminatorKPlus1
from utils.metrics import accuracy_from_logits, compute_confusion_matrix, print_confusion_matrix


# -----------------------------------------------------------------------------
# Evaluate discriminator on MNIST test set
# -----------------------------------------------------------------------------
def evaluate(model_path, batch_size=256, device="cpu"):
    # ----------------------------
    # Load model
    # ----------------------------
    model = DiscriminatorKPlus1(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"\nLoaded discriminator model from: {model_path}\n")

    # ----------------------------
    # MNIST test loader
    # ----------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ----------------------------
    # Compute accuracy
    # ----------------------------
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)[:, :10]  # remove fake class
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy of GAN Discriminator (as classifier): {accuracy:.4f}")

<<<<<<< HEAD
    # ----------------------------
    # Optional: Confusion Matrix
=======
    # -------------------------------------------------
    # Save accuracy to file
    # -------------------------------------------------
    os.makedirs("./experiments", exist_ok=True)
    with open("./experiments/results_sgan.txt", "w") as f:
        f.write(f"Final Test Accuracy (SGAN): {accuracy:.4f}\n")

    # ----------------------------
    # Confusion Matrix
>>>>>>> 49fecc8 (Initial implementation of Semi-Supervised GAN for MNIST (100 labels))
    # ----------------------------
    print("\nComputing confusion matrix…")
    cm = compute_confusion_matrix(model, test_loader, device)
    print_confusion_matrix(cm)

<<<<<<< HEAD
    return accuracy
=======
    # Save confusion matrix to file
    os.makedirs("./experiments", exist_ok=True)

    np.savetxt(
        "./experiments/confusion_matrix_sgan.txt",
        cm,
        fmt="%d"
    )

    # -------------------------------------------------
    # Save confusion matrix as PNG heatmap
    # -------------------------------------------------
    os.makedirs("./experiments", exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix – SGAN (100 labels)")
    plt.colorbar()

    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.tight_layout()
    plt.savefig("./experiments/confusion_matrix_sgan.png")
    plt.close()
>>>>>>> 49fecc8 (Initial implementation of Semi-Supervised GAN for MNIST (100 labels))


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "./experiments/gan_discriminator.pt"

    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} does not exist. Train GAN first.")
    else:
        evaluate(model_path, device=device)
