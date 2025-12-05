import torch
from torchvision import datasets, transforms
import json
import os

def create_mnist_100_labels(root="./data", save_dir="./datasets"):
    """
    Creates two JSON files:
        - labeled_indices.json   (100 examples: 10 per class)
        - unlabeled_indices.json (the rest, labels ignored)
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Transform: convert to tensor only
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Download MNIST
    mnist = datasets.MNIST(root=root, train=True, download=True, transform=transform)

    # Prepare storage
    labeled_indices = []
    unlabeled_indices = []

    # We want exactly 10 samples per class → 10 * 10 = 100 labeled
    per_class_count = {c: 0 for c in range(10)}
    max_per_class = 10

    # First pass: select labeled indices
    for idx, (_, label) in enumerate(mnist):
        if per_class_count[label] < max_per_class:
            labeled_indices.append(idx)
            per_class_count[label] += 1

        # Stop early if we already collected 100 samples
        if len(labeled_indices) == 100:
            break

    # Second pass: all others → unlabeled
    all_indices = set(range(len(mnist)))
    labeled_set = set(labeled_indices)

    unlabeled_indices = list(all_indices - labeled_set)

    # Save as JSON
    labeled_path = os.path.join(save_dir, "labeled_indices.json")
    unlabeled_path = os.path.join(save_dir, "unlabeled_indices.json")

    with open(labeled_path, "w") as f:
        json.dump(labeled_indices, f)

    with open(unlabeled_path, "w") as f:
        json.dump(unlabeled_indices, f)

    print("✓ MNIST split created successfully!")
    print(f"  Labeled samples:   {len(labeled_indices)} (10 per class)")
    print(f"  Unlabeled samples: {len(unlabeled_indices)}")
    print(f"  Saved to: {save_dir}/")


if __name__ == "__main__":
    create_mnist_100_labels()
