import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    """
    Simple CNN baseline for MNIST classification.
    Designed for training on only 100 labeled samples.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 1x28x28 -> 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 32x14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 64x7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Quick test
    model = BaselineCNN()
    x = torch.randn(4, 1, 28, 28)  # batch of 4 fake MNIST images
    logits = model(x)
    print("Output shape:", logits.shape)  # Expected: [4, 10]
