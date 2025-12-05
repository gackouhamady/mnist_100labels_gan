import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorKPlus1(nn.Module):
    """
    Semi-Supervised GAN Discriminator (K+1 classes):
        - First K classes = real MNIST classes (0..9)
        - (K+1)-th class = FAKE / GENERATED

    This architecture follows Salimans et al. (2016).
    It exposes 'features' for Feature Matching Loss.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.num_classes = num_classes

        # Feature extractor
        self.features_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),   # 1x28x28 -> 64x28x28
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),                             # 64x14x14

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 128x14x14
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),                             # 128x7x7

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.LeakyReLU(0.2)
        )

        # Logits for K+1 classes
        self.classifier = nn.Linear(256, num_classes + 1)

    def forward(self, x, return_features=False):
        """
        If return_features=True:
            returns (logits, features) for Feature Matching
        """
        features = self.features_extractor(x)
        logits = self.classifier(features)

        if return_features:
            return logits, features

        return logits


if __name__ == "__main__":
    # Quick test
    model = DiscriminatorKPlus1(num_classes=10)
    x = torch.randn(4, 1, 28, 28)
    logits = model(x)
    print("Logits shape:", logits.shape)  # Expected: [4, 11]
