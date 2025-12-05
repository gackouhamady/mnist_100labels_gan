import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Lightweight DCGAN-style generator for MNIST (28x28).
    Input: latent vector z (batch, z_dim)
    Output: image (batch, 1, 28, 28) in [-1, 1]
    """

    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim

        self.net = nn.Sequential(
            # Fully connected layer → reshape to feature map
            nn.Linear(z_dim, 128 * 7 * 7),
            nn.ReLU(True),

            nn.Unflatten(1, (128, 7, 7)),  # Shape: (128, 7, 7)

            # 7×7 → 14×14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            # 14×14 → 28×28
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output normalized to [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


if __name__ == "__main__":
    # Quick test
    z = torch.randn(4, 100)
    G = Generator(z_dim=100)
    fake_imgs = G(z)
    print("Output shape:", fake_imgs.shape)  # Expected: [4, 1, 28, 28]
