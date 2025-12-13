import os
import torch
from torchvision.utils import save_image, make_grid

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gan_generator import Generator

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    z_dim = 100
    n_samples = 64          # 8x8 grid
    out_dir = "./experiments"
    os.makedirs(out_dir, exist_ok=True)

    # Load generator
    G = Generator(z_dim=z_dim).to(device)
    ckpt = os.path.join(out_dir, "gan_generator.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Generator checkpoint not found: {ckpt} (train SGAN first)")

    G.load_state_dict(torch.load(ckpt, map_location=device))
    G.eval()
    print("Loaded generator from:", ckpt)

    # Sample noise and generate images
    with torch.no_grad():
        z = torch.randn(n_samples, z_dim, device=device)
        x_fake = G(z)  # expected in [-1, 1] if you used Normalize(0.5,0.5)

        # Convert to [0,1] for saving
        x_fake = (x_fake + 1) / 2
        x_fake = x_fake.clamp(0, 1)

        grid = make_grid(x_fake, nrow=8)  # 8x8 grid
        out_path = os.path.join(out_dir, "gan_samples_grid.png")
        save_image(grid, out_path)

    print("Saved samples grid to:", out_path)

if __name__ == "__main__":
    main()
