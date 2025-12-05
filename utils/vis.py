import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os


# ---------------------------------------------------------
# Show a batch of images (MNIST or generated)
# ---------------------------------------------------------
def show_images(images, nrow=8, title="Image Grid"):
    """
    Display a grid of images using matplotlib.
    Images should be tensors in [-1, 1] or [0, 1].
    """

    # Convert [-1,1] → [0,1] for display
    if images.min() < 0:
        images = (images + 1) / 2

    grid = vutils.make_grid(images, nrow=nrow, padding=2)
    np_img = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(np_img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


# ---------------------------------------------------------
# Save generated images to file
# ---------------------------------------------------------
def save_image_grid(images, filename, nrow=8):
    """
    Save a grid of images to disk.
    Automatically converts images from [-1, 1] to [0, 1].
    """

    # Convert [-1,1] → [0,1]
    if images.min() < 0:
        images = (images + 1) / 2

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    vutils.save_image(images, filename, nrow=nrow)
    print(f"Saved image grid to: {filename}")


# ---------------------------------------------------------
# Generate and show samples from Generator
# ---------------------------------------------------------
def generate_and_show(generator, num_samples=64, z_dim=100, device="cpu"):
    """
    Draw random latent vectors, generate samples, and show them.
    """

    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, z_dim, device=device)
        fake_images = generator(z)

    show_images(fake_images, title="Generated Samples")
    return fake_images


# ---------------------------------------------------------
# Generate and save samples
# ---------------------------------------------------------
def generate_and_save(generator, filename, num_samples=64, z_dim=100, device="cpu"):
    """
    Generate samples and save them to a PNG file.
    """

    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, z_dim, device=device)
        fake_images = generator(z)

    save_image_grid(fake_images, filename)
    return fake_images
