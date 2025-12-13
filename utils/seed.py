import random
import numpy as np
import torch

def set_seed(seed=42):
<<<<<<< HEAD
=======
    """
    Ensures full reproducibility of results.
    Handles Python, NumPy, PyTorch CPU & GPU.
    """

>>>>>>> 49fecc8 (Initial implementation of Semi-Supervised GAN for MNIST (100 labels))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
<<<<<<< HEAD
        torch.cuda.manual_seed_all(seed)

=======
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
>>>>>>> 49fecc8 (Initial implementation of Semi-Supervised GAN for MNIST (100 labels))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[Seed] Set random seed to: {seed}")
