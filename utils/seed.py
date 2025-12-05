import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Ensures full reproducibility of results.
    Handles Python, NumPy, PyTorch CPU & GPU.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[Seed] Set random seed to: {seed}")
