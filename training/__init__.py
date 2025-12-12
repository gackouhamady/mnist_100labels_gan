"""
Training package for the MNIST 100-label Semi-Supervised GAN project.

This module exposes:
- Baseline supervised training utilities
- Semi-supervised GAN training components
- Evaluation helpers
"""

# ------------------------------------------------------------------
# Baseline supervised CNN training
# ------------------------------------------------------------------
from .train_baseline import (
    LabeledMNIST as BaselineLabeledMNIST,
    train as train_baseline_epoch,
    evaluate as evaluate_baseline
)

# ------------------------------------------------------------------
# Semi-supervised GAN training
# ------------------------------------------------------------------
from .train_semisup_gan import (
    LabeledMNIST as SGANLabeledMNIST,
    UnlabeledMNIST,
    feature_matching_loss,
    evaluate_classifier,
)

# ------------------------------------------------------------------
# Discriminator evaluation utilities
# ------------------------------------------------------------------
from .evaluate_classifier import (
    evaluate as evaluate_discriminator,
)

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
__all__ = [
    # Baseline
    "BaselineLabeledMNIST",
    "train_baseline_epoch",
    "evaluate_baseline",

    # SGAN
    "SGANLabeledMNIST",
    "UnlabeledMNIST",
    "feature_matching_loss",
    "evaluate_classifier",

    # Evaluation
    "evaluate_discriminator",
]
