"""
training package

This package exposes the training utilities:
- train_baseline : train a supervised CNN on 100 labeled samples
- train_semisup_gan : train the Semi-Supervised GAN (K+1 classifier)
- evaluate_classifier : evaluate any trained classifier on MNIST test set
"""

from .train_baseline import train_baseline_model
from .train_semisup_gan import train_semisupervised_gan
from .evaluate_classifier import evaluate_model

__all__ = [
    "train_baseline_model",
    "train_semisupervised_gan",
    "evaluate_model",
]
