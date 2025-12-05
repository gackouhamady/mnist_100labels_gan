import os
import torch
from utils.seed import set_seed

# Paths
BASELINE_SCRIPT = "training/train_baseline.py"
GAN_SCRIPT = "training/train_semisup_gan.py"
EVAL_SCRIPT = "training/evaluate_classifier.py"  # optional if you add it later

# -----------------------------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------------------------
def run_baseline():
    print("\n===== TRAINING BASELINE CNN (100 labels) =====\n")
    os.system(f"python {BASELINE_SCRIPT}")


def run_gan():
    print("\n===== TRAINING SEMI-SUPERVISED GAN =====\n")
    os.system(f"python {GAN_SCRIPT}")


def run_evaluation():
    if os.path.exists(EVAL_SCRIPT):
        print("\n===== EVALUATING GAN DISCRIMINATOR =====\n")
        os.system(f"python {EVAL_SCRIPT}")
    else:
        print("\n(No evaluate_classifier.py file found — skipping evaluation.)")


def main():
    # -------------------------------------------------
    # Step 0: Reproducibility
    # -------------------------------------------------
    set_seed(42)

    # -------------------------------------------------
    # Step 1: Train baseline model
    # -------------------------------------------------
    run_baseline()

    # -------------------------------------------------
    # Step 2: Train semi-supervised GAN
    # -------------------------------------------------
    run_gan()

    # -------------------------------------------------
    # Step 3: Evaluate discriminator as classifier
    # -------------------------------------------------
    run_evaluation()

    print("\n===== PROJECT COMPLETED SUCCESSFULLY =====\n")


if __name__ == "__main__":
    main()
