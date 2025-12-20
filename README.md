## Semi-Supervised GAN for MNIST (100 Labels)
=======
# Semi-Supervised GAN for MNIST with 100 Labels

<p align="center">
<img alt="University Paris Cité" src="[https://img.shields.io/badge/University-Paris%20Cité-6f42c1?style=for-the-badge&logo=academia&logoColor=white](https://img.shields.io/badge/University-Paris%20Cité-6f42c1?style=for-the-badge&logo=academia&logoColor=white)">
<img alt="Master ML for Data Science" src="[https://img.shields.io/badge/Master-Machine%20Learning%20for%20Data%20Science-1976D2?style=for-the-badge&logo=python&logoColor=white](https://img.shields.io/badge/Master-Machine%20Learning%20for%20Data%20Science-1976D2?style=for-the-badge&logo=python&logoColor=white)">
<img alt="Deep Learning Project" src="[https://img.shields.io/badge/Project-Deep%20Learning%20-%20Semi--Supervised%20GAN-FF9800?style=for-the-badge&logo=jupyter&logoColor=white](https://img.shields.io/badge/Project-Deep%20Learning%20-%20Semi--Supervised%20GAN-FF9800?style=for-the-badge&logo=jupyter&logoColor=white)">
<img alt="Academic Year" src="[https://img.shields.io/badge/Year-2025%2F2026-009688?style=for-the-badge&logo=googlecalendar&logoColor=white](https://img.shields.io/badge/Year-2025%2F2026-009688?style=for-the-badge&logo=googlecalendar&logoColor=white)">
</p>

---

## Project Information

**Université Paris Cité — Master 2 Machine Learning for Data Science** **Course:** Deep Learning

**Project:** Semi-Supervised GAN (K+1 Discriminator) for Low-Label Image Classification[1]

**Team**

* Manel LOUNISSI – *manel2.lounissi@gmail.com* - Sandeep-Singh NIRMAL – *nirmalsinghsandeep@gmail.com* - Brice SAILLARD – *brice.saillard.bs@gmail.com* - Hamady GACKOU – *hamady.gackou@etu.u-paris.fr* **Supervisor:** Blaise Hanczar[1]

**Dataset & Setting**

* MNIST: 60 000 train / 10 000 test
* Only **100 labeled samples** (10 per class) are used for supervised learning
* Remaining **59 900** training samples are treated as **unlabeled**[1]

---

## Problem & Approach

Labeling large datasets is costly, while unlabeled data is abundant. In this project, the goal is to perform **handwritten digit classification on MNIST** using **only 100 labeled images**, and to exploit the remaining images as unlabeled data through a **Semi-Supervised GAN (SGAN)**.[1]

The key idea follows **Salimans et al., “Improved Techniques for Training GANs” (NeurIPS 2016)**:

* The **discriminator** is turned into a **K+1 classifier**:
* Classes 0–9: real digits
* Class 10: fake (generated) samples[1]


* The **generator** is trained with **feature matching**, making its objective to match intermediate feature statistics of real data instead of directly fooling the discriminator.[1]

This allows the model to **leverage the structure of unlabeled data** in addition to the 100 labels.

---

## Model Architectures

### Baseline Supervised CNN

A compact CNN trained **only on the 100 labeled samples** serves as the supervised baseline.[1]

* Convolutional feature extractor (2 conv + ReLU + max-pooling blocks)
* Fully-connected classifier:
* Flatten
* Linear → 128 + ReLU + Dropout(0.3)
* Linear → 10 logits (digits 0–9)[1]



### SGAN Discriminator (K+1 Classes)

The **discriminator** outputs **K+1 logits**:

* 10 logits for real classes (0–9)
* 1 logit for the “fake” class[1]

Architecture:

* Conv2d(1 → 64) + LeakyReLU + MaxPool2d
* Conv2d(64 → 128) + LeakyReLU + MaxPool2d
* Flatten → Linear(128×7×7 → 256) + LeakyReLU
* Final Linear(256 → 11)
* Optionally returns intermediate **features** for feature matching[1]

### SGAN Generator

The **generator** maps a latent vector  to a 28×28 MNIST image:[1]

* Linear(z_dim → 128×7×7) + ReLU
* Unflatten → ConvTranspose2d blocks upsampling to 1×28×28
* Final activation: Tanh[1]

---

## Project Team
## Training Objectives

### Discriminator Loss

The discriminator minimizes a combination of:

* **Supervised loss** : cross-entropy on labeled samples using the first K logits (0–9)
* **Unsupervised loss** : real vs fake separation on unlabeled real data and generated data[1]

**Supervisor:** Blaise Hanczar

---

##  Project Summary

This project explores the power of **semi-supervised learning** using Generative Adversarial Networks (GAN). In a scenario where only **100 labeled images** (10 per class) are available out of the 60,000 in the MNIST dataset, we demonstrate how a **Semi-Supervised GAN (SGAN)** can drastically outperform a classic CNN.

### The Solution: Discriminator

The core of our approach lies in modifying the discriminator so that it does not just distinguish "real" from "fake," but acts as an 11-class classifier:

* **Classes 0-9:** Real handwritten digits.
* **Class 10:** Generated images ("Fake").

---

##  Comparative Performance

| Model | Labeled Data | Unlabeled Data | Test Accuracy (%) |
| --- | --- | --- | --- |
| **Baseline CNN** | 100 | None | 82.73% |
| **SGAN (K+1 + Feature Matching)** | 100 | **59,900** | **97.82%** |

---

## 🛠 Code Structure & Pipeline
=======
Total discriminator loss:

Implementation uses softmax over the K+1 logits, and constructs real/fake probabilities from the (K+1)-th logit.[1]

### Generator Loss — Feature Matching

The generator is trained **only** with **feature matching**:[1]

where  are intermediate features from the discriminator. This stabilizes training and avoids mode collapse.[1]

---

## Experimental Setup

* Dataset: MNIST, with **100 labeled** and **59 900 unlabeled** training images[1]
* Hardware: single **NVIDIA L4 GPU**, CUDA enabled[1]
* Framework: **PyTorch** - Reproducibility:
* Fixed random seed (Python, NumPy, torch, CUDA, deterministic cuDNN)[1]


* Data loaders:
* Labeled subset (balanced 10 per class) via `Subset`
* Unlabeled subset is the complement
* Dedicated loaders for labeled, unlabeled, and test sets[1]



### Training Control

* Optimizer: Adam (for both CNN and SGAN)[1]
* Early stopping:
* Best model at **epoch 62**, training stopped at **epoch 77** (no further improvement)[1]


* Best models stored as:
* `best_discriminator_sgan.pt`
* `best_generator_sgan.pt`[1]



---

## Results

### Quantitative Performance

| Model | Labeled Data | Unlabeled Data | Best Test Accuracy |
| --- | --- | --- | --- |
| **Baseline CNN (Supervised)** | 100 | 0 | 82.73% |
| **SGAN (K+1 + Feature Matching)** | 100 | 59 900 | **97.82%** |

Key observations:[1]

* Accuracy improves from ~86.6% at epoch 1 to >95% within the first 10 epochs.
* Peak accuracy: **97.82%** on MNIST test set (10 000 images).
* Accuracy stabilizes in a narrow band after convergence, indicating stable training.

### Discussion

* The **baseline CNN** overfits the 100 labels and cannot generalize as well.[1]
* The **SGAN**:
* Exploits the **structure of unlabeled data** through adversarial learning.
* Learns richer internal representations via the K+1 discriminator.
* Achieves performance close to fully supervised MNIST using **<0.2% labeled data**.[1]


* **Feature matching** acts as a strong regularizer and is crucial for stability.[1]
* **Early stopping** is essential in such low-label regimes to avoid overfitting.[1]

---

## Code Structure


```bash
mnist_100labels_gan/
├── main.py                  # Main notebook/script entry point
├── models/
│   ├── cnn_baseline.py       # Baseline model architecture
│   ├── gan_generator.py      # DCGAN-style Generator
│   └── gan_discriminator.py  # Discriminator (K+1 logits)
├── training/
│   ├── train_baseline.py     # Supervised training script
│   └── train_semisup_gan.py  # SGAN logic + Feature Matching
├── report/
│   └── report_neurips.pdf    # Final scientific report
└── main.py                   # Single entry point
=======
│   ├── cnn_baseline.py      # Baseline supervised CNN
│   ├── gan_generator.py     # SGAN generator
│   └── gan_discriminator.py # SGAN K+1 discriminator
├── training/
│   ├── train_baseline.py    # Supervised CNN training
│   └── train_semisup_gan.py # SGAN training (losses + feature matching)
├── utils/
│   ├── data_utils.py        # Dataset splits, loaders, visualization
│   ├── metrics.py           # Accuracy, confusion matrix, evaluation
│   └── seed_utils.py        # Reproducibility utilities
├── experiments/
│   └── results.json         # Logged metrics and best scores
└── report/
    └── report_neurips.pdf   # Final scientific report (NeurIPS format)

```

*(Adapter les chemins exacts à la structure réelle si besoin.)*[1]

---

##  Report Plan (Scientific Structure)

Below is the rigorous plan adopted for the writing of our article (NeurIPS format):

1. **Introduction**
* The problem of labeling costs.
* Motivation for using GANs in semi-supervised learning.
## How to Run

1. **Clone the repository**

```bash
git clone <URL_DU_REPO>.git
cd mnist_100labels_gan

```

2. **State of the Art & Baseline**
* Description of the supervised CNN.
* Analysis of overfitting in low-data regimes.
2. **Create environment & install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate  # ou: .venv\Scripts\activate sous Windows

3. **SGAN Methodology**
*  classifier architecture.
* Formulation of loss functions (Supervised vs. Unsupervised).
* **Feature Matching:** Technique for stabilizing Generator training.
pip install -r requirements.txt

```

4. **Implementation Details**
* Hyperparameters (Adam, learning rates, batch sizes).
* MNIST dataset management (100/59,900 split).
3. **Train the baseline CNN**

```bash
python -m training.train_baseline

```

4. **Train the Semi-Supervised GAN**

5. **Experimental Results**
* Convergence and accuracy curves.
* Visualization of images generated by the SGAN.
```bash
python -m training.train_semisup_gan

```

6. **Discussion & Analysis**
* Why does the SGAN generalize better?
* The role of structural information from unlabeled data.
or, with a single entry point if you l’as prévu:

```bash
python main.py

7. **Conclusion & Perspectives**
* Extensibility to more complex datasets (CIFAR-10).
=======
```

5. **Inspect results**
8. **References & Appendices**

---

##  How to Reproduce

1. Clone the repository.
2. Run the full training:
```bash
python main.py
```
3. View results in `/experiments/results.json`.
---

<p align="center"><i>Produced with rigor and passion by the GACKOU-LOUNISSI-NIRMAL-SAILLARD team.</i></p>

---
=======
* Metrics and logs: `experiments/results.json`
* Best models: `best_discriminator_sgan.pt`, `best_generator_sgan.pt`
* Generated samples and curves: figures saved in the experiments/report folders.[1]

---

## Reproducibility & Evaluation

* **Seed control**: all experiments use a fixed seed (`42`) applied to Python, NumPy, torch, and CUDA; cuDNN is set to deterministic.[1]
* **Classifier evaluation**:
* Only the first K logits (0–9) of the discriminator are used for classification.
* Accuracy and confusion matrix computed on the standard MNIST test set (10 000 images).[1]


* The discriminator can be reused as:
* A standalone classifier
* A feature extractor
* A pretrained backbone for other tasks[1]



---

## References

* T. Salimans et al., **“Improved Techniques for Training GANs”**, NeurIPS 2016.[1]
* I. Goodfellow et al., **“Generative Adversarial Networks”**, NeurIPS 2014.[1]
* Y. LeCun et al., **“Gradient-Based Learning Applied to Document Recognition”**, Proc. IEEE 1998.[1]
* **PyTorch documentation** — [https://pytorch.org/docs](https://pytorch.org/docs)[1]
* **MNIST dataset** — [http://yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist)[1]

---

*Produced with rigor and passion by the GACKOU–LOUNISSI–NIRMAL–SAILLARD team.*

Would you like me to help you generate a `requirements.txt` file or a specific README section for your model results?

