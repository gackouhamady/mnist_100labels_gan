# Semi-Supervised GAN for MNIST (100 Labels)
Using GAN Discriminators as K+1 Classifiers for Low-Label Deep Learning

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![Status](https://img.shields.io/badge/Status-Ongoing-yellow)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-blue)
![Labels](https://img.shields.io/badge/Labels-Only%20100-important)
![Task](https://img.shields.io/badge/Task-Semi--Supervised%20Learning-green)
![Model](https://img.shields.io/badge/Model-SGAN%20(K%2B1%20classes)-purple)
![Reproducibility](https://img.shields.io/badge/Reproducibility-Guaranteed-brightgreen)


# Project Overview
This project aims to classify MNIST digits using only 100 labeled examples, leveraging a Semi-Supervised GAN (SGAN) inspired by Salimans et al., 2016.  
The key idea is to transform the GAN Discriminator into a (K+1)-class classifier:

- Classes 0–9 = real digits  
- Class 10 = Fake / Generated  

This allows the Discriminator to learn from:
- Labeled data (supervised)
- Unlabeled real data (unsupervised)
- Generated samples (GAN adversarial training)

The final trained Discriminator is reused as a classifier on 10 classes.

# Key Objectives
- Build a supervised baseline CNN (100 labels only)
- Implement a Semi-Supervised GAN (K+1 Discriminator)
- Use Feature Matching for Generator stability
- Compare baseline vs SGAN performance
- Deliver a clean, reproducible deep learning pipeline

# Directory Structure
```bash
mnist_100labels_gan/
├── data/                      
├── models/
│   ├── cnn_baseline.py       
│   ├── gan_generator.py      
│   └── gan_discriminator.py  
├── datasets/
│   └── mnist_100_labels.py    
├── training/
│   ├── train_baseline.py     
│   └── train_semisup_gan.py  
├── utils/
│   ├── seed.py
│   ├── metrics.py
│   └── vis.py
├── experiments/
│   ├── logs_tensorboard/
│   └── results.json
├── report/
│   └── report.tex
└── main.py                   
```
# Complete Project Plan (A → Z)

-----------------------------------------------------
PHASE 0 — SETUP & DATASET PREPARATION
-----------------------------------------------------
Tools: Python 3.10, PyTorch, torchvision, TensorBoard  
Tasks:
- Setup repository structure
- Download MNIST
- Create split:
  - 100 labeled samples (10 per class)
  - 59,900 unlabeled samples

-----------------------------------------------------
PHASE 1 — BASELINE CNN (100 LABELS SUPERVISED)
-----------------------------------------------------
Architecture:
- 2 Conv blocks
- ReLU + MaxPool
- FC classifier → 10 logits

Training:
- Loss: CrossEntropy
- Optimizer: Adam
- Epochs: 50–100
- Expected accuracy: 60–80%

Purpose:
- Establish baseline performance

-----------------------------------------------------
PHASE 2 — SEMI-SUPERVISED GAN (SGAN)
-----------------------------------------------------

### 2.1 Discriminator (K+1 Class Classifier)
Output shape: 11 logits  
- logits[0..9] → real MNIST classes  
- logits[10] → Fake class

Returns intermediate features for Feature Matching.

### 2.2 Generator
DCGAN-style:
- Fully connected → reshape → ConvTranspose2d
- Output: 28×28 image

### 2.3 Loss Functions
A) Supervised loss (on 100 labels):
CrossEntropy on logits[0..9].

B) Unsupervised loss:
- Real unlabeled images → should be NOT fake  
- Generated images → should be fake

C) Generator Feature Matching loss:
L_G = || mean(f(real)) − mean(f(fake)) ||²

### 2.4 Training Loop
1. Train Discriminator:
   - L_sup (labeled)
   - L_unsup_real (unlabeled)
   - L_unsup_fake (generated)
2. Train Generator using Feature Matching
3. Repeat for ~200-300 epochs

-----------------------------------------------------
PHASE 3 — EVALUATION
-----------------------------------------------------

Comparisons:

Model                     | Labeled | Unlabeled | Expected Accuracy
-------------------------|---------|-----------|-------------------
Baseline CNN             | 100     | No        | 60–80%
CNN + Augmentation       | 100     | No        | 75–85%
SGAN (this project)      | 100     | Yes       | >90%

Artifacts:
- Accuracy curves  
- Loss curves  
- Generated samples  
- Confusion matrices  

-----------------------------------------------------
PHASE 4 — REPORT
-----------------------------------------------------
Sections:
1. Introduction  
2. Baseline method  
3. SGAN methodology (K+1 classifier, feature matching)  
4. Implementation details  
5. Experiments and results  
6. Discussion  
7. Conclusion  
8. Appendix: code excerpts  

# Key Insights
- Handling weak supervision  
- Leveraging GANs beyond generation  
- Feature Matching for stabilization  
- Combining supervised & unsupervised training  
- Designing a reproducible ML pipeline  

# Technologies Used
- PyTorch  
- Torchvision  
- TensorBoard  
- NumPy / Pandas  
- Matplotlib / Seaborn  

# Final Outcome
A complete M2-level deep learning project that:
- Demonstrates semi-supervised learning with only 100 labels
- Implements SGAN with K+1 Discriminator
- Produces significant performance gains over baseline
- Provides a fully reproducible pipeline
- Meets academic and industrial standards

