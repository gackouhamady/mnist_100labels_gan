Voici une version **exceptionnelle** et **fidèle** de votre projet, présentée sous forme de fichier Markdown (`README.md` ou `Main.md`) parfaitement structurée pour votre dépôt. Elle intègre vos noms, votre programme de Master à l'Université Paris Cité, ainsi que le plan détaillé du rapport final.

---

# # Semi-Supervised GAN for MNIST (100 Labels)

<p align="center">
<img alt="University Paris Cité" src="https://img.shields.io/badge/University-Paris%20Cité-6f42c1?style=for-the-badge&logo=academia&logoColor=white">
<img alt="Master ML for Data Science" src="https://img.shields.io/badge/Master-Machine%20Learning%20for%20Data%20Science-1976D2?style=for-the-badge&logo=python&logoColor=white">
<img alt="Deep Learning Project" src="https://img.shields.io/badge/Project-Deep%20Learning%20-%20Semi--Supervised%20GAN-FF9800?style=for-the-badge&logo=jupyter&logoColor=white">
<img alt="Academic Year" src="https://img.shields.io/badge/Year-2025%2F2026-009688?style=for-the-badge&logo=googlecalendar&logoColor=white">
</p>

---

## 👨‍🔬 Équipe Projet

**Université Paris Cité — Master 2 Machine Learning for Data Science**

* **Manel LOUNISSI** ([manel2.lounissi@gmail.com](mailto:manel2.lounissi@gmail.com))
* **Sandeep-Singh NIRMAL** ([nirmalsinghsandeep@gmail.com](mailto:nirmalsinghsandeep@gmail.com))
* **Brice SAILLARD** ([brice.saillard.bs@gmail.com](mailto:brice.saillard.bs@gmail.com))
* **Hamady GACKOU** ([hamady.gackou@etu.u-paris.fr](mailto:hamady.gackou@etu.u-paris.fr))

**Superviseur :** Blaise Hanczar

---

## 🎯 Résumé du Projet

Ce projet explore la puissance de l'apprentissage **semi-supervisé** à l'aide de réseaux antagonistes génératifs (GAN). Dans un scénario où seulement **100 images étiquetées** (10 par classe) sont disponibles sur les 60 000 du dataset MNIST, nous démontrons comment un **Semi-Supervised GAN (SGAN)** peut surpasser drastiquement un CNN classique.

### La Solution : Discriminateur 

Le cœur de notre approche réside dans la modification du discriminateur pour qu'il ne se contente pas de distinguer le "vrai" du "faux", mais qu'il agisse comme un classificateur à 11 classes :

* **Classes 0-9 :** Chiffres manuscrits réels.
* **Classe 10 :** Images générées ("Fake").

---

## 📊 Performances Comparatives

| Modèle | Données Étiquetées | Données Non-Étiquetées | Précision Test (%) |
| --- | --- | --- | --- |
| **Baseline CNN** | 100 | Non | 82.73% |
| **SGAN (K+1 + Feature Matching)** | 100 | **59,900** | **97.82%** |

---

## 🛠 Structure du Code & Pipeline

```bash
mnist_100labels_gan/
├── models/
│   ├── cnn_baseline.py       # Architecture du modèle témoin
│   ├── gan_generator.py      # Générateur DCGAN-style
│   └── gan_discriminator.py  # Discriminateur (K+1 logits)
├── training/
│   ├── train_baseline.py     # Script d'entraînement supervisé
│   └── train_semisup_gan.py  # Logique SGAN + Feature Matching
├── report/
│   └── report_neurips.pdf    # Rapport scientifique final
└── main.py                   # Point d'entrée unique

```

---

## 📝 Plan de Rapport (Structure Scientifique)

Voici le plan rigoureux adopté pour la rédaction de notre article (format NeurIPS) :

1. **Introduction**
* Problématique du coût de l'étiquetage.
* Motivation pour l'utilisation des GANs en semi-supervisé.


2. **État de l'art & Baseline**
* Description du CNN supervisé.
* Analyse du sur-apprentissage (overfitting) en régime de faibles données.


3. **Méthodologie SGAN**
* Architecture du classificateur .
* Formulation des fonctions de perte (Supervised vs Unsupervised).
* **Feature Matching :** Technique de stabilisation de l'entraînement du Générateur.


4. **Détails d'Implémentation**
* Hyperparamètres (Adam, learning rates, batch sizes).
* Gestion du dataset MNIST (Split 100/59,900).


5. **Résultats Expérimentaux**
* Courbes de convergence et d'accuracy.
* Visualisation des images générées par le SGAN.


6. **Discussion & Analyse**
* Pourquoi le SGAN généralise-t-il mieux ?
* Rôle de l'information structurelle des données non-étiquetées.


7. **Conclusion & Perspectives**
* Extensibilité à des datasets plus complexes (CIFAR-10).


8. **Références & Annexes**

---

## 🚀 Comment Reproduire

1. Cloner le dépôt.
2. Installer les dépendances : `pip install -r requirements.txt`.
3. Lancer l'entraînement complet :
```bash
python main.py --mode all --labels 100

```


4. Consulter les résultats dans `/experiments/results.json`.

---

<p align="center"><i>Réalisé avec rigueur et passion par l'équipe Gackou-Lounissi-Nirmal-Saillard.</i></p>

---

Souhaitez-vous que je développe davantage une section spécifique du rapport (par exemple, la démonstration mathématique de la perte du discriminateur) ?
