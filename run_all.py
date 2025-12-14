import os
import subprocess
import sys

def run(cmd):
    print("\n" + "=" * 80)
    print(f"Running: {cmd}")
    print("=" * 80)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n❌ Error while running: {cmd}")
        sys.exit(1)

if __name__ == "__main__":
    print("\n🚀 START FULL MNIST SGAN PIPELINE\n")

    # 1. Train semi-supervised GAN (50 epochs)
    run("python training/train_semisup_gan.py")

    # 2. Generate samples from trained generator
    run("python training/generate_samples.py")

    # 3. Evaluate classifier + save confusion matrix & accuracy
    run("python training/evaluate_classifier.py")

    # 4. Plot accuracy curve
    run("python training/plot_accuracy_curve.py")

    print("\n✅ ALL DONE SUCCESSFULLY")
    print("Check the ./experiments/ folder for results.\n")

 #experiments/
 #├── gan_discriminator.pt
 #├── gan_generator.pt
 #├── results_sgan.txt<
 #├── confusion_matrix_sgan.txt
 #├── confusion_matrix_sgan.png
 #├── accuracy_per_epoch.txt
 #├── accuracy_curve.png
 #└── gan_samples_grid.png


#gan_discriminator.pt      -> modèle final
#gan_generator.pt          -> générateur
#results_sgan.txt          -> accuracy finale
#accuracy_per_epoch.txt    -> courbe d’apprentissage
#accuracy_curve.png        -> figure pour le rapport
#confusion_matrix_sgan.txt -> matrice brute
#confusion_matrix_sgan.png -> figure pour le rapport
#gan_samples_grid.png      -> qualité visuelle du GAN


#📁 datasets/ → préparation des 100 labels

#📁 models/ → architectures

#📁 training/ → entraînement / évaluation

#📁 experiments/ → TOUS les résultats automatiques

#📁 reports/ → rapport final (LaTeX + PDF)