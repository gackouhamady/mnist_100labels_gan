import matplotlib.pyplot as plt
import os

accuracy_file = "./experiments/accuracy_per_epoch.txt"
output_path = "./experiments/accuracy_curve.png"

epochs = []
accuracies = []

with open(accuracy_file, "r") as f:
    next(f)  # skip header
    for line in f:
        epoch, acc = line.strip().split(",")
        epochs.append(int(epoch))
        accuracies.append(float(acc))

plt.figure(figsize=(8, 5))
plt.plot(epochs, accuracies, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("SGAN Accuracy vs Epochs (100 labels)")
plt.grid(True)

os.makedirs("./experiments", exist_ok=True)
plt.tight_layout()
plt.savefig(output_path)
plt.close()

print(f"Accuracy curve saved to {output_path}")
