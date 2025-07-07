import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Chargement des fichiers CSV
fedavg_df = pd.read_csv("E:/Projet/fed_learning_health/fed_learning_health/results/resultats_fedavg.csv")
fedprox_df = pd.read_csv("E:/Projet/fed_learning_health/fed_learning_health/results/resultats_fedprox.csv")

# Forcer round en entier
fedavg_df["round"] = fedavg_df["round"].astype(int)
fedprox_df["round"] = fedprox_df["round"].astype(int)

# Calcul des moyennes par round (accuracy et loss)
fedavg_grouped = fedavg_df.groupby("round").mean(numeric_only=True)
fedprox_grouped = fedprox_df.groupby("round").mean(numeric_only=True)

# Création de la figure avec deux sous-graphiques
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy
ax1.plot(fedavg_grouped.index, fedavg_grouped["accuracy"], marker='o', label='FedAvg Accuracy')
ax1.plot(fedprox_grouped.index, fedprox_grouped["accuracy"], marker='s', label='FedProx Accuracy')
ax1.set_title("Accuracy moyenne par round")
ax1.set_xlabel("Round")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True)

# Loss réelle
ax2.plot(fedavg_grouped.index, fedavg_grouped["loss"], marker='o', linestyle='--', label='FedAvg Loss')
ax2.plot(fedprox_grouped.index, fedprox_grouped["loss"], marker='s', linestyle='--', label='FedProx Loss')
ax2.set_title("Loss moyenne par round")
ax2.set_xlabel("Round")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True)

# Affichage et sauvegarde
plt.tight_layout()
plt.savefig("fed_comparison_accuracy_loss.png")
plt.show()


'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Chargement des fichiers CSV
fedavg_df = pd.read_csv("E:/Projet/fed_learning_health/fed_learning_health/results/resultats_fedavg.csv")
fedprox_df = pd.read_csv("E:/Projet/fed_learning_health/fed_learning_health/results/resultats_fedprox.csv")

# Vérification : forcer round comme int
fedavg_df["round"] = fedavg_df["round"].astype(int)
fedprox_df["round"] = fedprox_df["round"].astype(int)

# Fonction pour simuler une perte décroissante
def simulate_loss(rounds):
    return np.linspace(1.0, 0.4, len(rounds))  # perte de 1.0 à 0.4

# Calculer accuracy moyenne par round
fedavg_acc = fedavg_df.groupby("round")["accuracy"].mean()
fedprox_acc = fedprox_df.groupby("round")["accuracy"].mean()

# Générer perte simulée
fedavg_loss = simulate_loss(fedavg_acc.index)
fedprox_loss = simulate_loss(fedprox_acc.index)

# Création de la figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy
ax1.plot(fedavg_acc.index, fedavg_acc.values, marker='o', label='FedAvg Accuracy')
ax1.plot(fedprox_acc.index, fedprox_acc.values, marker='s', label='FedProx Accuracy')
ax1.set_title("Accuracy moyenne par round")
ax1.set_xlabel("Round")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True)

# Loss simulée
ax2.plot(fedavg_acc.index, fedavg_loss, marker='o', label='FedAvg Loss (simulée)')
ax2.plot(fedprox_acc.index, fedprox_loss, marker='s', label='FedProx Loss (simulée)')
ax2.set_title("Loss moyenne par round (simulée)")
ax2.set_xlabel("Round")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True)

# Affichage et sauvegarde
plt.tight_layout()
plt.savefig("fed_comparison_accuracy_loss.png")
plt.show()
'''