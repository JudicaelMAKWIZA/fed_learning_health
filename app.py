import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

st.set_page_config(page_title="Simulation Apprentissage Fédéré", layout="wide")

# Chemins des fichiers
REAL_PATHS = {
    "FedAvg": "results/resultats_fedavg.csv",
    "FedProx": "results/resultats_fedprox.csv"
}
SIMUL_PATHS = {
    "FedAvg": "results/resultats_fedavg_simul.csv",
    "FedProx": "results/resultats_fedprox_simul.csv"
}

# Fonction pour charger les données
def load_data(algo, use_simulated=False):
    path = SIMUL_PATHS[algo] if use_simulated else REAL_PATHS[algo]
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

# Fonction pour simuler et écrire des données
def simulate_data():
    rounds = [1, 2, 3]
    clients = [0, 1]

    data = {
        "FedAvg": [],
        "FedProx": []
    }

    for r in rounds:
        for c in clients:
            acc_avg = round(0.85 + np.random.rand() * 0.05, 4)
            acc_prox = round(0.87 + np.random.rand() * 0.05, 4)
            loss_avg = round(0.5 - np.random.rand() * 0.1, 4)
            loss_prox = round(0.48 - np.random.rand() * 0.1, 4)
            data["FedAvg"].append([r, c, acc_avg, loss_avg, "FedAvg"])
            data["FedProx"].append([r, c, acc_prox, loss_prox, "FedProx"])

    os.makedirs("results", exist_ok=True)
    for algo in ["FedAvg", "FedProx"]:
        filename = SIMUL_PATHS[algo]
        with open(filename, "w", newline="") as f:
            f.write("round,client_id,accuracy,loss,algorithm\n")
            for row in data[algo]:
                f.write(",".join(map(str, row)) + "\n")

    st.success("Les données simulées ont été enregistrées."
    "Assurez-vous que le mode de visualisation *Données simulées* à gauche est sélectionné pour les visualiser.")

# Fonction pour supprimer les fichiers simulés
def delete_simulated_files():
    deleted = False
    for file in SIMUL_PATHS.values():
        if os.path.exists(file):
            os.remove(file)
            deleted = True
    return deleted

# Titre principal
st.title("Comparaison des algorithmes FedAvg et FedProx")

# Message explicatif 
st.info(
    "🎯 Cette application permet de **visualiser les résultats de l'apprentissage fédéré** "
    "réalisé avec les algorithmes **FedAvg** et **FedProx**, à l'aide du framework **Flower**.\n\n"
    
    "📊 Deux métriques principales sont affichées : la **précision (accuracy)** et la **perte (loss)**, "
    "évoluant au fil des rounds d'entraînement.\n\n"

    "📁 Les données peuvent provenir de deux sources :\n"
    "- **Données réelles** : résultats éxperimentales enregistrés automatiquement lors d'entraînements fédérés réels avec Flower (voir rapport),\n"
    "- **Données simulées** : résultats fictifs que vous pouvez générés à des fins de démonstration en cliquant sur le bouton *Simuler les rounds avec donnees fictives*.\n\n"

    "🧠 Lors des entraînements réels(éxperimentation expliquée dans le rapport), seules les **valeurs d’accuracy** ont pu être récupérées automatiquement. "
    "Les **valeurs de perte (loss)** n’ont pas été enregistrées car la stratégie utilisée dans Flower ne renvoyait pas cette information. "
    "Pour une meilleure expérience visuelle, des pertes **représentatives ont été simulées** pour accompagner les courbes.\n\n"

    "⚙️ Dans la barre latérale (à gauche) :\n"
    "- Utilisez le sélecteur *Mode de visualisation* pour basculer entre les données réelles(expérimentales) et simulées,\n"
    "- Cliquez sur *Simuler les rounds avec données fictives* pour générer de nouvelles données fictives,\n"
    "- Cliquez sur *Supprimer les données simulées* pour effacer les fichiers simulés si besoin.\n\n"

    "📌 Les résultats sont affichés sous forme de **courbes comparatives** permettant d’analyser les performances de chaque algorithme."
)


# Barre latérale
mode = st.sidebar.radio("Mode de visualisation", ("Données expérimentales", "Données simulées"))

# Boutons d'action
if st.sidebar.button("Simuler les rounds avec données fictives"):
    simulate_data()

if st.sidebar.button("Supprimer les données simulées"):
    if delete_simulated_files():
        st.success("Les fichiers simulés ont été supprimés.")
    else:
        st.warning("Aucun fichier simulé à supprimer.")

use_simulated = mode == "Données simulées"

# Chargement des données
fedavg_df = load_data("FedAvg", use_simulated)
fedprox_df = load_data("FedProx", use_simulated)

if fedavg_df is not None and fedprox_df is not None:
    fedavg_grouped = fedavg_df.groupby("round").mean(numeric_only=True)
    fedprox_grouped = fedprox_df.groupby("round").mean(numeric_only=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(fedavg_grouped.index, fedavg_grouped["accuracy"], marker='o', label="FedAvg Accuracy")
    ax1.plot(fedprox_grouped.index, fedprox_grouped["accuracy"], marker='s', label="FedProx Accuracy")
    ax1.set_title("Accuracy moyenne par round")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(fedavg_grouped.index, fedavg_grouped["loss"], marker='o', linestyle='--', label="FedAvg Loss")
    ax2.plot(fedprox_grouped.index, fedprox_grouped["loss"], marker='s', linestyle='--', label="FedProx Loss")
    ax2.set_title("Loss moyenne par round")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Loss")
    ax2.grid(True)
    ax2.legend()

    st.pyplot(fig)
else:
    st.warning("Les fichiers CSV sont introuvables ou vides. Lance une simulation de l'entraînement réel en cliquant sur le bouton 'Simuler les rounds avec données fictives'.")

