# Comparaison des algorithmes FedAvg et FedProx (fed_learning_health)

Ce projet présente une mini application d’évaluation de deux stratégies d’apprentissage fédéré : **FedAvg** et **FedProx**, appliquées à des données médicales simulées, en utilisant **PyTorch** et **Flower**. Une interface **Streamlit** est fournie pour visualiser les résultats.

N.B: Veuillez suivre rigouresement les indications ci-dessous : 👇

---

## Prérequis

- Python 3.10 ou + recommandé
- Navigateur web (Chrome, Firefox...)
- Connexion internet

---

## Objectif

Comparer les performances des deux stratégies (accuracy et perte) à travers plusieurs rounds d'entraînement répartis entre deux clients simulés.

---
## Cloner le projet 

Pour cloner le projet, utiliser ce script :

```bash
git clone https://github.com/JudicaelMAKWIZA/fed_learning_health.git
```

---

## Installation locale (manuelle)

1. Créer un environnement virtuel (optionnel)

```bash
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
```

La commande ci dessous permet d'installer toutes les dépendances nécessaires pour ce projet :

```bash
pip install -r requirements.txt
```

---

## Lancement de l'application

```bash
streamlit run app.py
```

---

## Structure du projet
```
├── app.py → Application Streamlit
├── client.py → Client fédéré Flower
├── server.py → Serveur FedAvg
├── server_fedprox.py → Serveur FedProx
├── dataset.py → Chargement des données MNIST
├── model.py → Modèle simple en PyTorch
├── simul.py → Génération de graphique accuracy/loss
├── requirements.txt → Dépendances
└── results/ → Fichiers CSV générés
```
