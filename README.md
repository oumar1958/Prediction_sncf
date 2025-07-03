# Projet de Prévision des Temps d'Attente des Trains Transilien

## Description du Projet
Ce projet vise à prédire les temps d'attente des trains Transilien SNCF en utilisant un modèle XGBoost.

## Structure du Projet
```
train_prediction_project/
├── data/              # Données brutes et préparées
├── models/           # Modèle entraîné
├── requirements.txt   # Dépendances du projet
├── train_model.py    # Script d'entraînement
├── generate_submission.py # Script de génération de soumission
└── README.md         # Documentation du projet
```

## Installation
```bash
pip install -r requirements.txt
```

## Utilisation
1. Entraîner le modèle :
```bash
python train_model.py
```

2. Générer la soumission :
```bash
python generate_submission.py
```

## Données
Les données nécessaires sont :
- `y_train_final.csv` : Données d'entraînement (cible)
- `y_sample_final.csv` : Exemple de format de soumission
