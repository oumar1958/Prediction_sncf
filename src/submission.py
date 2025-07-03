import pandas as pd
import numpy as np
import joblib
import os

from .data.prepare_data import load_data, preprocess_data
from .models.train_models import train_models

def generate_submission():
    """Génère la soumission finale"""
    # Entraînement des modèles
    results = train_models()
    
    # Chargement des données
    x_train, y_train, x_test = load_data()
    
    # Prétraitement des données de test
    _, x_test_processed = preprocess_data(x_train, x_test)
    
    # Sélection du meilleur modèle (celui avec le MAE le plus bas)
    best_model_name = min(results, key=lambda k: results[k]['mae'])
    print(f"\nMeilleur modèle: {best_model_name}")
    
    # Chargement du meilleur modèle
    model_path = results[best_model_name]['model_path']
    best_model = joblib.load(model_path)
    
    # Prédiction sur les données de test
    predictions = best_model.predict(x_test_processed)
    
    # Préparation du fichier de soumission
    submission_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'submission.csv')
    prepare_submission(predictions, submission_path)
    
    print(f"\nSoumission générée avec succès à {submission_path}")
    print(f"MAE du meilleur modèle: {results[best_model_name]['mae']:.2f}")
    
    return predictions
