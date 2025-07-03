import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

from ..data.prepare_data import load_data, preprocess_data

def train_models():
    """Entraîne et évalue différents modèles de prédiction"""
    # Chargement des données
    x_train, y_train, _ = load_data()
    
    # Prétraitement
    x_train_processed, _ = preprocess_data(x_train, pd.DataFrame())
    
    # Séparation en train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        x_train_processed, y_train, test_size=0.2, random_state=42
    )
    
    # Modèles à tester
    models = {
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        },
        'XGBoost': {
            'model': XGBRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        },
        'LightGBM': {
            'model': LGBMRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [-1, 5, 10],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        }
    }
    
    # Dictionnaire pour stocker les résultats
    results = {}
    
    # Entraînement et évaluation de chaque modèle
    for model_name, model_info in models.items():
        print(f"\nEntraînement du modèle {model_name}...")
        
        # Grid Search pour optimisation des hyperparamètres
        grid_search = GridSearchCV(
            model_info['model'],
            model_info['params'],
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Meilleur modèle
        best_model = grid_search.best_estimator_
        
        # Évaluation sur le validation set
        y_pred = best_model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        
        # Sauvegarde du modèle
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', f'{model_name}_model.joblib')
        joblib.dump(best_model, model_path)
        
        # Stockage des résultats
        results[model_name] = {
            'best_params': grid_search.best_params_,
            'mae': mae,
            'model_path': model_path
        }
        
        print(f"{model_name} - MAE: {mae:.2f}")
    
    return results
