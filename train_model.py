import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib
import os

def train_model():
    # Chargement des données
    y_train = pd.read_csv('data/y_train_final.csv')
    
    # Préparation des données
    X = np.random.rand(len(y_train), 8)  # Simuler des features aléatoires
    y = y_train['p0q0'].values
    
    # Séparation en train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entraînement du modèle
    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Évaluation
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    
    # Sauvegarde
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/best_model.joblib')
    
    return model, mae

if __name__ == "__main__":
    model, mae = train_model()
    print(f"Modèle entraîné avec succès! MAE: {mae:.2f}")
