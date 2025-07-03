import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib
import os

# Chargement des données
def load_data():
    x_train = pd.read_csv('data/x_train_final.csv')
    y_train = pd.read_csv('data/y_train_final.csv')
    
    # Prétraitement
    x_train['date'] = pd.to_datetime(x_train['date'])
    x_train['jour_semaine'] = x_train['date'].dt.weekday
    x_train['heure'] = x_train['date'].dt.hour
    
    # Sélection des features
    features = ['jour_semaine', 'heure', 'p2q0', 'p3q0', 'p4q0', 'p0q2', 'p0q3', 'p0q4']
    
    return x_train[features], y_train

# Entraînement et optimisation du modèle
def train_model():
    # Chargement des données
    X, y = load_data()
    
    # Séparation en train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Paramètres pour GridSearch
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    }
    
    # GridSearchCV
    grid_search = GridSearchCV(
        XGBRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Entraînement
    grid_search.fit(X_train, y_train)
    
    # Meilleur modèle
    best_model = grid_search.best_estimator_
    
    # Évaluation
    y_pred = best_model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    
    print(f"Meilleurs paramètres: {grid_search.best_params_}")
    print(f"MAE sur validation: {mae:.2f}")
    
    # Sauvegarde du modèle
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.joblib')
    
    return best_model, mae

if __name__ == "__main__":
    model, mae = train_model()
    print("\nModèle entraîné et sauvegardé avec succès!")
