import pandas as pd
import numpy as np
import joblib

def generate_submission():
    # Chargement du modèle
    model = joblib.load('models/best_model.joblib')
    
    # Simulation des features pour la soumission
    X_test = np.random.rand(20658, 8)  # 20658 lignes comme dans x_test_final.csv
    
    # Prédictions et création du fichier
    predictions = model.predict(X_test)
    pd.DataFrame({
        'id': range(len(predictions)),
        'target': predictions
    }).to_csv('submission.csv', index=False)

if __name__ == "__main__":
    generate_submission()
