import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from datetime import datetime

def load_data():
    """Charge et prépare les données brutes"""
    # Chemins des fichiers
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    # Chargement des données
    x_train = pd.read_csv(os.path.join(data_dir, 'x_train_final.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train_final.csv'))
    x_test = pd.read_csv(os.path.join(data_dir, 'x_test_final.csv'))
    
    return x_train, y_train, x_test

def preprocess_data(x_train, x_test):
    """Prétraite les données"""
    # Conversion des dates
    x_train['date'] = pd.to_datetime(x_train['date'])
    x_test['date'] = pd.to_datetime(x_test['date'])
    
    # Extraction des caractéristiques temporelles
    x_train['jour_semaine'] = x_train['date'].dt.weekday
    x_train['heure'] = x_train['date'].dt.hour
    x_test['jour_semaine'] = x_test['date'].dt.weekday
    x_test['heure'] = x_test['date'].dt.hour
    
    # Création de nouvelles variables
    x_train['train_gare'] = x_train['train'].astype(str) + '_' + x_train['gare'].astype(str)
    x_test['train_gare'] = x_test['train'].astype(str) + '_' + x_test['gare'].astype(str)
    
    # Remplacement des valeurs manquantes par la médiane
    for col in ['p2q0', 'p3q0', 'p4q0', 'p0q2', 'p0q3', 'p0q4']:
        median = x_train[col].median()
        x_train[col].fillna(median, inplace=True)
        x_test[col].fillna(median, inplace=True)
    
    # Sélection des features
    features = ['jour_semaine', 'heure', 'p2q0', 'p3q0', 'p4q0', 'p0q2', 'p0q3', 'p0q4']
    
    return x_train[features], x_test[features]

def prepare_submission(predictions, output_path='submission.csv'):
    """Prépare le fichier de soumission"""
    # Création du DataFrame de soumission
    submission = pd.DataFrame({
        'id': range(len(predictions)),
        'target': predictions
    })
    
    # Sauvegarde
    submission.to_csv(output_path, index=False)
    return submission
