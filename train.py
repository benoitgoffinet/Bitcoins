import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder  
from datetime import datetime, date
import joblib
from azure.storage.blob import BlobClient
import io
import requests


def save_model_pkl(model, local_path, blob_path):
    """
    Sauvegarde un modèle en .pkl localement puis le charge dans Azure (container: models)
    model      : objet Python (sklearn, dict, etc.)
    local_path : chemin local pour écrire le fichier .pkl
    blob_path  : chemin dans Azure Blob Storage (models/model.pkl)
    """
    # 1. Sauvegarde du modèle en local
    with open(local_path, "wb") as f:
        pickle.dump(model, f)

    # 2. Connexion Azure
    blob = BlobClient.from_connection_string(
        conn_str=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
        container_name="models",  # <-- ton conteneur pour les modèles
        blob_name=blob_path
    )

    # 3. Upload du fichier
    with open(local_path, "rb") as f:
        blob.upload_blob(f, overwrite=True)

    print(f"✔ Modèle sauvegardé localement : {local_path}")
    print(f"✔ Modèle uploadé dans Azure    : models/{blob_path}")




def train(dexplicative, target):
    # Variables explicatives et cible
 
 X = dexplicative
 y = target



# 🎲 Séparation des données avec stratification basée sur les quartiles de la cible
 X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42#, stratify=y_quartiles  # Stratification sur y_quartiles
)

# 🔧 Pipeline de prétraitement
# Nous remplaçons le OneHotEncoder par un TargetEncoder pour les variables catégorielles
 preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X_train.select_dtypes(exclude='object').columns.tolist()),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), X_train.select_dtypes(include='object').columns.tolist())
    ]
)

# 💡 Pipeline complet avec GridSearchCV
 pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', RandomForestClassifier(random_state=42))  # Modèle ElasticNet
])

# Paramètres pour GridSearchCV
 param_grid = {
    'model__max_depth': [10, 5, 20, 30],  # Using model__ to specify RandomForestRegressor
    'model__n_estimators': [10, 50, 100, 200],  # Add more parameters for the model if needed
    'model__min_samples_split': [3, 2, 4, 5, 6]
    # Add other hyperparameters for preprocessing steps if needed
}

# GridSearchCV
 grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring='f1_macro',
    cv=5,
    verbose=1,
    n_jobs=-1
)

# 🚀 Entraînement avec GridSearchCV
 grid_search.fit(X_train, y_train)

# ✅ Meilleur modèle trouvé
 best_model = grid_search.best_estimator_

 print(f"✅ Meilleur alpha trouvé : {grid_search.best_params_}")
 print(f"📊 Meilleur f1_macro (validation croisée) : {grid_search.best_score_:.4f}")

# 🟢 Prédictions sur l'ensemble de test
 y_pred = best_model.predict(X_test)

 accuracy = accuracy_score(y_test, y_pred)
 f1_macro = f1_score(y_test, y_pred, average='macro')
 f1_weighted = f1_score(y_test, y_pred, average='weighted')

 print("📊 Évaluation sur TEST")
 print(f"🎯 Accuracy        : {accuracy:.4f}")
 print(f"💠 F1 Macro        : {f1_macro:.4f}")
 print(f"🔷 F1 Weighted     : {f1_weighted:.4f}")
 # 1. Nom de l'expérience
 today_str = date.today().strftime("%Y-%m-%d")
 model = best_model
 f1_cv = grid_search.best_score_
 pathmodel = F'model/model_{today_str}.pkl'
 local_path = F'model/model.pkl'
 blob_path = F'model_{today_str}.pkl'
 ave_model_pkl(model, local_path, blob_path):