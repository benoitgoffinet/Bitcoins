import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mlflow
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

def sandmlflow(experience, run, model, f1_cv, f1_test, acc_test, typemodel, cible):
     # 1. Nom de l'expérience
 today_str = date.today().strftime("%Y-%m-%d")
 experiment_name = experience
 mlflow.set_experiment(experiment_name)

# 2. Nom du run = Data_YYYY-MM-DD
 run_name = run
 with mlflow.start_run(run_name=run_name):
    # --- PARAMS / TAGS ---
    mlflow.log_param("model_type", typemodel)
    mlflow.log_param("source", "pipeline_daily")
    mlflow.set_tag("date", today_str)

    # (optionnel) log des hyperparamètres du modèle
    if hasattr(model, "get_params"):
        params = model.get_params()
        # tu peux filtrer si tu ne veux pas tout envoyer
        for k, v in params.items():
            mlflow.log_param(k, v)

    # --- METRIQUES ---
    mlflow.log_metric("f1_macro_cv", f1_cv)
    mlflow.log_metric("f1_macro_test", f1_test)
    mlflow.log_metric("accuracy_test", acc_test)
     # --- MODÈLE ENREGISTRÉ COMME ARTIFACT ---
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=None  # ou un nom global si tu utilises le Model Registry
    )




def train(dexplicative, target, ciblename):
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
 cible = ciblename
 experience = f"Model{cible}"
 typemodel = 'Randomforest'
 run = f"{experience}{today_str}"
 model = best_model
 f1_cv = grid_search.best_score_
 sandmlflow(experience, run, model, f1_cv, f1_macro, accuracy, typemodel, cible)


def load_latest_model(targetname: str, experiment_name: str, run_name, df):
    """
    - Vérifie si l'expérience existe
    - Vérifie si un run existe
    - Vérifie si le modèle se charge
    - Si quelque chose manque → build_data() + train()
    """

    # ---------------------------
    # 1) Vérifier si l'expérience existe
    # ---------------------------
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        # si l’expérience n’existe pas → build + train
              ndf = df.copy()
              ndf["target_up"] = (ndf["price"].shift(-3) > ndf["price"]).astype(int)
              ndf = ndf.iloc[:-3]
              target = ndf['target_up']
              dataexplicative = ndf.drop(columns=['date', 'target_up'])
              model = train(dataexplicative, target, targetname)
              return model

    # maintenant on est sûr que l'expérience existe
    experiment_id = experiment.experiment_id


    # petite fonction interne pour chercher le dernier modèle
    def _search_and_load():
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=(
                f"tags.mlflow.runName = '{run_name}' "
                f"and attributes.status = 'FINISHED'"
            ),
            max_results=1,
        )
        if runs.empty:
            return None

        run_id = runs.iloc[0].run_id
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        return model

    # --------------------------------------
    # 2) Première tentative de chargement
    # --------------------------------------
    try:
        model = _search_and_load()
    except Exception:
        model = None

    # --------------------------------------
    # 3) Si rien trouvé → build + train
    # --------------------------------------
    if model is None: 
        ndf = df.copy()
        ndf["target_up"] = (ndf["price"].shift(-3) > ndf["price"]).astype(int)
        ndf = ndf.iloc[:-3]
        target = ndf['target_up']
        dataexplicative = ndf.drop(columns=['date', 'target_up'])
        ndf["target_up"] = (ndf["price"].shift(-7) > ndf["price"]).astype(int)
        ndf = ndf.iloc[:-7]
        target = ndf['target_up']
        dataexplicative = ndf.drop(columns=['date', 'target_up'])
        model = train(dataexplicative, target, targetname)
    return model


