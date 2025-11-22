import requests
import pandas as pd
import numpy as np
import mlflow
from datetime import date
import os
import tempfile
# ======================================================
# 1. Fonction pour récupérer l'historique Bitcoin (prix + volume)
# ======================================================

def get_btc_history(days=365):
    """
    Récupère les prix et volumes BTC/EUR sur X jours (données daily).
    Retourne un DataFrame brut : timestamp, price, volume, date.
    """
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "eur", "days": days}

    r = requests.get(url, params=params)
    data = r.json()
    d = pd.DataFrame(data)
    print(d.columns)
    prices = data["prices"]          # [timestamp_ms, price]
    volumes = data["total_volumes"]  # [timestamp_ms, volume]

    df_price = pd.DataFrame(prices, columns=["timestamp", "price"])
    df_vol = pd.DataFrame(volumes, columns=["timestamp", "volume"])

    df = df_price.copy()
    df["volume"] = df_vol["volume"]
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["date"] = df["time"].dt.date

    return df[["date", "price", "volume"]]

# ======================================================
# 2. Calcul des 10 features pour le machine learning
# ======================================================

def add_features(df):
    df = df.copy()

    # Variation 1 jour (%)
    df["return_1d"] = df["price"].pct_change() * 100

    # Variation 3 jours (%)
    df["return_3d"] = df["price"].pct_change(3) * 100

    # Variation 7 jours (%)
    df["return_7d"] = df["price"].pct_change(7) * 100

    # Variation 14 jours (%)
    df["return_14d"] = df["price"].pct_change(14) * 100

    # Moyennes mobiles
    df["MA7"]  = df["price"].rolling(window=7).mean()
    df["MA30"] = df["price"].rolling(window=30).mean()

    # Diff MA7-MA30
    df["MA_diff"] = df["MA7"] - df["MA30"]

    # Volatilité
    df["vol_7d"] = df["return_1d"].rolling(window=7).std()

    # Volume MA7
    df["vol_MA7"] = df["volume"].rolling(window=7).mean()

    # Plus haut / bas 30 jours
    df["highest_30"] = df["price"].rolling(30).max()
    df["lowest_30"]  = df["price"].rolling(30).min()

    # Position dans le canal
    df["pos_channel_30"] = (
        (df["price"] - df["lowest_30"]) /
        (df["highest_30"] - df["lowest_30"])
    )

  
    

    # On supprime les colonnes originales 
    df = df.dropna(axis=1, how='all')


    return df


# ======================================================
# 3. Construction du dataset final
# ======================================================


def build_dataset(days=365):
    df_raw = get_btc_history(days)
    df_feat = add_features(df_raw)
    sandmlflowdata(df_feat)
    return df_feat

def sandmlflowdata(df):
    # 1. Nom de l'expérience
   experiment_name = "Data"
   mlflow.set_experiment(experiment_name)

# 2. Nom du run = Data_YYYY-MM-DD
   today_str = date.today().strftime("%Y-%m-%d")
   run_name = f"Data_{today_str}"

   with mlflow.start_run(run_name=run_name):
    # (optionnel) log de quelques infos
    mlflow.log_param("source", "pipeline_daily")
    mlflow.set_tag("date", today_str)

    # 3. Sauver le df temporairement en CSV puis l'envoyer comme artifact
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, f"data_{today_str}.csv")
        df.to_csv(file_path, index=False)

        # artifact_path = dossier logique dans MLflow
        mlflow.log_artifact(file_path, artifact_path="dataframe")

def load_latest_data(experiment_name: str, run_name):
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
        df = build_dataset()
        # recréer l'expérience après entraînement
        experiment = mlflow.get_experiment_by_name(experiment_name)

    # maintenant on est sûr que l'expérience existe
    experiment_id = experiment.experiment_id


    # petite fonction interne pour chercher le dernier modèle
    def _search_and_load():
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=(
                f"and tags.mlflow.runName = '{run_name}' "
                f"and attributes.status = 'FINISHED'"
            ),
            max_results=1,
        )
        if runs.empty:
            return None

        run_id = runs.iloc[0].run_id
        client = MlflowClient()
        local_path = client.download_artifacts(run_id, "dataframe")
        csv_files = [f for f in os.listdir(local_path) if f.endswith(".csv")]
        if not csv_files:
                return None  # aucun CSV trouvé = problème
        csv_path = os.path.join(local_path, csv_files[0])  # premier CSV trouvé
        df = pd.read_csv(csv_path)
        return df

    # --------------------------------------
    # 2) Première tentative de chargement
    # --------------------------------------
    try:
        df = _search_and_load()
    except Exception:
        df = None

    # --------------------------------------
    # 3) Si rien trouvé → build + train
    # --------------------------------------
    if df is None:  
        df = build_dataset()

    return df

