import requests
import pandas as pd
import numpy as np
from datetime import date
import os
import tempfile
from azure.storage.blob import BlobClient
import io
import requests
# ======================================================
# 1. Fonction pour récupérer l'historique Bitcoin (prix + volume)
# ======================================================

def save_df(df, local_path, blob_path):
    # 1. Sauvegarde du DataFrame en local
    df.to_csv(local_path, index=False)

    # 2. Connexion à Azure Blob Storage
    blob = BlobClient.from_connection_string(
        conn_str=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
        container_name="data",
        blob_name=blob_path
    )
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
    today_str = date.today().strftime("%Y-%m-%d")
    path = F'data/data.csv'
    path_blob = F'data_{today_str}.csv'
    save_df(df, path, path_blob)
    return df_feat