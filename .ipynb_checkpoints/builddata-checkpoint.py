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

    # 1) RETURNS (log-returns + pct)
    # ============================
    df["return_1d"] = df["price"].pct_change(1)
    df["return_3d"] = df["price"].pct_change(3)
    df["return_7d"] = df["price"].pct_change(7)
    df["return_14d"] = df["price"].pct_change(14)
    df["return_30d"] = df["price"].pct_change(30)

    df["log_return_1d"] = np.log(df["price"] / df["price"].shift(1))

    # ============================
    # 2) MOVING AVERAGES
    # ============================
    df["MA7"] = df["price"].rolling(7).mean()
    df["MA30"] = df["price"].rolling(30).mean()

    df["MA_diff"] = df["MA7"] - df["MA30"]
    df["MA7_over_MA30"] = df["MA7"] / df["MA30"]

    # Slopes
    df["slope_MA7"] = df["MA7"].diff()
    df["slope_MA30"] = df["MA30"].diff()

    # ============================
    # 3) VOLATILITY (from returns)
    # ============================
    df["vol_7d"] = df["return_1d"].rolling(7).std()
    df["vol_14d"] = df["return_1d"].rolling(14).std()
    df["vol_30d"] = df["return_1d"].rolling(30).std()

    df["realized_vol_7"] = np.sqrt((df["return_1d"]**2).rolling(7).sum())
    df["realized_vol_14"] = np.sqrt((df["return_1d"]**2).rolling(14).sum())

    # ============================
    # 4) PRICE RANGE (with only price)
    # ============================
    df["highest_30"] = df["price"].rolling(30).max()
    df["lowest_30"] = df["price"].rolling(30).min()

    df["distance_to_high_30"] = df["highest_30"] - df["price"]
    df["distance_to_low_30"] = df["price"] - df["lowest_30"]
    df["range_30"] = df["highest_30"] - df["lowest_30"]
    df["range_30_ratio"] = df["range_30"] / df["price"]

    # ============================
    # 5) VOLUME FEATURES
    # ============================
    df["volume_change_1d"] = df["volume"].pct_change(1)
    df["vol_MA7"] = df["volume"].rolling(7).mean()
    df["vol_MA30"] = df["volume"].rolling(30).mean()
    df["volume_ratio"] = df["volume"] / df["vol_MA30"]

    # OBV (simplifié)
    df["OBV"] = (np.sign(df["return_1d"]) * df["volume"]).fillna(0).cumsum()

    # ============================
    # 6) RSI / MACD (based on price)
    # ============================
    delta = df["price"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI14"] = 100 - (100 / (1 + rs))

    df["EMA12"] = df["price"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["price"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # ============================
    # 7) REGIME FEATURES
    # ============================
    df["regime"] = (df["price"] > df["MA30"]).astype(int)
    df["trend_strength"] = df["MA_diff"] / df["MA30"]
    df["volatility_regime"] = (df["vol_7d"] > df["vol_30d"]).astype(int)



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