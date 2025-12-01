import os
import requests
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, date
from builddata import build_dataset
from builddata import get_btc_history
from train import train
import joblib
from azure.storage.blob import BlobClient
import io
import requests








# =========================
# 1. Fonctions utilitaires
# =========================


def load_latest_data(blob_path):
    """
    Charge un fichier CSV spécifique dans Azure Blob Storage.
    Si le fichier est introuvable, appelle build_dataset().
    """
    try:
        blob = BlobClient.from_connection_string(
            conn_str=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
            container_name="data",
            blob_name=blob_path
        )

        stream = io.BytesIO(blob.download_blob().readall())
        df = pd.read_csv(stream)

        print(f"✔ Fichier chargé depuis Azure : {blob_path}")
        return df

    except Exception as e:
        print(f"❌ Erreur lors du chargement ({blob_path}) : {e}")
        print("➡ Exécution de build_dataset()")
        df = build_dataset()
        return df



def load_latest_model(blob_path, df):
    """
    Charge un modèle .pkl spécifique depuis Azure Blob Storage (conteneur 'models').
    Si le fichier est introuvable, appelle build_model().
    """
    try:
        blob = BlobClient.from_connection_string(
            conn_str=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
            container_name="models",
            blob_name=blob_path
        )

        # Télécharger en mémoire
        stream = io.BytesIO(blob.download_blob().readall())

        # Charger le .pkl
        model = pickle.load(stream)

        print(f"✔ Modèle chargé depuis Azure : {blob_path}")
        return model

    except Exception as e:
        ndf = df.copy()
        ndf["target_up"] = (ndf["price"].shift(-7) > ndf["price"]).astype(int)
        ndf = ndf.iloc[:-7]
        target = ndf["target_up"]
        dataexplicative = ndf.drop(columns=["date", "target_up", "return_7d"])
        model = train(dataexplicative, target)
        print("✅ Nouveau modèle entraîné.")
        return model



# test
print("DEBUG STORAGE =", os.getenv("AZURE_STORAGE_CONNECTION_STRING"))


# =========================
# 2. App Dash
# =========================

app = dash.Dash(__name__)
server = app.server  # utile si déploiement



# On récupère une première fois les données (90 jours)
hist_df = get_btc_history(90)

min_date = hist_df["date"].min()
max_date = hist_df["date"].max()




# =========================
# 3. Layout
# =========================

app.layout = html.Div(
    style={"fontFamily": "Arial", "margin": "20px"},
    children=[
        html.H1("Dashboard Bitcoin (BTC/EUR)", style={"textAlign": "center"}),

        # --- Bloc haut : indicateurs clefs ---
        html.Div(
            style={"display": "flex", "justifyContent": "space-around", "marginBottom": "30px"},
            children=[
                html.Div(
                    style={"textAlign": "center"},
                    children=[
                        html.H3("Prix actuel"),
                        html.Div(id="current-price", style={"fontSize": "28px", "fontWeight": "bold"})
                    ]
                ),
                html.Div(
                    style={"textAlign": "center"},
                    children=[
                        html.H3("Moyenne 30 jours"),
                        html.Div(id="avg-30d", style={"fontSize": "22px"}),
                        html.Div(id="avg-30d-diff", style={"fontSize": "18px"})
                    ]
                ),
               
            ]
        ),
            
        
        html.Div(
    style={
        "textAlign": "center",
        "marginBottom": "30px",
        "padding": "15px",
        "backgroundColor": "#f2f2f2",
        "borderRadius": "10px"
    },
    children=[
        html.H3("Prédiction du modèle", style={"marginBottom": "10px"}),
        html.Div(id="prediction-3j", style={"fontSize": "20px", "fontWeight": "bold"}),
        html.Div(id="prediction-7j", style={"fontSize": "20px", "fontWeight": "bold"})
        
    ]
),
        # --- Graphique de prix ---
        dcc.Graph(id="price-chart"),

        # Interval pour rafraîchir (facultatif, ici toutes les 60s)
        dcc.Interval(
            id="interval-component",
            interval=60 * 1000,  # en millisecondes
            n_intervals=0
        )
    ]
)


# =========================
# 4. Callbacks
# =========================

@app.callback(
    [
        Output("current-price", "children"),
        Output("avg-30d", "children"),
        Output("avg-30d-diff", "children"),
        Output("price-chart", "figure"),
        Output("prediction", "children"),
    ],
    [
        Input("interval-component", "n_intervals"),
    ]
)
def update_dashboard(n):
    # On recharge l'historique à chaque mise à jour (simple, mais pas optimisé)
    today_str = date.today().strftime("%Y-%m-%d")
    path = F'data_{today_str}.csv'
    df = load_latest_data(path)
    pathmodel = F'model_{today_str}.pkl'
    model  = load_latest_model(pathmodel, df)
    dfderniereligne = df.drop(columns=['date', "return_7d"]).iloc[-1:]
    prediction = model.predict(dfderniereligne)
    proba_hausse = model.predict_proba(dfderniereligne)[0][1] * 100
    proba_baisse = 100 - proba_hausse
    df_90 = df.tail(90)
    # --- Prix actuel ---
    last_row = df_90.iloc[-1]
    current_price = last_row["price"]
    current_date = last_row["date"]

    current_text = f"{current_price:,.2f} €".replace(",", " ")

    # --- Moyenne 30 jours ---
    last_30 = df_90.tail(30)
    avg_30 = last_30["price"].mean()
    avg_text = f"{avg_30:,.2f} €".replace(",", " ")

    # Différence entre prix actuel et moyenne
    diff_pct = (current_price / avg_30 - 1) * 100
    if diff_pct >= 0:
        diff_text = f"Actuellement ~ +{diff_pct:.1f} % au-dessus de la moyenne 30 j"
    else:
        diff_text = f"Actuellement ~ {diff_pct:.1f} % en-dessous de la moyenne 30 j"

   
    # predictions
    if prediction == 0:
       prediction = f"📉 Tendance sur 7 jours : baisse probable (probabilitée = {round(proba_baisse, 2)} %)"
    else:
       prediction = f"📈 Tendance sur 7 jours : Hausse probable (probabilitée = {round(proba_hausse, 2)} %)"
        
    # --- Graphique ---
    fig = {
        "data": [
            go.Scatter(
                x=df["date"],
                y=df["price"],
                mode="lines",
                name="BTC/EUR"
            )
        ],
        "layout": go.Layout(
            title=f"Évolution du prix du Bitcoin<br> (derniers 90 jours)",
            xaxis={"title": "Date"},
            yaxis={"title": "Prix (EUR)"},
            margin={"l": 60, "r": 20, "t": 100, "b": 40}
        )
    }

    return current_text, avg_text, diff_text, fig, prediction


if __name__ == "__main__":
    app.run(debug=True)