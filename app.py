import os
import requests
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, date
import mlflow
from mlflow.tracking import MlflowClient
from builddata import build_dataset
from builddata import get_btc_history
from builddata import load_latest_data
from train import train
from train import load_latest_model



# =====================================
# 1. CONFIG MLFLOW POUR AZURE ML
# =====================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "azureml://canadacentral.api.azureml.ms/mlflow/v1.0/subscriptions/b115f392-8b15-499a-a548-edd84815dbcb/resourceGroups/rg-bitcoins/providers/Microsoft.MachineLearningServices/workspaces/bitcoins_ws",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()


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
                html.Div(
                    style={"textAlign": "center"},
                    children=[
                        html.H3("Par rapport à votre date"),
                        dcc.DatePickerSingle(
                            id="purchase-date",
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            date=max_date,  # par défaut, la dernière date des données
                            display_format="DD/MM/YYYY"
                        ),
                        html.Div(id="purchase-compare", style={"marginTop": "10px", "fontSize": "18px"})
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
        Output("purchase-compare", "children"),
        Output("price-chart", "figure"),
        Output("prediction-3j", "children"),
    ],
    [
        Input("interval-component", "n_intervals"),
        Input("purchase-date", "date"),
    ]
)
def update_dashboard(n, purchase_date):
    # On recharge l'historique à chaque mise à jour (simple, mais pas optimisé)
    today_str = date.today().strftime("%Y-%m-%d")
    experiment_name = f"Data"
    run = f"{experiment_name}_{today_str}"
    df = load_latest_data(experiment_name, run)
    cible = "3j"
    experiment_name = f"Model{cible}"
    run = f"{experiment_name}{today_str}"
    model_3j  = load_latest_model(cible,  experiment_name, run, df)
    dfderniereligne = df.drop(columns=['date']).iloc[-1:]
    prediction3j = model_3j.predict(dfderniereligne)
    proba_3j_hausse = model_3j.predict_proba(dfderniereligne)[0][1] * 100
    proba_3j_baisse = 100 - proba_3j_hausse
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

    # --- Comparaison avec la date choisie ---
    purchase_msg = "Choisissez une date pour comparer."
    if purchase_date is not None:
        try:
            purchase_date_obj = datetime.fromisoformat(purchase_date).date()
            # On cherche si cette date existe dans le df
            df_by_date = df_90.set_index("date")
            if purchase_date_obj in df_by_date.index:
                purchase_price = df_by_date.loc[purchase_date_obj]["price"]
                var_pct = (current_price / purchase_price - 1) * 100
                direction = "hausse" if var_pct >= 0 else "baisse"
                purchase_msg = (
                    f"Depuis le {purchase_date_obj.strftime('%d/%m/%Y')} : "
                    f"{var_pct:+.1f} % ({direction})\n"
                    f"De {purchase_price:,.2f} € à {current_price:,.2f} €"
                ).replace(",", " ")
            else:
                purchase_msg = "Pas de données pour cette date (trop ancienne ou future)."
        except Exception:
            purchase_msg = "Date invalide."
    # predictions
    if prediction3j == 0:
       prediction_3j = f"📉 Tendance sur 3 jours : baisse probable (probabilitée = {round(proba_3j_baisse, 2)} %)"
    else:
       prediction_3j = f"📈 Tendance sur 3 jours : Hausse probable (probabilitée = {round(proba_3j_hausse, 2)} %)"
        
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

    return current_text, avg_text, diff_text, purchase_msg, fig, prediction_3j


if __name__ == "__main__":
    app.run(debug=True)
