"""
Streamlit — Prix de vente d'une voiture (Régression, Car_data)
Reprend l'application de régression du prof (app_voiture.py) : on recharge les encodeurs et le
pipeline du meilleur modèle (RobustScaler + régresseur) sauvegardés dans le notebook.

En local :  streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st

st.set_page_config(page_title="Prix d'une voiture", page_icon="💰", layout="wide")


# ---------- Objets issus du notebook (chargés une seule fois) ----------
@st.cache_resource
def charger_objets():
    encoders = jb.load("encoders.joblib")              # Fuel_Type, Seller_Type, Transmission
    pipe_from_grid = jb.load("pipe_from_grid.joblib")  # pipeline du meilleur modèle
    return encoders, pipe_from_grid


encoders, pipe_from_grid = charger_objets()


# ---------- Prédiction pour une voiture ----------
def Pred_func(Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age):
    code_fuel = encoders[0].transform([Fuel_Type])[0]
    code_seller = encoders[1].transform([Seller_Type])[0]
    code_trans = encoders[2].transform([Transmission])[0]
    # même ordre de colonnes que dans le notebook
    vecteur = np.array([Kms_Driven, Present_Price, code_fuel, code_seller, code_trans, Age]).reshape(1, -1)
    prix = pipe_from_grid.predict(vecteur)[0]   # le pipeline normalise lui-même les données
    return round(float(prix), 2)


# ---------- Prédiction pour un fichier ----------
def Pred_func_csv(fichier):
    tableau = pd.read_csv(fichier)
    resultats = []
    for ligne in tableau.values:
        resultats.append(Pred_func(ligne[0], ligne[1], ligne[2], ligne[3], ligne[4], ligne[5]))
    tableau["Selling_Price prédit (k$)"] = resultats
    return tableau


# ---------- Barre latérale ----------
with st.sidebar:
    st.header("À propos")
    st.write("Six algorithmes de régression ont été comparés (linéaire, Ridge, Lasso, arbre, "
             "Random Forest, XGBoost) ; le meilleur sur la validation est utilisé ici.")
    st.caption("Projet Machine Learning — régression")

st.title("💰 À quel prix cette voiture peut-elle se vendre ?")
choix = st.radio("Mode", ["Une voiture", "Un fichier CSV"], horizontal=True)

if choix == "Une voiture":
    gauche, droite = st.columns(2)
    with gauche:
        Kms_Driven = st.number_input("Kilométrage", min_value=0, value=60_000, step=5_000)
        Present_Price = st.number_input("Prix du modèle neuf (k$)", min_value=0.0, value=15.0, step=0.5)
        Age = st.slider("Âge de la voiture (années)", 0, 25, 7)
    with droite:
        Fuel_Type = st.radio("Carburant", list(encoders[0].classes_), horizontal=True)
        Seller_Type = st.radio("Vendeur", list(encoders[1].classes_), horizontal=True)
        Transmission = st.radio("Boîte de vitesses", list(encoders[2].classes_), horizontal=True)

    if st.button("Lancer la prédiction", type="primary"):
        try:
            prix = Pred_func(Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age)
            st.metric("Prix de vente estimé", f"{prix} k$")
        except Exception as erreur:
            st.error(f"Prédiction impossible : {erreur}")
else:
    st.info("Colonnes attendues, dans cet ordre : Kms_Driven, Present_Price, Fuel_Type, Seller_Type, "
            "Transmission, Age.")
    fichier = st.file_uploader("Choisir un fichier CSV", type="csv")
    if fichier is not None:
        try:
            tableau = Pred_func_csv(fichier)
            st.dataframe(tableau, use_container_width=True)
            st.download_button("Récupérer les résultats (CSV)", tableau.to_csv(index=False).encode("utf-8"),
                               "resultats_prix.csv", "text/csv")
        except Exception as erreur:
            st.error(f"Fichier non traité : {erreur}")
