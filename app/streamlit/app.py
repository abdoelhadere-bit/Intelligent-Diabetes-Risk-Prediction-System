"""Interface praticien : saisie des constantes cliniques et affichage du risque."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 10  # secondes

ADVICE = {
    True: [
        "Orienter la patiente vers son médecin traitant pour un bilan approfondi.",
        "Confirmer par des examens biologiques (glycémie à jeun, HbA1c).",
        "Accompagner vers une alimentation équilibrée et une activité physique régulière.",
        "Planifier un suivi rapproché des constantes (glycémie, poids, tension).",
    ],
    False: [
        "Maintenir une alimentation équilibrée et une activité physique régulière.",
        "Poursuivre un dépistage périodique, en particulier en cas d'antécédents familiaux.",
        "Reconsulter en cas de symptômes (soif intense, fatigue, mictions fréquentes).",
    ],
}

st.set_page_config(page_title="Diabetes Risk", page_icon="🩺", layout="centered")
st.title("🩺 Estimation du risque de diabète")
st.caption("Outil d'aide au dépistage destiné aux praticiens. Il ne remplace pas un diagnostic médical.")

with st.form("patient_form"):
    st.subheader("Constantes cliniques de la patiente")
    col1, col2 = st.columns(2)
    with col1:
        glucose = st.number_input("Glycémie à 2 h (mg/dL)", min_value=40.0, max_value=300.0, value=120.0, step=1.0)
        bmi = st.number_input("IMC (kg/m²)", min_value=10.0, max_value=80.0, value=30.0, step=0.1)
        age = st.number_input("Âge (ans)", min_value=21, max_value=100, value=35, step=1)
    with col2:
        insulin = st.number_input("Insuline à 2 h (µU/mL)", min_value=1.0, max_value=1000.0, value=120.0, step=1.0)
        dpf = st.number_input("Antécédents familiaux (DPF)", min_value=0.01, max_value=3.0, value=0.40, step=0.01,
                              help="Diabetes Pedigree Function : score résumant les antécédents familiaux de diabète.")
    submitted = st.form_submit_button("Évaluer le risque", use_container_width=True)

if submitted:
    payload = {"Glucose": glucose, "BMI": bmi, "DiabetesPedigreeFunction": dpf, "Age": int(age), "Insulin": insulin}
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException:
        st.error("Impossible de joindre l'API de prédiction. Vérifiez qu'elle est démarrée.")
        st.stop()

    if response.status_code == 422:
        st.error("Certaines valeurs saisies sont invalides. Vérifiez les constantes.")
        st.stop()
    if response.status_code != 200:
        st.error(f"Le service de prédiction est indisponible (code {response.status_code}).")
        st.stop()

    result = response.json()
    is_high_risk = result["is_high_risk"]
    color, label = ("#d62828", "RISQUE ÉLEVÉ") if is_high_risk else ("#2a9d8f", "RISQUE FAIBLE")

    st.markdown(
        f"<div style='background-color:{color};color:white;padding:18px;border-radius:12px;"
        f"text-align:center;font-size:26px;font-weight:bold;'>{label}</div>",
        unsafe_allow_html=True,
    )
    st.metric("Probabilité de profil à risque élevé", f"{result['probability_high_risk']:.0%}")

    st.subheader("Conseils de suivi")
    for advice in ADVICE[is_high_risk]:
        st.markdown(f"- {advice}")

    st.caption(f"Modèle utilisé : diabetes-risk-pipeline, version {result['model_version']}")