"""
Application Streamlit — Détection de Transactions Bancaires Frauduleuses
Master MSDE 7 — EHTP — Module 5 Machine Learning
Encadrant : Pr. Abdelhamid Fadil
Design : Inspiré du style bancaire professionnel (Bank of America)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION DE LA PAGE
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield — Détection de Fraude",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# CSS BANCAIRE PROFESSIONNEL
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Imports Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Variables couleurs BOA ── */
    :root {
        --boa-red:      #e31837;
        --boa-dark:     #012169;
        --boa-navy:     #001a57;
        --boa-blue:     #0077c8;
        --boa-light:    #f5f7fa;
        --boa-border:   #d9dde1;
        --boa-text:     #1a1a1a;
        --boa-muted:    #6b7280;
        --success:      #0a7a45;
        --danger:       #c0162c;
        --warning-bg:   #fff8e1;
    }

    /* ── Reset global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--boa-text);
    }

    /* ── Fond principal ── */
    .main .block-container {
        background-color: #f5f7fa;
        padding-top: 0 !important;
        max-width: 1400px;
    }

    /* ── Barre de navigation bancaire ── */
    .bank-navbar {
        background: linear-gradient(135deg, #012169 0%, #001a57 100%);
        padding: 0.8rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 3px solid #e31837;
        box-shadow: 0 2px 12px rgba(0,0,0,0.25);
    }

    .bank-logo {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .bank-logo-icon {
        background: #e31837;
        color: white;
        width: 38px;
        height: 38px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        font-weight: 700;
    }

    .bank-logo-text {
        color: white;
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .bank-logo-sub {
        color: rgba(255,255,255,0.65);
        font-size: 0.72rem;
        font-weight: 400;
        letter-spacing: 1px;
        text-transform: uppercase;
        display: block;
        margin-top: -3px;
    }

    .bank-nav-badge {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        color: white;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    /* ── Section titre ── */
    .page-title-section {
        text-align: center;
        padding: 1.5rem 1rem 1rem;
        margin-bottom: 1.5rem;
    }

    .page-title {
        font-size: 1.9rem;
        font-weight: 700;
        color: #012169;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.3px;
    }

    .page-subtitle {
        font-size: 0.95rem;
        color: #6b7280;
        font-weight: 400;
    }

    /* ── Cartes métriques KPI ── */
    .kpi-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.8rem;
    }

    .kpi-card {
        background: white;
        border: 1px solid #e5e9ef;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        flex: 1;
        display: flex;
        align-items: center;
        gap: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        transition: box-shadow 0.2s;
    }

    .kpi-card:hover {
        box-shadow: 0 4px 16px rgba(1,33,105,0.12);
    }

    .kpi-icon {
        background: #eef2fb;
        color: #012169;
        width: 44px;
        height: 44px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        flex-shrink: 0;
    }

    .kpi-label {
        font-size: 0.72rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
        margin-bottom: 2px;
    }

    .kpi-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #012169;
        line-height: 1;
    }

    /* ── Panneau gauche/droite ── */
    .panel {
        background: white;
        border: 1px solid #e5e9ef;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    .panel-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 1.2rem;
        padding-bottom: 0.9rem;
        border-bottom: 2px solid #f0f2f6;
    }

    .panel-header-icon {
        background: #012169;
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 7px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
    }

    .panel-title {
        font-size: 1rem;
        font-weight: 600;
        color: #012169;
        margin: 0;
    }

    /* ── Carte résultat FRAUDE ── */
    .result-fraud {
        background: linear-gradient(135deg, #fff0f0 0%, #ffe8e8 100%);
        border: 2px solid #e31837;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
    }

    .result-fraud-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #e31837;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.8rem;
    }

    .result-fraud-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #c0162c;
        margin: 0 0 0.5rem 0;
    }

    .result-fraud-desc {
        font-size: 0.9rem;
        color: #7a1020;
        margin: 0;
        line-height: 1.5;
    }

    /* ── Carte résultat LÉGITIME ── */
    .result-legit {
        background: linear-gradient(135deg, #f0fdf6 0%, #e8f8ef 100%);
        border: 2px solid #0a7a45;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
    }

    .result-legit-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #0a7a45;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.8rem;
    }

    .result-legit-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0a7a45;
        margin: 0 0 0.5rem 0;
    }

    .result-legit-desc {
        font-size: 0.9rem;
        color: #0a4a2a;
        margin: 0;
        line-height: 1.5;
    }

    /* ── Jauge de probabilité ── */
    .proba-section {
        background: #f8fafc;
        border: 1px solid #e5e9ef;
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 1rem;
    }

    .proba-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 1rem;
    }

    .proba-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.8rem;
    }

    .proba-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #374151;
        width: 80px;
        flex-shrink: 0;
    }

    .proba-bar-container {
        flex: 1;
        background: #e5e9ef;
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
    }

    .proba-bar-legit {
        height: 100%;
        background: linear-gradient(90deg, #0a7a45, #16a05e);
        border-radius: 6px;
        transition: width 0.6s ease;
    }

    .proba-bar-fraud {
        height: 100%;
        background: linear-gradient(90deg, #e31837, #ff4d6d);
        border-radius: 6px;
        transition: width 0.6s ease;
    }

    .proba-pct {
        font-size: 0.9rem;
        font-weight: 700;
        width: 52px;
        text-align: right;
        flex-shrink: 0;
    }

    /* ── Tableau données saisies ── */
    .data-table {
        background: #f8fafc;
        border: 1px solid #e5e9ef;
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 1rem;
    }

    .data-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.65rem 1rem;
        border-bottom: 1px solid #eef0f4;
    }

    .data-row:last-child {
        border-bottom: none;
    }

    .data-row-label {
        font-size: 0.82rem;
        color: #6b7280;
        font-weight: 500;
    }

    .data-row-value {
        font-size: 0.9rem;
        font-weight: 600;
        color: #012169;
    }

    /* ── Alerte d'attente ── */
    .waiting-card {
        background: #f0f4ff;
        border: 1px dashed #012169;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        color: #012169;
    }

    .waiting-icon {
        font-size: 2.5rem;
        margin-bottom: 0.8rem;
    }

    .waiting-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .waiting-desc {
        font-size: 0.85rem;
        color: #6b7280;
    }

    /* ── Sidebar bancaire ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #012169 0%, #001a57 100%) !important;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        color: rgba(255,255,255,0.85) !important;
        font-size: 0.85rem !important;
    }

    [data-testid="stSidebar"] input[type="number"] {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        color: white !important;
        border-radius: 6px !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        background: #e31837 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        padding: 0.7rem !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
        transition: all 0.2s !important;
        box-shadow: 0 4px 12px rgba(227,24,55,0.4) !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: #c0162c !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(227,24,55,0.5) !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }

    [data-testid="stSidebar"] .stCaption {
        color: rgba(255,255,255,0.6) !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15) !important;
    }

    /* ── Tags radio sidebar ── */
    [data-testid="stSidebar"] .stRadio > div {
        background: rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 0.5rem;
    }

    /* ── Footer bancaire ── */
    .bank-footer {
        background: #012169;
        color: rgba(255,255,255,0.7);
        text-align: center;
        padding: 1.2rem;
        margin: 2rem -1rem -1rem -1rem;
        font-size: 0.78rem;
        border-top: 2px solid #e31837;
        letter-spacing: 0.3px;
    }

    /* ── Confidence badge ── */
    .confidence-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: #012169;
        color: white;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.8rem;
    }

    /* ── Section détails techniques ── */
    .tech-details {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.82rem;
    }

    .tech-row {
        display: flex;
        justify-content: space-between;
        padding: 0.35rem 0;
        border-bottom: 1px solid #eef0f4;
        color: #4b5563;
    }

    .tech-row:last-child { border-bottom: none; }
    .tech-key { font-weight: 500; color: #6b7280; }
    .tech-val { font-weight: 600; color: #012169; font-family: monospace; }

    /* ── Masquer éléments Streamlit par défaut ── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# CHARGEMENT DU PIPELINE
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return joblib.load('fraud_detection_pipeline.pkl')

try:
    pipeline = load_pipeline()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Erreur de chargement du modèle : {e}")
    model_loaded = False

# ─────────────────────────────────────────────────────────────────
# NAVBAR BANCAIRE
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="bank-navbar">
    <div class="bank-logo">
        <div class="bank-logo-icon">🛡️</div>
        <div>
            <span class="bank-logo-text">FraudShield</span>
            <span class="bank-logo-sub">Fraud Detection System</span>
        </div>
    </div>
    <div>
        <span class="bank-nav-badge">🔒 Système Sécurisé</span>
        &nbsp;
        <span class="bank-nav-badge">⚡ Temps Réel</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# TITRE
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-title-section">
    <h1 class="page-title">Analyse de Transaction par Carte Bancaire</h1>
    <p class="page-subtitle">Système de détection de fraude basé sur l'intelligence artificielle — Modèle Extra Trees (PR-AUC : 0.88)</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="kpi-row">
    <div class="kpi-card">
        <div class="kpi-icon">🎯</div>
        <div>
            <div class="kpi-label">PR-AUC</div>
            <div class="kpi-value">0.8806</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">🔍</div>
        <div>
            <div class="kpi-label">Recall</div>
            <div class="kpi-value">84.69%</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">✅</div>
        <div>
            <div class="kpi-label">Précision</div>
            <div class="kpi-value">90.22%</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">⚡</div>
        <div>
            <div class="kpi-label">F1-Score</div>
            <div class="kpi-value">0.8737</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">📊</div>
        <div>
            <div class="kpi-label">G-Mean</div>
            <div class="kpi-value">0.9202</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📋 Nouvelle Transaction")
st.sidebar.markdown("---")

st.sidebar.markdown("**Mode de saisie**")
mode = st.sidebar.radio(
    "",
    ["✏️ Saisie manuelle", "✅ Exemple Légitime", "⚠️ Exemple Fraude"],
    index=0
)

# Valeurs par défaut
if "Légitime" in mode:
    default_time, default_amount = 50000.0, 100.0
    default_v = [-0.5, 0.3, 0.8, 0.1, -0.2, 0.4, -0.1, 0.05, -0.3,
                 0.2, 0.5, -0.1, 0.3, 0.1, -0.2, 0.4, 0.6, -0.2,
                 0.3, -0.1, -0.05, 0.1, 0.0, 0.2, -0.1, 0.0, 0.05, -0.05]
elif "Fraude" in mode:
    default_time, default_amount = 100000.0, 250.0
    default_v = [-2.3, 2.8, -3.5, 2.1, -1.8, -0.9, -3.2, 1.5, -2.1,
                 -5.5, 3.2, -6.1, 0.2, -7.3, 0.1, -3.0, -7.8, -2.5,
                 1.2, 0.3, 0.5, 0.1, -0.2, 0.0, 0.1, -0.05, 0.2, 0.0]
else:
    default_time, default_amount = 0.0, 0.0
    default_v = [0.0] * 28

st.sidebar.markdown("---")
st.sidebar.markdown("**Variables principales**")

time_val = st.sidebar.number_input(
    "⏱️ Time (secondes)",
    min_value=0.0, max_value=172800.0,
    value=float(default_time), step=100.0,
    help="Temps écoulé depuis la première transaction."
)
amount_val = st.sidebar.number_input(
    "💰 Amount (€)",
    min_value=0.0, max_value=30000.0,
    value=float(default_amount), step=10.0,
    help="Montant de la transaction en euros."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Composantes PCA (V1–V28)**")
st.sidebar.caption("Valeurs typiques autour de 0.")

v_values = {}
with st.sidebar.expander("📂 Modifier V1 à V28", expanded=False):
    for i in range(1, 29):
        v_values[f'V{i}'] = st.number_input(
            f"V{i}",
            min_value=-50.0, max_value=50.0,
            value=float(default_v[i-1]),
            step=0.1, key=f"v{i}", format="%.2f"
        )

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔍 ANALYSER LA TRANSACTION", type="primary")

# ─────────────────────────────────────────────────────────────────
# CORPS PRINCIPAL — Données + Résultat
# ─────────────────────────────────────────────────────────────────
columns_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
input_dict = {'Time': time_val, **v_values, 'Amount': amount_val}
input_df = pd.DataFrame([input_dict])[columns_order]

col_left, col_right = st.columns([1, 1], gap="large")

# ── COLONNE GAUCHE : Données saisies
with col_left:
    st.markdown("""
    <div class="panel">
        <div class="panel-header">
            <div class="panel-header-icon">📋</div>
            <p class="panel-title">Récapitulatif de la transaction</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Affichage propre des variables principales
    st.markdown(f"""
    <div class="data-table">
        <div class="data-row">
            <span class="data-row-label">⏱️ Time</span>
            <span class="data-row-value">{time_val:,.2f} s</span>
        </div>
        <div class="data-row">
            <span class="data-row-label">💰 Amount</span>
            <span class="data-row-value">{amount_val:,.2f} €</span>
        </div>
        <div class="data-row">
            <span class="data-row-label">🔢 Variables PCA</span>
            <span class="data-row-value">V1 à V28 (30 features)</span>
        </div>
        <div class="data-row">
            <span class="data-row-label">📐 Dimensions</span>
            <span class="data-row-value">1 × 30</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📊 Voir les composantes PCA (V1–V28)", expanded=False):
        v_df = pd.DataFrame({
            'Variable': list(v_values.keys()),
            'Valeur': [round(v, 4) for v in v_values.values()]
        })
        st.dataframe(v_df, hide_index=True, height=320, use_container_width=True)

    # Info modèle
    st.markdown("""
    <div style='background:#eef2fb; border-radius:8px; padding:1rem; margin-top:1rem; border-left:4px solid #012169;'>
        <p style='margin:0; font-size:0.82rem; color:#012169; font-weight:600;'>🤖 Modèle : Extra Trees Classifier</p>
        <p style='margin:0.3rem 0 0; font-size:0.78rem; color:#4b5563;'>300 arbres • max_depth=None • RandomizedSearchCV (30 iter, 3 folds)</p>
    </div>
    """, unsafe_allow_html=True)

# ── COLONNE DROITE : Résultat
with col_right:
    st.markdown("""
    <div class="panel">
        <div class="panel-header">
            <div class="panel-header-icon">🎯</div>
            <p class="panel-title">Résultat de l'analyse</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.warning("⚠️ Le modèle n'est pas chargé. Vérifiez que `fraud_detection_pipeline.pkl` est présent.")

    elif predict_btn:
        with st.spinner("🔍 Analyse en cours..."):
            prediction  = pipeline.predict(input_df)[0]
            proba_fraud = pipeline.predict_proba(input_df)[0, 1]
            proba_legit = 1 - proba_fraud
            confidence  = max(proba_fraud, proba_legit)

        pct_legit = int(proba_legit * 100)
        pct_fraud = int(proba_fraud * 100)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-fraud">
                <div class="result-fraud-badge">🚨 Alerte Fraude</div>
                <h3 class="result-fraud-title">Transaction Frauduleuse Détectée</h3>
                <p class="result-fraud-desc">
                    Cette transaction présente des caractéristiques anormales détectées par notre système IA.
                    <br><br>
                    <strong>Action recommandée :</strong> Blocage immédiat et vérification manuelle par l'équipe de sécurité.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-legit">
                <div class="result-legit-badge">✅ Transaction Validée</div>
                <h3 class="result-legit-title">Transaction Légitime</h3>
                <p class="result-legit-desc">
                    Cette transaction présente les caractéristiques d'une opération normale.
                    <br><br>
                    <strong>Aucune action requise</strong> — la transaction peut être autorisée.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Barres de probabilité
        st.markdown(f"""
        <div class="proba-section">
            <div class="proba-title">Niveau de confiance du modèle</div>

            <div class="proba-row">
                <span class="proba-label" style="color:#0a7a45;">Légitime</span>
                <div class="proba-bar-container">
                    <div class="proba-bar-legit" style="width:{pct_legit}%;"></div>
                </div>
                <span class="proba-pct" style="color:#0a7a45;">{proba_legit*100:.1f}%</span>
            </div>

            <div class="proba-row">
                <span class="proba-label" style="color:#e31837;">Fraude</span>
                <div class="proba-bar-container">
                    <div class="proba-bar-fraud" style="width:{pct_fraud}%;"></div>
                </div>
                <span class="proba-pct" style="color:#e31837;">{proba_fraud*100:.1f}%</span>
            </div>

            <div style="margin-top:0.8rem; text-align:right;">
                <span class="confidence-badge">🎯 Confiance : {confidence*100:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Détails techniques
        with st.expander("🔬 Détails techniques", expanded=False):
            st.markdown(f"""
            <div class="tech-details">
                <div class="tech-row">
                    <span class="tech-key">Classe prédite</span>
                    <span class="tech-val">{int(prediction)} — {'Fraude' if prediction == 1 else 'Légitime'}</span>
                </div>
                <div class="tech-row">
                    <span class="tech-key">P(Légitime)</span>
                    <span class="tech-val">{proba_legit:.6f}</span>
                </div>
                <div class="tech-row">
                    <span class="tech-key">P(Fraude)</span>
                    <span class="tech-val">{proba_fraud:.6f}</span>
                </div>
                <div class="tech-row">
                    <span class="tech-key">Seuil de décision</span>
                    <span class="tech-val">0.50</span>
                </div>
                <div class="tech-row">
                    <span class="tech-key">Algorithme</span>
                    <span class="tech-val">ExtraTreesClassifier</span>
                </div>
                <div class="tech-row">
                    <span class="tech-key">n_estimators</span>
                    <span class="tech-val">300</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="waiting-card">
            <div class="waiting-icon">🔐</div>
            <div class="waiting-title">En attente d'analyse</div>
            <div class="waiting-desc">
                Renseignez les données de la transaction dans le panneau latéral,<br>
                puis cliquez sur <strong>ANALYSER LA TRANSACTION</strong>.
                <br><br>
                💡 Utilisez les exemples prédéfinis pour un test rapide.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SECTION À PROPOS
# ─────────────────────────────────────────────────────────────────
st.markdown("---")

with st.expander("ℹ️ À propos de FraudShield", expanded=False):
    col_a1, col_a2, col_a3 = st.columns(3)

    with col_a1:
        st.markdown("""
        **🎓 Contexte académique**

        Projet réalisé dans le cadre du **Master Data Engineering (MSDE 7)**
        à l'**École Hassania des Travaux Publics (EHTP)**.

        Module 5 — Machine Learning
        Encadrant : **Pr. Abdelhamid Fadil**
        """)

    with col_a2:
        st.markdown("""
        **📊 Dataset**

        - Source : *Credit Card Fraud Detection* (Kaggle)
        - Période : Septembre 2013 (porteurs européens)
        - Taille : **284 807** transactions
        - Fraudes : seulement **0.172%**
        - Features : 30 (Time, Amount, V1–V28 via PCA)
        """)

    with col_a3:
        st.markdown("""
        **🛠️ Méthodologie**

        1. RobustScaler (Amount & Time)
        2. SMOTE (rééquilibrage classes)
        3. Test de 11 algorithmes ML
        4. Tuning RandomizedSearchCV
        5. **Extra Trees** — modèle final

        | Métrique | Score |
        |---|---|
        | PR-AUC | **0.8806** |
        | Recall | **84.69%** |
        | Precision | **90.22%** |
        """)

# ─────────────────────────────────────────────────────────────────
# FOOTER BANCAIRE
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="bank-footer">
    🛡️ FraudShield — Système de Détection de Fraude par IA &nbsp;|&nbsp;
    MSDE 7 · EHTP · 2025/2026 &nbsp;|&nbsp;
    Encadrant : Pr. Abdelhamid Fadil &nbsp;|&nbsp;
    🔒 Données traitées localement — Aucune transmission externe
</div>
""", unsafe_allow_html=True)
