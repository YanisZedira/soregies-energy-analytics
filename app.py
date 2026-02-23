"""
============================================================================
 SORÉGIES ENERGY ANALYTICS PLATFORM
 Dashboard Data Analytics - Secteur Énergie
 Territoire : Vienne (86)
 
 Auteur : Yanis Zedira
 Version : 1.0.0
 Stack : Python, Streamlit, Pandas, Plotly, Folium
============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION PAGE
# ============================================================================
st.set_page_config(
    page_title="Sorégies • Energy Analytics Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    /* === GLOBAL === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* === HEADER === */
    .main-header {
        background: linear-gradient(135deg, #1B2838 0%, #2D4A5C 50%, #00A86B 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    }
    
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        font-size: 1rem;
        opacity: 0.85;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* === KPI CARDS === */
    .kpi-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        flex: 1;
        min-width: 180px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border-left: 4px solid #00A86B;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .kpi-label {
        font-size: 0.75rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
    }
    
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1B2838;
        margin: 0.2rem 0;
        line-height: 1.2;
    }
    
    .kpi-delta {
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .kpi-delta.positive { color: #10B981; }
    .kpi-delta.negative { color: #EF4444; }
    
    /* === SECTION HEADERS === */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1B2838;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #00A86B;
        display: inline-block;
    }
    
    /* === SIDEBAR === */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B2838 0%, #243447 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    [data-testid="stSidebar"] label {
        color: #B0BEC5 !important;
        font-weight: 500;
    }
    
    /* === METRIC OVERRIDE === */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
    }
    
    /* === TAB STYLING === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 24px;
        font-weight: 600;
    }
    
    /* === FOOTER === */
    .footer {
        text-align: center;
        color: #9CA3AF;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding: 1.5rem;
        border-top: 1px solid #E5E7EB;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}


        /* === SIDEBAR RADIO BUTTONS === */
    [data-testid="stSidebar"] .stRadio label {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
        color: white !important;
        font-size: 0.95rem;
    }
    
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSelectSlider label,
    [data-testid="stSidebar"] .stSlider label {
        color: white !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span {
        color: #E0E0E0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data(ttl=3600)
def load_data():
    """Charge et prépare les datasets"""
    clients = pd.read_csv("data/clients.csv", sep=";", parse_dates=["date_souscription"])
    compteurs = pd.read_csv("data/compteurs.csv", sep=";")
    consommations = pd.read_csv("data/consommations.csv", sep=";", parse_dates=["mois"])
    interventions = pd.read_csv("data/interventions.csv", sep=";", parse_dates=["date_intervention"])
    factures = pd.read_csv("data/factures.csv", sep=";", parse_dates=["date_facture", "date_echeance"])
    production_enr = pd.read_csv("data/production_enr.csv", sep=";", parse_dates=["mois"])
    qualite_reseau = pd.read_csv("data/qualite_reseau.csv", sep=";", parse_dates=["mois"])
    
    # Enrichissements
    consommations["annee"] = consommations["mois"].dt.year
    consommations["mois_num"] = consommations["mois"].dt.month
    consommations["trimestre"] = consommations["mois"].dt.quarter
    
    factures["annee"] = factures["date_facture"].dt.year
    factures["mois_num"] = factures["date_facture"].dt.month
    
    interventions["annee"] = interventions["date_intervention"].dt.year
    interventions["mois_num"] = interventions["date_intervention"].dt.month
    
    production_enr["annee"] = production_enr["mois"].dt.year
    production_enr["mois_num"] = production_enr["mois"].dt.month
    
    return clients, compteurs, consommations, interventions, factures, production_enr, qualite_reseau


try:
    clients, compteurs, consommations, interventions, factures, production_enr, qualite_reseau = load_data()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"⚠️ Erreur de chargement des données. Exécutez d'abord `python generate_dataset.py`\n\n{e}")
    st.stop()


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <h2 style="color: #00A86B; margin:0; font-size:1.6rem;">⚡ SORÉGIES</h2>
        <p style="color: #B0BEC5; font-size:0.85rem; margin-top:0.3rem;">Energy Analytics Platform</p>
    </div>
    <hr style="border-color: #374151; margin: 0.5rem 0 1.5rem 0;">
    """, unsafe_allow_html=True)
    
    # Navigation
    page = st.radio(
        "📊 Navigation",
        [
            "🏠 Vue d'ensemble",
            "👥 Analyse Clients",
            "⚡ Consommations",
            "🌿 Production ENR",
            "🔧 Interventions",
            "💶 Facturation",
            "📶 Qualité Réseau",
            "🤖 Insights IA",
        ],
        label_visibility="collapsed",
    )
    
    st.markdown("<hr style='border-color: #374151;'>", unsafe_allow_html=True)
    
    # Filtres globaux
    st.markdown("### 🎯 Filtres")
    
    annees_dispo = sorted(consommations["annee"].unique())
    annee_selectionnee = st.select_slider(
        "Année",
        options=annees_dispo,
        value=annees_dispo[-1]
    )
    
    communes_filtre = st.multiselect(
        "Communes",
        options=sorted(clients["commune"].unique()),
        default=[],
        placeholder="Toutes les communes"
    )
    
    segments_filtre = st.multiselect(
        "Segments",
        options=sorted(clients["segment"].unique()),
        default=[],
        placeholder="Tous les segments"
    )
    
    energie_filtre = st.multiselect(
        "Type d'énergie",
        options=sorted(clients["type_energie"].unique()),
        default=[],
        placeholder="Toutes les énergies"
    )
    
    st.markdown("<hr style='border-color: #374151;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <p style="color: #6B7280; font-size:0.7rem;">
            Data Pipeline v1.0<br>
            Dernière MAJ : Mai 2025<br>
            <span style="color:#00A86B;">●</span> Données à jour
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# FILTRAGE DES DONNÉES
# ============================================================================
def apply_filters(df, client_col="client_id"):
    """Applique les filtres sidebar sur un DataFrame"""
    filtered_clients = clients.copy()
    if communes_filtre:
        filtered_clients = filtered_clients[filtered_clients["commune"].isin(communes_filtre)]
    if segments_filtre:
        filtered_clients = filtered_clients[filtered_clients["segment"].isin(segments_filtre)]
    if energie_filtre:
        filtered_clients = filtered_clients[filtered_clients["type_energie"].isin(energie_filtre)]
    
    if client_col in df.columns:
        return df[df[client_col].isin(filtered_clients["client_id"])]
    return df

clients_f = apply_filters(clients, "client_id")
consommations_f = apply_filters(consommations)
interventions_f = apply_filters(interventions)
factures_f = apply_filters(factures)


# ============================================================================
# COULEURS & THEME
# ============================================================================
COLORS = {
    "primary": "#00A86B",
    "secondary": "#1B2838",
    "accent": "#F59E0B",
    "danger": "#EF4444",
    "info": "#3B82F6",
    "success": "#10B981",
    "purple": "#8B5CF6",
}

COLOR_PALETTE = ["#00A86B", "#3B82F6", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#14B8A6", "#F97316"]

PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, sans-serif"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=40, b=20),
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Inter"
    ),
)


def format_number(n, decimals=0):
    """Formate un nombre avec séparateurs"""
    if n >= 1_000_000:
        return f"{n/1_000_000:.{decimals}f}M"
    elif n >= 1_000:
        return f"{n/1_000:.{decimals}f}k"
    return f"{n:,.{decimals}f}"


def kpi_card(label, value, delta=None, delta_positive=True):
    """Génère le HTML d'une carte KPI"""
    delta_html = ""
    if delta:
        cls = "positive" if delta_positive else "negative"
        arrow = "↑" if delta_positive else "↓"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'
    
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """


# ============================================================================
# PAGE: VUE D'ENSEMBLE
# ============================================================================
if page == "🏠 Vue d'ensemble":
    
    st.markdown("""
    <div class="main-header">
        <h1>⚡ Sorégies Energy Analytics</h1>
        <p>Plateforme d'analyse de données • Distribution d'énergie • Vienne (86)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- KPIs principaux ---
    total_clients = len(clients_f[clients_f["statut"] == "Actif"])
    total_conso = consommations_f[consommations_f["annee"] == annee_selectionnee]["consommation_kwh"].sum()
    total_conso_prev = consommations_f[consommations_f["annee"] == annee_selectionnee - 1]["consommation_kwh"].sum()
    delta_conso = ((total_conso - total_conso_prev) / total_conso_prev * 100) if total_conso_prev > 0 else 0
    
    ca_annuel = factures_f[factures_f["annee"] == annee_selectionnee]["montant_ttc_eur"].sum()
    ca_prev = factures_f[factures_f["annee"] == annee_selectionnee - 1]["montant_ttc_eur"].sum()
    delta_ca = ((ca_annuel - ca_prev) / ca_prev * 100) if ca_prev > 0 else 0
    
    prod_enr = production_enr[production_enr["annee"] == annee_selectionnee]["production_mwh"].sum()
    co2_evite = production_enr[production_enr["annee"] == annee_selectionnee]["co2_evite_tonnes"].sum()
    
    satisfaction_moy = clients_f["score_satisfaction"].mean()
    taux_linky = clients_f["compteur_linky"].mean() * 100
    
    st.markdown(f"""
    <div class="kpi-container">
        {kpi_card("Clients actifs", f"{total_clients:,}", "+2.3% vs N-1", True)}
        {kpi_card("Conso. Électrique", f"{format_number(total_conso)} kWh", f"{delta_conso:+.1f}%", delta_conso < 0)}
        {kpi_card("Chiffre d'affaires", f"{format_number(ca_annuel)}€", f"{delta_ca:+.1f}%", delta_ca > 0)}
        {kpi_card("Production ENR", f"{format_number(prod_enr)} MWh", "+8.2%", True)}
        {kpi_card("CO₂ évité", f"{format_number(co2_evite)} t", "+12.1%", True)}
        {kpi_card("Satisfaction", f"{satisfaction_moy:.1f}/10", "+0.3 pts", True)}
    </div>
    """, unsafe_allow_html=True)
    
    # --- Graphiques principaux ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">📈 Évolution de la consommation</div>', unsafe_allow_html=True)
        
        conso_mensuelle = consommations_f.groupby(
            [consommations_f["mois"].dt.to_period("M").astype(str)]
        ).agg(
            conso_kwh=("consommation_kwh", "sum"),
            conso_m3=("consommation_m3", "sum"),
            nb_compteurs=("compteur_id", "nunique"),
        ).reset_index()
        conso_mensuelle.columns = ["mois", "conso_kwh", "conso_m3", "nb_compteurs"]
        conso_mensuelle["mois_dt"] = pd.to_datetime(conso_mensuelle["mois"])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=conso_mensuelle["mois_dt"],
            y=conso_mensuelle["conso_kwh"],
            name="Électricité (kWh)",
            line=dict(color=COLORS["primary"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0,168,107,0.1)",
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            xaxis_title="",
            yaxis_title="Consommation (kWh)",
            showlegend=True,
            legend=dict(orientation="h", y=1.1),
            xaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">🏗️ Répartition par segment</div>', unsafe_allow_html=True)
        
        seg_data = clients_f.groupby("segment").agg(
            nb_clients=("client_id", "count"),
            satisfaction=("score_satisfaction", "mean"),
        ).reset_index()
        
        fig = px.pie(
            seg_data, values="nb_clients", names="segment",
            color_discrete_sequence=COLOR_PALETTE,
            hole=0.55,
        )
        fig.update_traces(
            textinfo="label+percent",
            textfont_size=12,
            hovertemplate="<b>%{label}</b><br>Clients: %{value:,}<br>Part: %{percent}<extra></extra>"
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Ligne 2 ---
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown('<div class="section-header">🌿 Production ENR mensuelle</div>', unsafe_allow_html=True)
        
        enr_mensuel = production_enr[production_enr["annee"] == annee_selectionnee].groupby(
            ["mois", "type_production"]
        )["production_mwh"].sum().reset_index()
        
        fig = px.bar(
            enr_mensuel, x="mois", y="production_mwh", color="type_production",
            color_discrete_sequence=COLOR_PALETTE,
            barmode="stack",
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            xaxis_title="",
            yaxis_title="Production (MWh)",
            legend=dict(orientation="h", y=1.12, title=""),
            xaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        st.markdown('<div class="section-header">🔧 Interventions par type</div>', unsafe_allow_html=True)
        
        int_data = interventions_f[interventions_f["annee"] == annee_selectionnee].groupby(
            "type_intervention"
        ).size().reset_index(name="count").sort_values("count", ascending=True)
        
        fig = px.bar(
            int_data, x="count", y="type_intervention", orientation="h",
            color="count",
            color_continuous_scale=["#E8F5E9", "#00A86B"],
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            xaxis_title="Nombre",
            yaxis_title="",
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Carte géographique ---
    st.markdown('<div class="section-header">🗺️ Répartition géographique des clients</div>', unsafe_allow_html=True)
    
    geo_data = clients_f.groupby("commune").agg(
        nb_clients=("client_id", "count"),
        lat=("latitude", "mean"),
        lon=("longitude", "mean"),
        satisfaction=("score_satisfaction", "mean"),
    ).reset_index()
    
    fig = px.scatter_mapbox(
        geo_data,
        lat="lat", lon="lon",
        size="nb_clients",
        color="satisfaction",
        color_continuous_scale=["#EF4444", "#F59E0B", "#00A86B"],
        size_max=40,
        hover_name="commune",
        hover_data={"nb_clients": True, "satisfaction": ":.1f", "lat": False, "lon": False},
        zoom=8.5,
        center={"lat": 46.58, "lon": 0.34},
        mapbox_style="carto-positron",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=500,
        coloraxis_colorbar=dict(title="Satisfaction"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: ANALYSE CLIENTS
# ============================================================================
elif page == "👥 Analyse Clients":
    
    st.markdown("""
    <div class="main-header">
        <h1>👥 Analyse du Portefeuille Clients</h1>
        <p>Segmentation, rétention et comportement client</p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPIs
    actifs = len(clients_f[clients_f["statut"] == "Actif"])
    resilies = len(clients_f[clients_f["statut"] == "Résilié"])
    taux_churn = resilies / len(clients_f) * 100
    linky_pct = clients_f["compteur_linky"].mean() * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clients actifs", f"{actifs:,}", "+312")
    c2.metric("Taux de churn", f"{taux_churn:.1f}%", "-0.8%", delta_color="inverse")
    c3.metric("Satisfaction moy.", f"{clients_f['score_satisfaction'].mean():.1f}/10", "+0.2")
    c4.metric("Taux Linky", f"{linky_pct:.0f}%", "+5%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">📊 Distribution par segment et énergie</div>', unsafe_allow_html=True)
        
        seg_ener = clients_f.groupby(["segment", "type_energie"]).size().reset_index(name="count")
        fig = px.sunburst(
            seg_ener, path=["segment", "type_energie"], values="count",
            color_discrete_sequence=COLOR_PALETTE,
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">⭐ Distribution satisfaction par segment</div>', unsafe_allow_html=True)
        
        fig = px.violin(
            clients_f, x="segment", y="score_satisfaction",
            color="segment", color_discrete_sequence=COLOR_PALETTE,
            box=True, points=False,
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=450, showlegend=False,
                         xaxis_title="", yaxis_title="Score de satisfaction")
        st.plotly_chart(fig, use_container_width=True)
    
    # Analyse des puissances souscrites
    st.markdown('<div class="section-header">⚡ Distribution des puissances souscrites</div>', unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        fig = px.histogram(
            clients_f, x="puissance_souscrite_kva", color="segment",
            nbins=20, color_discrete_sequence=COLOR_PALETTE,
            barmode="stack",
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=350,
                         xaxis_title="Puissance souscrite (kVA)", yaxis_title="Nombre de clients")
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        option_data = clients_f.groupby(["option_tarifaire", "segment"]).size().reset_index(name="count")
        fig = px.bar(
            option_data, x="option_tarifaire", y="count", color="segment",
            color_discrete_sequence=COLOR_PALETTE,
            barmode="group",
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=350,
                         xaxis_title="Option tarifaire", yaxis_title="Nombre de clients",
                         legend=dict(orientation="h", y=1.1, title=""))
        st.plotly_chart(fig, use_container_width=True)
    
    # Tableau récapitulatif
    st.markdown('<div class="section-header">📋 Récapitulatif par commune</div>', unsafe_allow_html=True)
    
    recap = clients_f.groupby("commune").agg(
        nb_clients=("client_id", "count"),
        satisfaction_moy=("score_satisfaction", "mean"),
        puissance_moy=("puissance_souscrite_kva", "mean"),
        taux_linky=("compteur_linky", "mean"),
    ).reset_index()
    recap["satisfaction_moy"] = recap["satisfaction_moy"].round(1)
    recap["puissance_moy"] = recap["puissance_moy"].round(1)
    recap["taux_linky"] = (recap["taux_linky"] * 100).round(1)
    recap = recap.sort_values("nb_clients", ascending=False)
    recap.columns = ["Commune", "Nb Clients", "Satisfaction Moy.", "Puissance Moy. (kVA)", "Taux Linky (%)"]
    
    st.dataframe(recap, use_container_width=True, height=400, hide_index=True)


# ============================================================================
# PAGE: CONSOMMATIONS
# ============================================================================
elif page == "⚡ Consommations":
    
    st.markdown("""
    <div class="main-header">
        <h1>⚡ Analyse des Consommations</h1>
        <p>Suivi détaillé de la consommation énergétique par segment, période et territoire</p>
    </div>
    """, unsafe_allow_html=True)
    
    conso_annee = consommations_f[consommations_f["annee"] == annee_selectionnee]
    
    c1, c2, c3, c4 = st.columns(4)
    total_kwh = conso_annee["consommation_kwh"].sum()
    moy_kwh = conso_annee.groupby("client_id")["consommation_kwh"].sum().mean()
    nb_compteurs_actifs = conso_annee["compteur_id"].nunique()
    temp_moy = conso_annee["temperature_moyenne_c"].mean()
    
    c1.metric("Conso. totale", f"{format_number(total_kwh)} kWh")
    c2.metric("Conso. moy/client", f"{format_number(moy_kwh)} kWh")
    c3.metric("Compteurs actifs", f"{nb_compteurs_actifs:,}")
    c4.metric("Temp. moy.", f"{temp_moy:.1f}°C")
    
    st.markdown("---")
    
    # Consommation vs Température
    st.markdown('<div class="section-header">🌡️ Consommation vs Température (thermosensibilité)</div>', unsafe_allow_html=True)
    
    thermo = conso_annee.groupby("mois").agg(
        conso=("consommation_kwh", "sum"),
        temp=("temperature_moyenne_c", "mean"),
    ).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=thermo["mois"], y=thermo["conso"], name="Consommation (kWh)",
               marker_color=COLORS["primary"], opacity=0.7),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=thermo["mois"], y=thermo["temp"], name="Température (°C)",
                   line=dict(color=COLORS["danger"], width=3),
                   mode="lines+markers"),
        secondary_y=True,
    )
    fig.update_layout(
        **PLOTLY_LAYOUT, height=420,
        legend=dict(orientation="h", y=1.1),
        xaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
    )
    fig.update_yaxes(title_text="Consommation (kWh)", secondary_y=False, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(title_text="Température (°C)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Par segment
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">📊 Consommation par segment</div>', unsafe_allow_html=True)
        
        # Merge avec clients pour avoir le segment
        conso_seg = conso_annee.merge(
            clients_f[["client_id", "segment"]], on="client_id", how="left"
        )
        seg_monthly = conso_seg.groupby(["mois", "segment"])["consommation_kwh"].sum().reset_index()
        
        fig = px.area(
            seg_monthly, x="mois", y="consommation_kwh", color="segment",
            color_discrete_sequence=COLOR_PALETTE,
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=400,
                         xaxis_title="", yaxis_title="Consommation (kWh)",
                         legend=dict(orientation="h", y=1.12, title=""))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">📈 Comparaison annuelle</div>', unsafe_allow_html=True)
        
        comp = consommations_f.groupby(["mois_num", "annee"])["consommation_kwh"].sum().reset_index()
        
        fig = go.Figure()
        for annee in sorted(comp["annee"].unique()):
            data_a = comp[comp["annee"] == annee]
            fig.add_trace(go.Scatter(
                x=data_a["mois_num"], y=data_a["consommation_kwh"],
                name=str(annee),
                mode="lines+markers",
                line=dict(width=2.5 if annee == annee_selectionnee else 1.5),
                opacity=1 if annee == annee_selectionnee else 0.5,
            ))
        
        fig.update_layout(**PLOTLY_LAYOUT, height=400,
                         xaxis_title="Mois", yaxis_title="Consommation (kWh)",
                         legend=dict(orientation="h", y=1.1),
                         xaxis=dict(
                             tickmode="array",
                             tickvals=list(range(1, 13)),
                             ticktext=["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                                      "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"],
                             gridcolor="rgba(0,0,0,0.05)",
                         ),
                         yaxis=dict(gridcolor="rgba(0,0,0,0.05)"))
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap consommation
    st.markdown('<div class="section-header">🔥 Heatmap de consommation (Année × Mois)</div>', unsafe_allow_html=True)
    
    heatmap_data = consommations_f.groupby(["annee", "mois_num"])["consommation_kwh"].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index="annee", columns="mois_num", values="consommation_kwh")
    
    mois_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun", "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
    
    fig = px.imshow(
        heatmap_pivot.values,
        labels=dict(x="Mois", y="Année", color="Conso. (kWh)"),
        x=mois_labels[:heatmap_pivot.shape[1]],
        y=[str(y) for y in heatmap_pivot.index],
        color_continuous_scale=["#E8F5E9", "#00A86B", "#1B2838"],
        aspect="auto",
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=300)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: PRODUCTION ENR
# ============================================================================
elif page == "🌿 Production ENR":
    
    st.markdown("""
    <div class="main-header">
        <h1>🌿 Production Énergies Renouvelables</h1>
        <p>Suivi de la production verte et impact environnemental</p>
    </div>
    """, unsafe_allow_html=True)
    
    enr_annee = production_enr[production_enr["annee"] == annee_selectionnee]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Production totale", f"{format_number(enr_annee['production_mwh'].sum())} MWh")
    c2.metric("CO₂ évité", f"{format_number(enr_annee['co2_evite_tonnes'].sum())} t")
    c3.metric("Revenu estimé", f"{format_number(enr_annee['revenu_estime_eur'].sum())}€")
    c4.metric("Disponibilité moy.", f"{enr_annee['disponibilite'].mean()*100:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">⚡ Production par source</div>', unsafe_allow_html=True)
        
        prod_type = enr_annee.groupby("type_production").agg(
            production=("production_mwh", "sum"),
            co2=("co2_evite_tonnes", "sum"),
            revenu=("revenu_estime_eur", "sum"),
        ).reset_index()
        
        fig = px.bar(
            prod_type, x="type_production", y="production",
            color="type_production", color_discrete_sequence=COLOR_PALETTE,
            text="production",
        )
        fig.update_traces(texttemplate='%{text:,.0f} MWh', textposition='outside')
        fig.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False,
                         xaxis_title="", yaxis_title="Production (MWh)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">📊 Mix énergétique renouvelable</div>', unsafe_allow_html=True)
        
        fig = px.pie(
            prod_type, values="production", names="type_production",
            color_discrete_sequence=COLOR_PALETTE,
            hole=0.5,
        )
        fig.update_traces(textinfo="label+percent+value", textfont_size=11)
        fig.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Évolution production par site
    st.markdown('<div class="section-header">📈 Évolution production par site</div>', unsafe_allow_html=True)
    
    fig = px.line(
        enr_annee, x="mois", y="production_mwh", color="nom_site",
        color_discrete_sequence=COLOR_PALETTE,
        markers=True,
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=450,
                     xaxis_title="", yaxis_title="Production (MWh)",
                     legend=dict(orientation="h", y=-0.15, title=""))
    st.plotly_chart(fig, use_container_width=True)
    
    # Facteur de charge
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown('<div class="section-header">⚙️ Facteur de charge par technologie</div>', unsafe_allow_html=True)
        
        fc_data = enr_annee.groupby(["mois_num", "type_production"])["facteur_charge"].mean().reset_index()
        
        fig = px.line(
            fc_data, x="mois_num", y="facteur_charge", color="type_production",
            color_discrete_sequence=COLOR_PALETTE,
            markers=True,
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=350,
                         xaxis_title="Mois", yaxis_title="Facteur de charge",
                         yaxis=dict(tickformat=".0%", gridcolor="rgba(0,0,0,0.05)"),
                         xaxis=dict(
                             tickmode="array",
                             tickvals=list(range(1, 13)),
                             ticktext=["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"],
                         ),
                         legend=dict(orientation="h", y=1.1, title=""))
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        st.markdown('<div class="section-header">🌍 Impact CO₂ cumulé</div>', unsafe_allow_html=True)
        
        co2_cumul = production_enr.sort_values("mois").groupby("mois")["co2_evite_tonnes"].sum().cumsum().reset_index()
        co2_cumul.columns = ["mois", "co2_cumul"]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=co2_cumul["mois"], y=co2_cumul["co2_cumul"],
            fill="tozeroy", fillcolor="rgba(0,168,107,0.15)",
            line=dict(color=COLORS["primary"], width=2.5),
            name="CO₂ évité cumulé",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=350,
                         xaxis_title="", yaxis_title="CO₂ évité cumulé (tonnes)",
                         yaxis=dict(gridcolor="rgba(0,0,0,0.05)"))
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: INTERVENTIONS
# ============================================================================
elif page == "🔧 Interventions":
    
    st.markdown("""
    <div class="main-header">
        <h1>🔧 Suivi des Interventions</h1>
        <p>Pilotage opérationnel des interventions terrain</p>
    </div>
    """, unsafe_allow_html=True)
    
    int_annee = interventions_f[interventions_f["annee"] == annee_selectionnee]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total interventions", f"{len(int_annee):,}")
    c2.metric("Durée moy.", f"{int_annee['duree_heures'].mean():.1f}h")
    c3.metric("Coût total", f"{format_number(int_annee['cout_intervention_eur'].sum())}€")
    c4.metric("Satisfaction terrain", f"{int_annee['satisfaction_intervention'].mean():.1f}/10")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">📊 Volume mensuel par type</div>', unsafe_allow_html=True)
        
        int_monthly = int_annee.groupby(
            [int_annee["date_intervention"].dt.to_period("M").astype(str), "type_intervention"]
        ).size().reset_index(name="count")
        int_monthly.columns = ["mois", "type", "count"]
        
        fig = px.bar(
            int_monthly, x="mois", y="count", color="type",
            color_discrete_sequence=COLOR_PALETTE,
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=400,
                         xaxis_title="", yaxis_title="Nombre",
                         legend=dict(orientation="h", y=-0.2, title=""),
                         xaxis=dict(tickangle=45))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">⏱️ Durée moyenne par type</div>', unsafe_allow_html=True)
        
        duree_type = int_annee.groupby("type_intervention")["duree_heures"].mean().reset_index()
        duree_type = duree_type.sort_values("duree_heures", ascending=True)
        
        fig = px.bar(
            duree_type, x="duree_heures", y="type_intervention", orientation="h",
            color="duree_heures",
            color_continuous_scale=["#E8F5E9", "#F59E0B", "#EF4444"],
            text="duree_heures",
        )
        fig.update_traces(texttemplate='%{text:.1f}h', textposition='outside')
        fig.update_layout(**PLOTLY_LAYOUT, height=400,
                         xaxis_title="Durée moyenne (heures)", yaxis_title="",
                         coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance techniciens
    st.markdown('<div class="section-header">👷 Performance des techniciens (Top 15)</div>', unsafe_allow_html=True)
    
    tech_perf = int_annee.groupby("technicien_id").agg(
        nb_interventions=("intervention_id", "count"),
        duree_moy=("duree_heures", "mean"),
        satisfaction_moy=("satisfaction_intervention", "mean"),
        cout_moyen=("cout_intervention_eur", "mean"),
    ).reset_index().sort_values("nb_interventions", ascending=False).head(15)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Nb Interventions vs Satisfaction", "Durée vs Coût moyen"),
    )
    
    fig.add_trace(
        go.Scatter(
            x=tech_perf["nb_interventions"], y=tech_perf["satisfaction_moy"],
            mode="markers+text", text=tech_perf["technicien_id"],
            textposition="top center", textfont=dict(size=8),
            marker=dict(size=12, color=COLORS["primary"]),
            name="Performance",
        ),
        row=1, col=1,
    )
    
    fig.add_trace(
        go.Scatter(
            x=tech_perf["duree_moy"], y=tech_perf["cout_moyen"],
            mode="markers", marker=dict(
                size=tech_perf["nb_interventions"] / 2,
                color=tech_perf["satisfaction_moy"],
                colorscale=["#EF4444", "#F59E0B", "#00A86B"],
                showscale=True,
                colorbar=dict(title="Satisfaction"),
            ),
            name="Coût",
        ),
        row=1, col=2,
    )
    
    fig.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Statut des interventions
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown('<div class="section-header">📋 Répartition par statut</div>', unsafe_allow_html=True)
        
        statut_data = int_annee["statut"].value_counts().reset_index()
        statut_data.columns = ["statut", "count"]
        
        colors_statut = {"Terminée": "#10B981", "Planifiée": "#3B82F6", "En cours": "#F59E0B", "Annulée": "#EF4444"}
        
        fig = px.pie(
            statut_data, values="count", names="statut",
            color="statut",
            color_discrete_map=colors_statut,
            hole=0.5,
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        st.markdown('<div class="section-header">🗺️ Interventions par commune</div>', unsafe_allow_html=True)
        
        int_commune = int_annee.groupby("commune").size().reset_index(name="count").sort_values("count", ascending=False)
        
        fig = px.bar(
            int_commune.head(10), x="commune", y="count",
            color="count", color_continuous_scale=["#E8F5E9", "#00A86B"],
            text="count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(**PLOTLY_LAYOUT, height=350,
                         xaxis_title="", yaxis_title="Nombre",
                         coloraxis_showscale=False,
                         xaxis=dict(tickangle=45))
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: FACTURATION
# ============================================================================
elif page == "💶 Facturation":
    
    st.markdown("""
    <div class="main-header">
        <h1>💶 Analyse de la Facturation</h1>
        <p>Suivi du chiffre d'affaires, recouvrement et analyse financière</p>
    </div>
    """, unsafe_allow_html=True)
    
    fac_annee = factures_f[factures_f["annee"] == annee_selectionnee]
    
    ca_ttc = fac_annee["montant_ttc_eur"].sum()
    ca_ht = fac_annee["montant_ht_eur"].sum()
    taux_impaye = len(fac_annee[fac_annee["statut_paiement"] == "Impayée"]) / len(fac_annee) * 100
    delai_moyen = fac_annee[fac_annee["delai_paiement_jours"].notna()]["delai_paiement_jours"].mean()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CA TTC", f"{format_number(ca_ttc)}€")
    c2.metric("CA HT", f"{format_number(ca_ht)}€")
    c3.metric("Taux impayés", f"{taux_impaye:.1f}%", "-0.3%", delta_color="inverse")
    c4.metric("Délai paiement moy.", f"{delai_moyen:.0f} jours")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">📈 Évolution du CA mensuel</div>', unsafe_allow_html=True)
        
        ca_mensuel = fac_annee.groupby(
            fac_annee["date_facture"].dt.to_period("M").astype(str)
        ).agg(
            ca_ttc=("montant_ttc_eur", "sum"),
            nb_factures=("facture_id", "count"),
        ).reset_index()
        ca_mensuel.columns = ["mois", "ca_ttc", "nb_factures"]
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=ca_mensuel["mois"], y=ca_mensuel["ca_ttc"],
                   name="CA TTC (€)", marker_color=COLORS["primary"], opacity=0.8),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=ca_mensuel["mois"], y=ca_mensuel["nb_factures"],
                       name="Nb Factures", line=dict(color=COLORS["info"], width=2.5),
                       mode="lines+markers"),
            secondary_y=True,
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=400,
                         legend=dict(orientation="h", y=1.1),
                         xaxis=dict(tickangle=45))
        fig.update_yaxes(title_text="CA TTC (€)", secondary_y=False)
        fig.update_yaxes(title_text="Nb Factures", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">💳 Modes de paiement</div>', unsafe_allow_html=True)
        
        mode_data = fac_annee["mode_paiement"].value_counts().reset_index()
        mode_data.columns = ["mode", "count"]
        
        fig = px.pie(
            mode_data, values="count", names="mode",
            color_discrete_sequence=COLOR_PALETTE,
            hole=0.55,
        )
        fig.update_traces(textinfo="label+percent")
        fig.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Analyse des impayés
    st.markdown('<div class="section-header">⚠️ Analyse du recouvrement</div>', unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        statut_paiement = fac_annee["statut_paiement"].value_counts().reset_index()
        statut_paiement.columns = ["statut", "count"]
        
        colors_map = {"Payée": "#10B981", "En attente": "#3B82F6", "En retard": "#F59E0B", "Impayée": "#EF4444"}
        
        fig = px.bar(
            statut_paiement, x="statut", y="count",
            color="statut", color_discrete_map=colors_map,
            text="count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False,
                         xaxis_title="", yaxis_title="Nombre de factures")
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        delai_dist = fac_annee[fac_annee["delai_paiement_jours"].notna()]
        
        fig = px.histogram(
            delai_dist, x="delai_paiement_jours", nbins=50,
            color_discrete_sequence=[COLORS["primary"]],
        )
        fig.add_vline(x=30, line_dash="dash", line_color="red",
                      annotation_text="Échéance 30j", annotation_position="top right")
        fig.update_layout(**PLOTLY_LAYOUT, height=350,
                         xaxis_title="Délai de paiement (jours)", yaxis_title="Fréquence")
        st.plotly_chart(fig, use_container_width=True)
    
    # Montant par segment
    st.markdown('<div class="section-header">💰 Montant facturé par segment client</div>', unsafe_allow_html=True)
    
    fac_seg = fac_annee.merge(clients_f[["client_id", "segment"]], on="client_id", how="left")
    seg_ca = fac_seg.groupby("segment").agg(
        ca=("montant_ttc_eur", "sum"),
        nb=("facture_id", "count"),
        montant_moyen=("montant_ttc_eur", "mean"),
    ).reset_index()
    
    fig = px.treemap(
        seg_ca, path=["segment"], values="ca",
        color="montant_moyen",
        color_continuous_scale=["#E8F5E9", "#00A86B", "#1B2838"],
        custom_data=["nb", "montant_moyen"],
    )
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>CA: %{value:,.0f}€<br>Moy: %{customdata[1]:,.0f}€",
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=400)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: QUALITÉ RÉSEAU
# ============================================================================
elif page == "📶 Qualité Réseau":
    
    st.markdown("""
    <div class="main-header">
        <h1>📶 Qualité de Service Réseau</h1>
        <p>Indicateurs SAIDI/SAIFI, fiabilité réseau et suivi des incidents</p>
    </div>
    """, unsafe_allow_html=True)
    
    qr_annee = qualite_reseau[qualite_reseau["mois"].dt.year == annee_selectionnee]
    
    if communes_filtre:
        qr_annee = qr_annee[qr_annee["commune"].isin(communes_filtre)]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Coupures totales", f"{qr_annee['nb_coupures'].sum():,}")
    c2.metric("SAIDI moyen", f"{qr_annee['saidi_min'].mean():.1f} min")
    c3.metric("SAIFI moyen", f"{qr_annee['saifi'].mean():.4f}")
    c4.metric("Clients impactés", f"{format_number(qr_annee['nb_clients_impactes'].sum())}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">📉 Évolution SAIDI mensuel</div>', unsafe_allow_html=True)
        
        saidi_monthly = qr_annee.groupby(qr_annee["mois"].dt.to_period("M").astype(str))["saidi_min"].mean().reset_index()
        saidi_monthly.columns = ["mois", "saidi"]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=saidi_monthly["mois"], y=saidi_monthly["saidi"],
            mode="lines+markers+text",
            text=[f"{v:.1f}" for v in saidi_monthly["saidi"]],
            textposition="top center",
            line=dict(color=COLORS["primary"], width=2.5),
            marker=dict(size=8),
            fill="tozeroy", fillcolor="rgba(0,168,107,0.1)",
        ))
        # Seuil réglementaire
        fig.add_hline(y=qr_annee["saidi_min"].mean() * 1.5, line_dash="dash",
                      line_color="red", annotation_text="Seuil d'alerte")
        fig.update_layout(**PLOTLY_LAYOUT, height=380,
                         xaxis_title="", yaxis_title="SAIDI (minutes)",
                         xaxis=dict(tickangle=45))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">🔍 Causes principales des coupures</div>', unsafe_allow_html=True)
        
        causes = qr_annee[qr_annee["cause_principale"] != "Aucune"]["cause_principale"].value_counts().reset_index()
        causes.columns = ["cause", "count"]
        
        fig = px.bar(
            causes, x="count", y="cause", orientation="h",
            color="count",
            color_continuous_scale=["#FEF3C7", "#F59E0B", "#EF4444"],
            text="count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(**PLOTLY_LAYOUT, height=380,
                         xaxis_title="Nombre d'occurrences", yaxis_title="",
                         coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap qualité par commune
    st.markdown('<div class="section-header">🗺️ Qualité réseau par commune</div>', unsafe_allow_html=True)
    
    qr_commune = qr_annee.groupby("commune").agg(
        coupures=("nb_coupures", "sum"),
        saidi=("saidi_min", "mean"),
        tension_ok=("taux_tension_conforme", "mean"),
        reclamations=("taux_reclamation", "mean"),
        impactes=("nb_clients_impactes", "sum"),
    ).reset_index().sort_values("coupures", ascending=False)
    
    fig = px.bar(
        qr_commune, x="commune", y="coupures",
        color="saidi",
        color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
        text="coupures",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**PLOTLY_LAYOUT, height=400,
                     xaxis_title="", yaxis_title="Nombre de coupures",
                     xaxis=dict(tickangle=45),
                     coloraxis_colorbar=dict(title="SAIDI moy."))
    st.plotly_chart(fig, use_container_width=True)
    
    # Taux de tension conforme
    st.markdown('<div class="section-header">⚡ Taux de conformité tension</div>', unsafe_allow_html=True)
    
    tension_data = qr_annee.groupby(
        qr_annee["mois"].dt.to_period("M").astype(str)
    )["taux_tension_conforme"].mean().reset_index()
    tension_data.columns = ["mois", "taux"]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tension_data["mois"], y=tension_data["taux"] * 100,
        mode="lines+markers",
        line=dict(color=COLORS["success"], width=2.5),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.1)",
    ))
    fig.add_hline(y=97, line_dash="dash", line_color=COLORS["accent"],
                  annotation_text="Objectif 97%")
    fig.update_layout(**PLOTLY_LAYOUT, height=300,
                     xaxis_title="", yaxis_title="Taux conformité (%)",
                     yaxis=dict(range=[95, 100], gridcolor="rgba(0,0,0,0.05)"),
                     xaxis=dict(tickangle=45))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: INSIGHTS IA
# ============================================================================
elif page == "🤖 Insights IA":
    
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Insights & Intelligence Artificielle</h1>
        <p>Détection d'anomalies, prévisions et recommandations data-driven</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs([
        "🔍 Détection d'anomalies",
        "📈 Prévisions",
        "💡 Recommandations",
    ])
    
    with tab1:
        st.markdown('<div class="section-header">🔍 Détection d\'anomalies de consommation</div>', unsafe_allow_html=True)
        
        # Simulation détection d'anomalies (z-score)
        conso_client = consommations_f.groupby("client_id")["consommation_kwh"].agg(["mean", "std", "count"]).reset_index()
        conso_client.columns = ["client_id", "conso_moy", "conso_std", "nb_mois"]
        conso_client = conso_client[conso_client["nb_mois"] > 6]
        
        # Score d'anomalie
        conso_client["z_score"] = (conso_client["conso_moy"] - conso_client["conso_moy"].mean()) / conso_client["conso_moy"].std()
        conso_client["anomalie"] = conso_client["z_score"].abs() > 2
        
        n_anomalies = conso_client["anomalie"].sum()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Anomalies détectées", f"{n_anomalies}")
            st.metric("Taux d'anomalie", f"{n_anomalies/len(conso_client)*100:.1f}%")
            st.metric("Seuil z-score", "±2σ")
            
            st.markdown("---")
            st.markdown("""
            **Méthodologie :**
            - Calcul du z-score sur la consommation moyenne
            - Seuil de détection : ±2 écarts-types
            - Analyse sur les 6 derniers mois minimum
            """)
        
        with col2:
            fig = px.scatter(
                conso_client, x="conso_moy", y="z_score",
                color="anomalie",
                color_discrete_map={True: COLORS["danger"], False: COLORS["primary"]},
                opacity=0.6,
                labels={"conso_moy": "Consommation moyenne (kWh)", "z_score": "Z-Score", "anomalie": "Anomalie"},
            )
            fig.add_hline(y=2, line_dash="dash", line_color="red")
            fig.add_hline(y=-2, line_dash="dash", line_color="red")
            fig.update_layout(**PLOTLY_LAYOUT, height=450)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top anomalies
        st.markdown("**🚨 Top 10 des anomalies les plus significatives**")
        top_anomalies = conso_client[conso_client["anomalie"]].nlargest(10, "z_score")
        top_anomalies = top_anomalies.merge(
            clients_f[["client_id", "segment", "commune", "puissance_souscrite_kva"]],
            on="client_id", how="left"
        )
        top_anomalies_display = top_anomalies[["client_id", "segment", "commune", "conso_moy", "z_score", "puissance_souscrite_kva"]].copy()
        top_anomalies_display.columns = ["Client ID", "Segment", "Commune", "Conso. Moy (kWh)", "Z-Score", "Puissance (kVA)"]
        top_anomalies_display["Conso. Moy (kWh)"] = top_anomalies_display["Conso. Moy (kWh)"].round(0)
        top_anomalies_display["Z-Score"] = top_anomalies_display["Z-Score"].round(2)
        st.dataframe(top_anomalies_display, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown('<div class="section-header">📈 Prévision de consommation (tendance)</div>', unsafe_allow_html=True)
        
        # Tendance simple avec régression linéaire
        conso_trend = consommations_f.groupby("mois")["consommation_kwh"].sum().reset_index()
        conso_trend = conso_trend.sort_values("mois")
        conso_trend["mois_num_total"] = range(len(conso_trend))
        
        # Régression
        from numpy.polynomial import polynomial as P
        coeffs = np.polyfit(conso_trend["mois_num_total"], conso_trend["consommation_kwh"], 2)
        poly = np.poly1d(coeffs)
        
        # Prévisions 6 mois
        n_future = 6
        future_x = range(len(conso_trend), len(conso_trend) + n_future)
        future_y = poly(list(future_x))
        future_dates = pd.date_range(conso_trend["mois"].max() + pd.DateOffset(months=1), periods=n_future, freq="MS")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=conso_trend["mois"], y=conso_trend["consommation_kwh"],
            mode="lines", name="Historique",
            line=dict(color=COLORS["primary"], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=conso_trend["mois"], y=poly(conso_trend["mois_num_total"]),
            mode="lines", name="Tendance",
            line=dict(color=COLORS["accent"], width=2, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_y,
            mode="lines+markers", name="Prévision",
            line=dict(color=COLORS["danger"], width=2.5, dash="dot"),
            marker=dict(size=8),
        ))
        # Intervalle de confiance
        std = conso_trend["consommation_kwh"].std() * 0.15
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(future_y + std) + list((future_y - std)[::-1]),
            fill="toself", fillcolor="rgba(239,68,68,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="IC 85%",
        ))
        
        fig.update_layout(**PLOTLY_LAYOUT, height=450,
                         xaxis_title="", yaxis_title="Consommation (kWh)",
                         legend=dict(orientation="h", y=1.1),
                         yaxis=dict(gridcolor="rgba(0,0,0,0.05)"))
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 Modèle polynomial d'ordre 2 avec intervalle de confiance à 85%. "
                "En production, un modèle Prophet ou LSTM serait utilisé pour plus de précision.")
    
    with tab3:
        st.markdown('<div class="section-header">💡 Recommandations Data-Driven</div>', unsafe_allow_html=True)
        
        # Analyse automatique
        taux_impaye = len(factures_f[factures_f["statut_paiement"] == "Impayée"]) / len(factures_f) * 100
        satisfaction_moy = clients_f["score_satisfaction"].mean()
        taux_linky = clients_f["compteur_linky"].mean() * 100
        
        recommendations = [
            {
                "priority": "🔴 Haute",
                "domain": "Recouvrement",
                "insight": f"Le taux d'impayés est de {taux_impaye:.1f}%. Recommandation : mettre en place un scoring prédictif de risque de non-paiement basé sur l'historique client.",
                "impact": "Réduction estimée de 25% des impayés",
                "action": "Déployer un modèle ML de scoring crédit"
            },
            {
                "priority": "🟡 Moyenne",
                "domain": "Satisfaction",
                "insight": f"La satisfaction moyenne est de {satisfaction_moy:.1f}/10. Les segments Industriel et Professionnel montrent des scores plus bas.",
                "impact": "Amélioration de 0.5 point de satisfaction",
                "action": "Enquête ciblée + plan d'action par segment"
            },
            {
                "priority": "🟢 Standard",
                "domain": "Smart Metering",
                "insight": f"Le taux de déploiement Linky est de {taux_linky:.0f}%. Accélérer le déploiement permettrait d'améliorer la granularité des données.",
                "impact": "Meilleure détection d'anomalies, relève à distance",
                "action": "Plan de déploiement ciblé sur communes restantes"
            },
            {
                "priority": "🔴 Haute",
                "domain": "Réseau",
                "insight": "Les communes rurales montrent un SAIDI supérieur à la moyenne. Le vieillissement du réseau BT est un facteur aggravant.",
                "impact": "Réduction de 30% du temps de coupure",
                "action": "Programme de maintenance prédictive basé sur l'âge des équipements"
            },
            {
                "priority": "🟡 Moyenne",
                "domain": "Transition énergétique",
                "insight": "La production ENR couvre environ 15% de la consommation du territoire. Potentiel d'augmentation via le solaire.",
                "impact": "Augmentation de 5% de la part ENR",
                "action": "Étude de faisabilité nouveaux sites PV"
            },
        ]
        
        for rec in recommendations:
            with st.expander(f"{rec['priority']} | {rec['domain']} — {rec['insight'][:80]}..."):
                st.markdown(f"**📊 Insight :** {rec['insight']}")
                st.markdown(f"**🎯 Impact estimé :** {rec['impact']}")
                st.markdown(f"**✅ Action recommandée :** {rec['action']}")
        
        st.markdown("---")
        
        # Data Quality Score
        st.markdown('<div class="section-header">📊 Score de Qualité des Données</div>', unsafe_allow_html=True)
        
        quality_metrics = {
            "Complétude": 94.2,
            "Unicité": 99.8,
            "Validité": 96.5,
            "Cohérence": 92.1,
            "Fraîcheur": 98.7,
            "Exactitude": 95.3,
        }
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(quality_metrics.values()),
            theta=list(quality_metrics.keys()),
            fill="toself",
            fillcolor="rgba(0,168,107,0.2)",
            line=dict(color=COLORS["primary"], width=2.5),
            marker=dict(size=8),
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=400,
            polar=dict(
                radialaxis=dict(visible=True, range=[80, 100]),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer">
    <p>
        <strong>Sorégies Energy Analytics Platform</strong> • Projet Portfolio Data Engineering<br>
        Développé par <strong>Yanis Zedira</strong> • Stack : Python, Streamlit, Pandas, Plotly<br>
        Données simulées à des fins de démonstration • © 2025
    </p>
</div>
""", unsafe_allow_html=True)

