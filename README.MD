# ⚡ Sorégies Energy Analytics Platform

> Plateforme d'analyse de données pour la distribution d'énergie - Territoire Vienne (86)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Contexte

Projet de data analytics appliqué au secteur de l'énergie, simulant les données opérationnelles
d'un distributeur d'énergie local (ELD). Ce dashboard couvre l'ensemble de la chaîne de valeur :
gestion du portefeuille clients, suivi des consommations, production d'énergies renouvelables,
interventions terrain, facturation et qualité de service réseau.

## 🏗️ Architecture
Pipeline de données :
[Génération CSV] → [Chargement Pandas] → [Transformations] → [Visualisation Streamlit]

Modélisation :
Tables de faits : consommations, interventions, factures, production_enr, qualite_reseau
Tables de dimensions : clients, compteurs


## 📊 Fonctionnalités

- **Vue d'ensemble** : KPIs clés, carte géographique, tendances globales
- **Analyse Clients** : Segmentation, satisfaction, déploiement smart meters
- **Consommations** : Thermosensibilité, comparaison annuelle, heatmaps
- **Production ENR** : Mix énergétique, facteur de charge, impact CO₂
- **Interventions** : Pilotage opérationnel, performance techniciens
- **Facturation** : CA, recouvrement, analyse des impayés
- **Qualité Réseau** : SAIDI/SAIFI, conformité tension, causes de coupures
- **Insights IA** : Détection d'anomalies, prévisions, recommandations

## 🚀 Installation

```bash
git clone https://github.com/[TON-USER]/soregies-energy-analytics.git
cd soregies-energy-analytics
pip install -r requirements.txt
python generate_dataset.py
streamlit run app.py

🛠️ Stack technique
Outil	Usage
Python 3.11	Langage principal
Streamlit	Interface web interactive
Pandas	Manipulation de données
Plotly	Visualisations avancées
NumPy	Calculs numériques
📁 Données

Dataset synthétique réaliste couvrant :

    15 000 clients (5 segments)
    16 500 compteurs (Linky, Gazpar, traditionnels)
    300 000+ lignes de consommation mensuelle
    12 000 interventions terrain
    45 000 factures
    8 sites de production ENR
    15 communes du territoire

👤 Auteur

Yanis ZEDIRA - Data Engineer / Analytics Engineer
📄 License

MIT License - Données simulées à des fins de démonstration uniquement.


---

## 🚀 Comment lancer le projet

### En local :
```bash
# 1. Créer le dossier et les fichiers
mkdir soregies-energy-analytics
cd soregies-energy-analytics
mkdir data .streamlit

# 2. Copier les fichiers (app.py, generate_dataset.py, requirements.txt, etc.)

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Générer les données
python generate_dataset.py

# 5. Lancer le dashboard
streamlit run app.py