"""
============================================================================
 SORÉGIES ENERGY ANALYTICS - Dataset Generator
 Générateur de données réalistes pour le secteur de l'énergie
 Simulation : Distribution électrique, gaz, et énergies renouvelables
 Territoire : Vienne (86) - Poitiers et agglomération
============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Période de simulation
DATE_DEBUT = datetime(2022, 1, 1)
DATE_FIN = datetime(2025, 5, 31)

# Communes de la Vienne (86) - Territoire Sorégies
COMMUNES = {
    "Poitiers": {"population": 90000, "code_postal": "86000", "lat": 46.5802, "lon": 0.3404},
    "Châtellerault": {"population": 31000, "code_postal": "86100", "lat": 46.8170, "lon": 0.5460},
    "Buxerolles": {"population": 10500, "code_postal": "86180", "lat": 46.6000, "lon": 0.3667},
    "Saint-Benoît": {"population": 7500, "code_postal": "86280", "lat": 46.5500, "lon": 0.3333},
    "Mignaloux-Beauvoir": {"population": 4200, "code_postal": "86550", "lat": 46.5333, "lon": 0.3833},
    "Chasseneuil-du-Poitou": {"population": 5200, "code_postal": "86360", "lat": 46.6333, "lon": 0.3667},
    "Migné-Auxances": {"population": 6500, "code_postal": "86440", "lat": 46.6167, "lon": 0.3167},
    "Jaunay-Marigny": {"population": 6800, "code_postal": "86130", "lat": 46.6833, "lon": 0.3500},
    "Montmorillon": {"population": 6200, "code_postal": "86500", "lat": 46.4267, "lon": 0.8700},
    "Loudun": {"population": 6800, "code_postal": "86200", "lat": 47.0000, "lon": 0.0833},
    "Civray": {"population": 2800, "code_postal": "86400", "lat": 46.1500, "lon": 0.2833},
    "Chauvigny": {"population": 7000, "code_postal": "86300", "lat": 46.5667, "lon": 0.6500},
    "Lusignan": {"population": 2800, "code_postal": "86600", "lat": 46.4333, "lon": 0.1167},
    "Vivonne": {"population": 4000, "code_postal": "86370", "lat": 46.4333, "lon": 0.2667},
    "Neuville-de-Poitou": {"population": 5800, "code_postal": "86170", "lat": 46.6833, "lon": 0.2500},
}

SEGMENTS_CLIENT = {
    "Résidentiel": 0.72,
    "Professionnel": 0.18,
    "Tertiaire": 0.06,
    "Industriel": 0.025,
    "Collectivité": 0.015,
}

TYPES_ENERGIE = ["Électricité", "Gaz", "Électricité + Gaz"]

TARIFS_ELEC = {
    "Base": 0.30,
    "HP/HC": 0.25,
    "Tempo": 0.22,
    "EJP": 0.20,
}

PUISSANCES_SOUSCRITES = [3, 6, 9, 12, 15, 18, 24, 30, 36]

N_CLIENTS = 15000
N_COMPTEURS = 16500

print("=" * 70)
print("  SORÉGIES - Générateur de Dataset Energy Analytics")
print("=" * 70)


# ============================================================================
# 1. TABLE CLIENTS
# ============================================================================
print("\n[1/7] Génération de la table CLIENTS...")

def generate_client_id(n):
    return [f"CLI-{86000 + i:06d}" for i in range(n)]

def generate_pdl():
    """Génère un Point de Livraison réaliste (14 chiffres)"""
    return f"{np.random.randint(10000000000000, 99999999999999, dtype=np.int64)}"

clients_ids = generate_client_id(N_CLIENTS)
segments = np.random.choice(
    list(SEGMENTS_CLIENT.keys()),
    size=N_CLIENTS,
    p=list(SEGMENTS_CLIENT.values())
)

communes_list = list(COMMUNES.keys())
# Pondération par population
pop_weights = np.array([COMMUNES[c]["population"] for c in communes_list], dtype=float)
pop_weights /= pop_weights.sum()
communes_clients = np.random.choice(communes_list, size=N_CLIENTS, p=pop_weights)

# Date de souscription
dates_souscription = [
    DATE_DEBUT - timedelta(days=np.random.randint(0, 3650)) for _ in range(N_CLIENTS)
]

# Type d'énergie
type_energie_probs = [0.55, 0.20, 0.25]
types_energie_clients = np.random.choice(TYPES_ENERGIE, size=N_CLIENTS, p=type_energie_probs)

# Puissance souscrite
puissances = []
for seg in segments:
    if seg == "Résidentiel":
        puissances.append(np.random.choice([3, 6, 9, 12], p=[0.05, 0.45, 0.35, 0.15]))
    elif seg == "Professionnel":
        puissances.append(np.random.choice([9, 12, 15, 18], p=[0.15, 0.35, 0.30, 0.20]))
    elif seg == "Tertiaire":
        puissances.append(np.random.choice([18, 24, 30, 36], p=[0.25, 0.30, 0.25, 0.20]))
    elif seg == "Industriel":
        puissances.append(np.random.choice([30, 36], p=[0.4, 0.6]))
    else:
        puissances.append(np.random.choice([12, 15, 18, 24], p=[0.2, 0.3, 0.3, 0.2]))

# Option tarifaire
options_tarifaires = []
for seg in segments:
    if seg in ["Résidentiel", "Collectivité"]:
        options_tarifaires.append(np.random.choice(
            ["Base", "HP/HC", "Tempo"], p=[0.30, 0.55, 0.15]
        ))
    else:
        options_tarifaires.append(np.random.choice(
            ["Base", "HP/HC", "Tempo", "EJP"], p=[0.20, 0.40, 0.20, 0.20]
        ))

# Statut client
statuts = np.random.choice(
    ["Actif", "Actif", "Actif", "Actif", "Résilié", "Suspendu"],
    size=N_CLIENTS
)

# Indicateur smart meter (Linky)
linky = np.random.choice([True, False], size=N_CLIENTS, p=[0.82, 0.18])

# Score satisfaction (1-10)
satisfaction = np.clip(np.random.normal(7.2, 1.5, N_CLIENTS), 1, 10).round(1)

clients_df = pd.DataFrame({
    "client_id": clients_ids,
    "segment": segments,
    "commune": communes_clients,
    "code_postal": [COMMUNES[c]["code_postal"] for c in communes_clients],
    "latitude": [COMMUNES[c]["lat"] + np.random.uniform(-0.02, 0.02) for c in communes_clients],
    "longitude": [COMMUNES[c]["lon"] + np.random.uniform(-0.02, 0.02) for c in communes_clients],
    "type_energie": types_energie_clients,
    "puissance_souscrite_kva": puissances,
    "option_tarifaire": options_tarifaires,
    "date_souscription": dates_souscription,
    "statut": statuts,
    "compteur_linky": linky,
    "score_satisfaction": satisfaction,
})

clients_df.to_csv(f"{OUTPUT_DIR}/clients.csv", index=False, sep=";")
print(f"   ✅ {len(clients_df)} clients générés")


# ============================================================================
# 2. TABLE COMPTEURS
# ============================================================================
print("\n[2/7] Génération de la table COMPTEURS...")

compteurs = []
compteur_idx = 0
for _, client in clients_df.iterrows():
    n_compteurs = 1 if client["type_energie"] != "Électricité + Gaz" else 2
    for j in range(n_compteurs):
        energie = client["type_energie"]
        if energie == "Électricité + Gaz":
            energie = "Électricité" if j == 0 else "Gaz"
        
        type_compteur = "Linky" if (energie == "Électricité" and client["compteur_linky"]) else \
                        "Gazpar" if (energie == "Gaz" and np.random.random() < 0.65) else \
                        "Électromécanique" if energie == "Électricité" else "Traditionnel"
        
        compteurs.append({
            "compteur_id": f"CPT-{86000 + compteur_idx:06d}",
            "pdl": generate_pdl(),
            "client_id": client["client_id"],
            "type_energie": energie,
            "type_compteur": type_compteur,
            "marque": np.random.choice(["Itron", "Sagemcom", "Landis+Gyr", "Iskraemeco"],
                                        p=[0.30, 0.35, 0.20, 0.15]),
            "date_installation": (
                client["date_souscription"] + timedelta(days=np.random.randint(0, 30))
            ).strftime("%Y-%m-%d"),
            "dernier_releve": (
                DATE_FIN - timedelta(days=np.random.randint(0, 60))
            ).strftime("%Y-%m-%d"),
            "statut": client["statut"],
        })
        compteur_idx += 1
        if compteur_idx >= N_COMPTEURS:
            break
    if compteur_idx >= N_COMPTEURS:
        break

compteurs_df = pd.DataFrame(compteurs)
compteurs_df.to_csv(f"{OUTPUT_DIR}/compteurs.csv", index=False, sep=";")
print(f"   ✅ {len(compteurs_df)} compteurs générés")


# ============================================================================
# 3. TABLE CONSOMMATIONS (données mensuelles)
# ============================================================================
print("\n[3/7] Génération de la table CONSOMMATIONS (mensuelles)...")

# Profils de consommation saisonniers (facteur multiplicatif par mois)
PROFIL_SAISONNIER_ELEC = {
    1: 1.45, 2: 1.35, 3: 1.15, 4: 0.90, 5: 0.75, 6: 0.70,
    7: 0.72, 8: 0.68, 9: 0.78, 10: 0.95, 11: 1.20, 12: 1.42
}

PROFIL_SAISONNIER_GAZ = {
    1: 1.80, 2: 1.65, 3: 1.30, 4: 0.85, 5: 0.45, 6: 0.25,
    7: 0.20, 8: 0.20, 9: 0.35, 10: 0.80, 11: 1.35, 12: 1.70
}

# Consommation annuelle moyenne par segment (kWh élec / m³ gaz)
CONSO_ANNUELLE_ELEC = {
    "Résidentiel": 5500,
    "Professionnel": 18000,
    "Tertiaire": 85000,
    "Industriel": 450000,
    "Collectivité": 120000,
}

CONSO_ANNUELLE_GAZ = {
    "Résidentiel": 1200,
    "Professionnel": 4500,
    "Tertiaire": 25000,
    "Industriel": 150000,
    "Collectivité": 35000,
}

# Échantillon de compteurs pour les consommations (performance)
sample_compteurs = compteurs_df.sample(n=min(8000, len(compteurs_df)), random_state=42)

consommations = []
mois_range = pd.date_range(DATE_DEBUT, DATE_FIN, freq='MS')

for _, cpt in sample_compteurs.iterrows():
    client = clients_df[clients_df["client_id"] == cpt["client_id"]].iloc[0]
    segment = client["segment"]
    energie = cpt["type_energie"]
    
    if energie == "Électricité":
        conso_base = CONSO_ANNUELLE_ELEC[segment] / 12
        profil = PROFIL_SAISONNIER_ELEC
    else:
        conso_base = CONSO_ANNUELLE_GAZ[segment] / 12
        profil = PROFIL_SAISONNIER_GAZ
    
    # Variabilité propre au client
    facteur_client = np.random.uniform(0.6, 1.5)
    conso_base *= facteur_client
    
    # Tendance annuelle (effet sobriété / transition)
    tendance_annuelle = np.random.uniform(-0.04, 0.01)  # Légère baisse tendancielle
    
    for mois in mois_range:
        annee_offset = (mois.year - DATE_DEBUT.year) + (mois.month - 1) / 12
        facteur_tendance = 1 + tendance_annuelle * annee_offset
        
        conso = conso_base * profil[mois.month] * facteur_tendance
        conso *= np.random.uniform(0.85, 1.15)  # Bruit aléatoire
        conso = max(0, round(conso, 2))
        
        # Calcul HP/HC pour élec
        if energie == "Électricité" and client["option_tarifaire"] == "HP/HC":
            ratio_hc = np.random.uniform(0.35, 0.50)
            conso_hc = round(conso * ratio_hc, 2)
            conso_hp = round(conso - conso_hc, 2)
        else:
            conso_hp = conso
            conso_hc = 0
        
        # Coût estimé
        if energie == "Électricité":
            tarif = TARIFS_ELEC.get(client["option_tarifaire"], 0.25)
            cout = round(conso * tarif, 2)
        else:
            cout = round(conso * 1.15, 2)  # Prix moyen gaz €/m³
        
        # Pic de puissance (kW)
        if energie == "Électricité":
            pic_puissance = round(
                client["puissance_souscrite_kva"] * np.random.uniform(0.4, 0.95), 2
            )
        else:
            pic_puissance = None
        
        consommations.append({
            "compteur_id": cpt["compteur_id"],
            "client_id": cpt["client_id"],
            "type_energie": energie,
            "mois": mois.strftime("%Y-%m-%d"),
            "consommation_kwh" if energie == "Électricité" else "consommation_m3": conso,
            "conso_hp": conso_hp if energie == "Électricité" else None,
            "conso_hc": conso_hc if energie == "Électricité" else None,
            "cout_estime_eur": cout,
            "pic_puissance_kw": pic_puissance,
            "temperature_moyenne_c": round(
                {1: 4.5, 2: 5.8, 3: 9.2, 4: 12.1, 5: 15.8, 6: 19.5,
                 7: 21.8, 8: 21.2, 9: 17.5, 10: 13.2, 11: 8.1, 12: 5.2
                }[mois.month] + np.random.uniform(-2, 2), 1
            ),
        })

consommations_df = pd.DataFrame(consommations)

# Harmoniser les colonnes
if "consommation_kwh" not in consommations_df.columns:
    consommations_df["consommation_kwh"] = None
if "consommation_m3" not in consommations_df.columns:
    consommations_df["consommation_m3"] = None

consommations_df.to_csv(f"{OUTPUT_DIR}/consommations.csv", index=False, sep=";")
print(f"   ✅ {len(consommations_df)} lignes de consommation générées")


# ============================================================================
# 4. TABLE INTERVENTIONS
# ============================================================================
print("\n[4/7] Génération de la table INTERVENTIONS...")

TYPES_INTERVENTION = {
    "Mise en service": 0.20,
    "Relève compteur": 0.18,
    "Changement compteur": 0.12,
    "Dépannage réseau": 0.15,
    "Coupure impayé": 0.05,
    "Rétablissement": 0.04,
    "Contrôle technique": 0.10,
    "Raccordement neuf": 0.06,
    "Modification puissance": 0.05,
    "Maintenance préventive": 0.05,
}

STATUTS_INTERVENTION = ["Planifiée", "En cours", "Terminée", "Annulée"]

N_INTERVENTIONS = 12000
dates_intervention = [
    DATE_DEBUT + timedelta(days=np.random.randint(0, (DATE_FIN - DATE_DEBUT).days))
    for _ in range(N_INTERVENTIONS)
]

types_int = np.random.choice(
    list(TYPES_INTERVENTION.keys()),
    size=N_INTERVENTIONS,
    p=list(TYPES_INTERVENTION.values())
)

clients_sample = np.random.choice(clients_ids, size=N_INTERVENTIONS)

durees = []
for t in types_int:
    if t in ["Mise en service", "Relève compteur", "Modification puissance"]:
        durees.append(round(np.random.uniform(0.25, 1.5), 2))
    elif t in ["Dépannage réseau", "Raccordement neuf"]:
        durees.append(round(np.random.uniform(1, 6), 2))
    elif t == "Changement compteur":
        durees.append(round(np.random.uniform(0.5, 2), 2))
    else:
        durees.append(round(np.random.uniform(0.25, 3), 2))

interventions_df = pd.DataFrame({
    "intervention_id": [f"INT-{i:06d}" for i in range(N_INTERVENTIONS)],
    "client_id": clients_sample,
    "type_intervention": types_int,
    "date_intervention": dates_intervention,
    "duree_heures": durees,
    "statut": np.random.choice(
        STATUTS_INTERVENTION, size=N_INTERVENTIONS,
        p=[0.08, 0.05, 0.82, 0.05]
    ),
    "technicien_id": [f"TECH-{np.random.randint(1, 45):03d}" for _ in range(N_INTERVENTIONS)],
    "commune": np.random.choice(communes_list, size=N_INTERVENTIONS, p=pop_weights),
    "cout_intervention_eur": [round(np.random.uniform(30, 800), 2) for _ in range(N_INTERVENTIONS)],
    "satisfaction_intervention": np.clip(
        np.random.normal(7.5, 1.8, N_INTERVENTIONS), 1, 10
    ).round(1),
})

interventions_df.to_csv(f"{OUTPUT_DIR}/interventions.csv", index=False, sep=";")
print(f"   ✅ {len(interventions_df)} interventions générées")


# ============================================================================
# 5. TABLE FACTURES
# ============================================================================
print("\n[5/7] Génération de la table FACTURES...")

N_FACTURES = 45000
factures_clients = np.random.choice(clients_ids, size=N_FACTURES)

montants = []
for i in range(N_FACTURES):
    seg = clients_df[clients_df["client_id"] == factures_clients[i]]["segment"].values[0]
    if seg == "Résidentiel":
        montants.append(round(np.random.uniform(35, 280), 2))
    elif seg == "Professionnel":
        montants.append(round(np.random.uniform(120, 1500), 2))
    elif seg == "Tertiaire":
        montants.append(round(np.random.uniform(500, 8000), 2))
    elif seg == "Industriel":
        montants.append(round(np.random.uniform(2000, 45000), 2))
    else:
        montants.append(round(np.random.uniform(800, 12000), 2))

dates_factures = [
    DATE_DEBUT + timedelta(days=np.random.randint(0, (DATE_FIN - DATE_DEBUT).days))
    for _ in range(N_FACTURES)
]

statuts_paiement = np.random.choice(
    ["Payée", "Payée", "Payée", "Payée", "En attente", "En retard", "Impayée"],
    size=N_FACTURES
)

# Délai de paiement
delais = []
for s in statuts_paiement:
    if s == "Payée":
        delais.append(np.random.randint(1, 35))
    elif s == "En attente":
        delais.append(None)
    elif s == "En retard":
        delais.append(np.random.randint(35, 90))
    else:
        delais.append(np.random.randint(90, 365))

factures_df = pd.DataFrame({
    "facture_id": [f"FAC-{i:06d}" for i in range(N_FACTURES)],
    "client_id": factures_clients,
    "date_facture": dates_factures,
    "date_echeance": [d + timedelta(days=30) for d in dates_factures],
    "montant_ht_eur": montants,
    "montant_ttc_eur": [round(m * 1.20, 2) for m in montants],
    "tva_eur": [round(m * 0.20, 2) for m in montants],
    "statut_paiement": statuts_paiement,
    "delai_paiement_jours": delais,
    "mode_paiement": np.random.choice(
        ["Prélèvement", "Virement", "CB en ligne", "Chèque", "TIP"],
        size=N_FACTURES, p=[0.55, 0.15, 0.15, 0.10, 0.05]
    ),
})

factures_df.to_csv(f"{OUTPUT_DIR}/factures.csv", index=False, sep=";")
print(f"   ✅ {len(factures_df)} factures générées")


# ============================================================================
# 6. TABLE PRODUCTION ENR (Énergies Renouvelables)
# ============================================================================
print("\n[6/7] Génération de la table PRODUCTION ENR...")

SITES_ENR = [
    {"site_id": "ENR-001", "nom": "Parc Éolien Neuville", "type": "Éolien", "commune": "Neuville-de-Poitou",
     "puissance_installee_mw": 12.0, "date_mise_service": "2019-06-15"},
    {"site_id": "ENR-002", "nom": "Centrale PV Jaunay", "type": "Solaire PV", "commune": "Jaunay-Marigny",
     "puissance_installee_mw": 8.5, "date_mise_service": "2020-03-01"},
    {"site_id": "ENR-003", "nom": "Parc Éolien Loudun", "type": "Éolien", "commune": "Loudun",
     "puissance_installee_mw": 18.0, "date_mise_service": "2018-11-20"},
    {"site_id": "ENR-004", "nom": "Centrale PV Civray", "type": "Solaire PV", "commune": "Civray",
     "puissance_installee_mw": 5.2, "date_mise_service": "2021-07-10"},
    {"site_id": "ENR-005", "nom": "Méthanisation Vivonne", "type": "Biogaz", "commune": "Vivonne",
     "puissance_installee_mw": 2.1, "date_mise_service": "2022-01-15"},
    {"site_id": "ENR-006", "nom": "Hydroélectrique Chauvigny", "type": "Hydraulique", "commune": "Chauvigny",
     "puissance_installee_mw": 3.8, "date_mise_service": "2015-09-01"},
    {"site_id": "ENR-007", "nom": "Toiture PV Chasseneuil", "type": "Solaire PV", "commune": "Chasseneuil-du-Poitou",
     "puissance_installee_mw": 1.2, "date_mise_service": "2023-04-01"},
    {"site_id": "ENR-008", "nom": "Parc Éolien Montmorillon", "type": "Éolien", "commune": "Montmorillon",
     "puissance_installee_mw": 15.0, "date_mise_service": "2020-08-12"},
]

# Profils de production par type et par mois (facteur de charge)
FC_EOLIEN = {1: 0.28, 2: 0.30, 3: 0.27, 4: 0.24, 5: 0.20, 6: 0.17,
             7: 0.15, 8: 0.14, 9: 0.19, 10: 0.25, 11: 0.29, 12: 0.30}
FC_SOLAIRE = {1: 0.06, 2: 0.08, 3: 0.12, 4: 0.15, 5: 0.18, 6: 0.20,
              7: 0.21, 8: 0.19, 9: 0.15, 10: 0.10, 11: 0.06, 12: 0.05}
FC_BIOGAZ = {m: np.random.uniform(0.80, 0.92) for m in range(1, 13)}
FC_HYDRAULIQUE = {1: 0.35, 2: 0.38, 3: 0.40, 4: 0.42, 5: 0.38, 6: 0.30,
                  7: 0.22, 8: 0.18, 9: 0.20, 10: 0.28, 11: 0.35, 12: 0.38}

FC_MAP = {
    "Éolien": FC_EOLIEN,
    "Solaire PV": FC_SOLAIRE,
    "Biogaz": FC_BIOGAZ,
    "Hydraulique": FC_HYDRAULIQUE,
}

production_enr = []
for site in SITES_ENR:
    fc_profil = FC_MAP[site["type"]]
    date_ms = datetime.strptime(site["date_mise_service"], "%Y-%m-%d")
    
    for mois in mois_range:
        if mois < date_ms:
            continue
        
        heures_mois = pd.Period(mois.strftime("%Y-%m"), "M").days_in_month * 24
        fc = fc_profil[mois.month] * np.random.uniform(0.85, 1.15)
        production_mwh = round(site["puissance_installee_mw"] * heures_mois * fc, 2)
        
        # CO2 évité (kg) - ~400g CO2/kWh mix moyen
        co2_evite_kg = round(production_mwh * 1000 * 0.4, 0)
        
        disponibilite = round(min(np.random.uniform(0.88, 0.99), 1.0), 3)
        
        production_enr.append({
            "site_id": site["site_id"],
            "nom_site": site["nom"],
            "type_production": site["type"],
            "commune": site["commune"],
            "puissance_installee_mw": site["puissance_installee_mw"],
            "mois": mois.strftime("%Y-%m-%d"),
            "production_mwh": production_mwh,
            "facteur_charge": round(fc, 3),
            "heures_fonctionnement": round(heures_mois * fc, 1),
            "co2_evite_tonnes": round(co2_evite_kg / 1000, 2),
            "disponibilite": disponibilite,
            "revenu_estime_eur": round(production_mwh * np.random.uniform(55, 95), 2),
        })

production_enr_df = pd.DataFrame(production_enr)
production_enr_df.to_csv(f"{OUTPUT_DIR}/production_enr.csv", index=False, sep=";")
print(f"   ✅ {len(production_enr_df)} lignes de production ENR générées")


# ============================================================================
# 7. TABLE QUALITE RESEAU
# ============================================================================
print("\n[7/7] Génération de la table QUALITÉ RÉSEAU...")

qualite_reseau = []
for commune in communes_list:
    for mois in mois_range:
        # Nombre de coupures (plus en hiver / tempêtes)
        base_coupures = 2 if mois.month in [1, 2, 11, 12] else 1
        n_coupures = max(0, np.random.poisson(base_coupures))
        
        # Durée moyenne coupure (minutes)
        duree_moy = round(np.random.exponential(45), 1) if n_coupures > 0 else 0
        
        # Temps moyen de coupure par client (SAIDI)
        saidi = round(n_coupures * duree_moy / max(np.random.uniform(0.5, 2), 1), 2)
        
        # Fréquence moyenne interruption (SAIFI)
        saifi = round(n_coupures * np.random.uniform(0.01, 0.05), 4)
        
        # Taux de tension conforme
        taux_tension = round(np.random.uniform(0.96, 1.0), 4)
        
        # Taux de réclamation
        taux_reclamation = round(np.random.uniform(0.001, 0.015), 4)
        
        qualite_reseau.append({
            "commune": commune,
            "mois": mois.strftime("%Y-%m-%d"),
            "nb_coupures": n_coupures,
            "duree_moyenne_coupure_min": duree_moy,
            "saidi_min": saidi,
            "saifi": saifi,
            "taux_tension_conforme": taux_tension,
            "taux_reclamation": taux_reclamation,
            "nb_clients_impactes": n_coupures * np.random.randint(10, 500) if n_coupures > 0 else 0,
            "cause_principale": np.random.choice(
                ["Intempéries", "Défaut matériel", "Travaux tiers", "Surcharge", "Végétation", "Aucune"],
                p=[0.25, 0.20, 0.15, 0.10, 0.10, 0.20]
            ) if n_coupures > 0 else "Aucune",
        })

qualite_reseau_df = pd.DataFrame(qualite_reseau)
qualite_reseau_df.to_csv(f"{OUTPUT_DIR}/qualite_reseau.csv", index=False, sep=";")
print(f"   ✅ {len(qualite_reseau_df)} lignes de qualité réseau générées")


# ============================================================================
# RÉSUMÉ
# ============================================================================
print("\n" + "=" * 70)
print("  GÉNÉRATION TERMINÉE - Résumé")
print("=" * 70)
print(f"  📁 Répertoire : {OUTPUT_DIR}/")
print(f"  👥 Clients     : {len(clients_df):>10,}")
print(f"  📊 Compteurs   : {len(compteurs_df):>10,}")
print(f"  ⚡ Conso       : {len(consommations_df):>10,}")
print(f"  🔧 Interventions: {len(interventions_df):>10,}")
print(f"  💶 Factures    : {len(factures_df):>10,}")
print(f"  🌿 Prod. ENR   : {len(production_enr_df):>10,}")
print(f"  📶 Qualité rés.: {len(qualite_reseau_df):>10,}")
print("=" * 70)