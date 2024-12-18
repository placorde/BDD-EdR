import geopandas as gpd
import pandas as pd
from matplotlib import *
import matplotlib as plot 
import numpy as np
import matplotlib.pyplot as plt
from datetime import * 
import pygwalker as pyg
from dotenv import load_dotenv
from dateutil import parser
import streamlit as st
import plotly.express as px
import streamlit_option_menu as option_menu 
from datetime import timedelta
from pygwalker.api.streamlit import StreamlitRenderer
import plotly.graph_objects as go
import re

#Head of the page 
st.set_page_config(page_title='EDR FUND BIG DATA', page_icon='EDMOND-DE-ROTSCHILD.png', layout='wide')

# Création de deux colonnes : 1/3 pour le logo, 2/3 pour le titre
col1, col2 = st.columns([1, 2])
col1.image('EDMOND-DE-ROTSCHILD.png', width=250)
col2.markdown(
        "<h1 style='text-align: center; margin-top: 20px;'>Analyse performance BIG DATA</h1>",
        unsafe_allow_html=True
    )
#load dataset

positions = pd.read_excel("Positions_BigData.xlsx") # Positions du fonds  
passif = pd.read_excel("Passif_Année_2_Albert School.xlsx") # Information sur la collecte des fonds 
bench_lines = pd.read_excel("Bench_Lignes.xlsx") # Composition mensuelle du benchmark 
caracteristiques_parts = pd.read_excel("Caracteristiques_Parts.xlsx") # En lien avec le banchmark 
fixing = pd.read_excel("Fixing.xlsx") # Taux de change des devises ; attention : nettoyage nécessaire 
frais = pd.read_excel("Frais.xlsx") # Frais payés par les fonds quotidiennement 
taux = pd.read_excel("Historique_Taux.xlsx") # Taux de change des devises ; attention : nettoyage nécessaire 
perf_bench = pd.read_excel("Perf_Bench.xlsx") # Performance du benchmark 
perf_fonds = pd.read_excel("Perf_Fonds.xlsx") # Performance du fonds 
indice_reference= pd.read_excel("USGG10YR Index.xlsx")
fixing_BDF = pd.read_excel("BDD_EDR_Taux_Change_EUR_USD_BDF.xlsx") # Indice de référence


# Charger les données

colonnes_conserver = [
    'Business Relationship', 
    'Fund', 
    'Share Type', 
    'Date', 
    'AUM (€)', 
    'Quantity', 
    'Net Inflows YTD (€)',
    'Net Inflows MTD (€)', 
    'BR Segmentation (Business Relationship) (Business Relationship)',
    'Reporting Line (Business Relationship) (Business Relationship)',
    'Business Country (Business Relationship) (Business Relationship)',
    'Asset Class (Fund) (EdRAM Product)'
]

passif_ok = passif[colonnes_conserver]

passif_ok = passif_ok[passif_ok['Fund'] == 'EdR Fund Big Data']

indice_reference_ok = indice_reference[['Quote Date', 'Quote Close']]

fixing_usd = fixing[fixing['PRICE_CURRENCY'] == 'USD']

fixing_BDF = pd.read_excel("BDD_EDR_Taux_Change_EUR_USD_BDF.xlsx")

# Supprimer les 4 premières lignes
fixing_BDF = fixing_BDF.iloc[4:]

# Renommer les colonnes
fixing_BDF.rename(columns={
    'Titre :': 'DATE',
    "Cours de change de l'euro contre dollar des Etats-Unis (USD) - fin de mois": 'COURS'
}, inplace=True)

# Supprimer la première ligne
fixing_bdf = fixing_BDF.iloc[1:]

fixing_BDF = fixing_BDF[fixing_BDF['DATE'].str.match(r'^\d{4}-\d{2}-\d{2}$', na=False)]

# Convertir la colonne 'DATE' en format datetime
fixing_BDF['DATE'] = pd.to_datetime(fixing_BDF['DATE'])

# Filtrer les dates entre septembre 2020 et septembre 2024
start_date = '2020-09-01'
end_date = '2024-09-30'
fixing_BDF = fixing_BDF[(fixing_BDF['DATE'] >= start_date) & (fixing_BDF['DATE'] <= end_date)]

# Convertir la colonne 'Quote Date' en format datetime
indice_reference_ok['Quote Date'] = pd.to_datetime(indice_reference_ok['Quote Date'])

# Filtrer les dates entre septembre 2020 et septembre 2024
start_date = '2020-09-01'
end_date = '2024-09-30'
indice_reference_ok = indice_reference_ok[(indice_reference_ok['Quote Date'] >= start_date) & (indice_reference_ok['Quote Date'] <= end_date)]

# Grouper par année et mois, et garder le dernier jour de chaque mois
indice_reference_ok = indice_reference_ok.groupby(indice_reference_ok['Quote Date'].dt.to_period('M')).tail(1).reset_index(drop=True)

# Fusion des deux DataFrames sur la date
indice_reference_merged = pd.merge(indice_reference_ok, fixing_BDF, left_on='Quote Date', right_on='DATE')

# Conversion des cours en EUR
indice_reference_merged['Quote Close (EUR)'] = indice_reference_merged['Quote Close'] / indice_reference_merged['COURS']

indice_reference_merged = indice_reference_merged[['Quote Date', 'Quote Close (EUR)']]

# Garder la valeur du mois précédent si nécessaire

# Générer toutes les dates mensuelles entre septembre 2020 et septembre 2024
all_months = pd.date_range(start="2020-09-30", end="2024-09-30", freq='M')

# Convertir en DataFrame pour fusionner
all_dates_df = pd.DataFrame({'Quote Date': all_months})

# Fusionner avec le DataFrame initial
indice_reference_merged_ok = pd.merge(all_dates_df, indice_reference_merged, on='Quote Date', how='left')

# Remplir les valeurs manquantes avec celles du mois précédent
indice_reference_merged_ok['Quote Close (EUR)'] = indice_reference_merged_ok['Quote Close (EUR)'].fillna(method='ffill')

# Convertir la colonne 'Quote Date' en type datetime
indice_reference_merged_ok['Quote Date'] = pd.to_datetime(indice_reference_merged_ok['Quote Date'])

# Créer une colonne MONTH qui extrait le mois (en texte)
indice_reference_merged_ok['DATE'] = indice_reference_merged_ok['Quote Date']

indice_reference_merged_ok = indice_reference_merged_ok.drop(columns=['Quote Date'])

indice_reference_merged_ok['DATE'] = indice_reference_merged_ok['DATE'].dt.strftime('%Y-%m')

# Grade 1 : Catégorisation par années

passif_ok['Date'] = pd.to_datetime(passif_ok['Date'])

# Liste des années et colonnes à conserver 
annees = [2020, 2021, 2022, 2023, 2024]
colonnes_conserver = [
    'Share Type',
    'Date', 
    'AUM (€)', 
    'Net Inflows YTD (€)', 
    'Net Inflows MTD (€)', 
    'BR Segmentation (Business Relationship) (Business Relationship)', 
    'Business Country (Business Relationship) (Business Relationship)'
]

# Dictionnaire pour stocker les dataframes par année
passif_annees = {}

for annee in annees:
    # 1. Création du DataFrame pour l'année donnée
    df_annee = passif_ok[passif_ok['Date'].dt.year == annee].copy()
    
    # 2. Ne garder que les colonnes souhaitées
    df_annee = df_annee[colonnes_conserver]
    
    # 3. Créer la colonne 'Month' à partir du mois de la colonne 'Date'
    df_annee['Month'] = df_annee['Date'].dt.month_name()
    
    # Stocker le dataframe dans un dictionnaire pour y accéder plus facilement
    passif_annees[annee] = df_annee


# Grade 2 : Catégorisation par BR Segmentation

# Dictionnaire pour stocker les DataFrames organisés
passif_segmentation = {}

# Identifier toutes les valeurs uniques de la colonne pour les 5 années
valeurs_segmentation = set()

for annee, df in passif_annees.items():
    valeurs_segmentation.update(df['BR Segmentation (Business Relationship) (Business Relationship)'].unique())

# Créer un mapping des valeurs de segmentation avec des clés numériques pour simplifier l'accès
valeurs_segmentation = sorted(list(valeurs_segmentation))
segmentation_mapping = {val: idx for idx, val in enumerate(valeurs_segmentation)}

# Afficher le mapping pour l'accès facile
print("Clés de segmentation :")
for val, idx in segmentation_mapping.items():
    print(f"{idx}: {val}")

# Créer des DataFrames séparés pour chaque combinaison année-valeur unique
for annee, df_annee in passif_annees.items():
    passif_segmentation[annee] = {}
    for valeur_segmentation, idx in segmentation_mapping.items():
        # Filtrer les lignes correspondant à la valeur actuelle
        df_segmentation = df_annee[
            df_annee['BR Segmentation (Business Relationship) (Business Relationship)'] == valeur_segmentation
        ].copy()
        # Ajouter le DataFrame au dictionnaire à deux niveaux
        passif_segmentation[annee][idx] = df_segmentation

# Etape 1 : Nettoyage du dataset perf fond

# Création d'un nouveau dataframe avec les colonnes sélectionnées
colonnes_a_conserver = [
    'SECURITY_NAME', 
    'PERFORMANCE_START_DATE', 
    'ANALYSIS_DATE_FULL', 
    'TWR_NET_CUMULATED', 
    'TWR_1MC', 
    'TWR_YTD', 
    'REPORTING_CURRENCY'
]
perf_fonds_reduit = perf_fonds[colonnes_a_conserver]

# Filtrer les lignes avec des valeurs spécifiques dans 'SECURITY_NAME'
valeurs_security_name = [
    'EdR Fund Big Data A - EUR', 'EdR Fund Big Data A - USD', 'EdR Fund Big Data A - CHF',
    'EdR Fund Big Data I - EUR', 'EdR Fund Big Data I - USD', 'EdR Fund Big Data K - EUR',
    'EdR Fund Big Data N - EUR', 'EdR Fund Big Data N - USD', 'EdR Fund Big Data N CHF',
    'EdR Fund Big Data R - EUR', 'EdR Fund Big Data R - USD', 'EdR Fund Big Data CR - USD',
    'EdR Fund Big Data CRD - USD', 'EdR Fund Big Data CR - EUR', 'EdR Fund Big Data CRD - EUR',
    'EdR Fund Big Data B - EUR', 'EdR Fund Big Data J - USD', 'EdR Fund Big Data N2 - EUR',
    'EdR Fund Big Data P - EUR', 'EdR Fund Big Data P - USD', 'EdR Fund Big Data CRM - EUR'
]
perf_fonds_ok = perf_fonds_reduit[perf_fonds_reduit['SECURITY_NAME'].isin(valeurs_security_name)]

perf_fonds_ok = perf_fonds_ok.dropna(subset=['TWR_1MC'])

perf_fonds_ok['ANALYSIS_DATE_FULL'] = pd.to_datetime(perf_fonds_ok['ANALYSIS_DATE_FULL'])

# Filtrer les lignes en gardant uniquement celles avec une date à partir de septembre 2020 (inclus)
date_seuil = pd.Timestamp('2020-09-01')
perf_fonds_ok = perf_fonds_ok[perf_fonds_ok['ANALYSIS_DATE_FULL'] >= date_seuil]


def filter_end_of_month_by_asset(df):
    # Ajouter une colonne avec l'année et le mois pour identifier les groupes
    df['YEAR_MONTH'] = df['ANALYSIS_DATE_FULL'].dt.to_period('M')
    
    # Grouper par type d'actif, année, et mois, puis choisir la date la plus proche du 30
    filtered_df = df.groupby(['SECURITY_NAME', 'YEAR_MONTH']).apply(
        lambda group: group[group['ANALYSIS_DATE_FULL'].dt.day.isin([30, 29, 28])]
        .sort_values(by='ANALYSIS_DATE_FULL', ascending=False)  # Trier pour privilégier le 30
        .head(1)  # Garder la date la plus proche du 30
    )
    
    # Enlever l'index supplémentaire créé par groupby
    filtered_df.reset_index(drop=True, inplace=True)
    # Supprimer la colonne temporaire YEAR_MONTH
    filtered_df.drop(columns=['YEAR_MONTH'], inplace=True)
    
    return filtered_df

# Appliquer le filtre
perf_fonds_ok = filter_end_of_month_by_asset(perf_fonds_ok)

# Filtrer les lignes qui dont la devise est EUR 
perf_fonds_ok = perf_fonds_ok[perf_fonds_ok['REPORTING_CURRENCY'] == 'EUR']

# Supprimer l'expression "- EUR", puis splitter la colonne 'SECURITY_NAME'
perf_fonds_ok['SECURITY_NAME'] = perf_fonds_ok['SECURITY_NAME'].str.replace(' - EUR', '', regex=False)

# Séparer la colonne 'SECURITY_NAME' en deux parties
perf_fonds_ok[['SECURITY_NAME', 'TYPE']] = perf_fonds_ok['SECURITY_NAME'].str.extract(r'^(.*) (\S+)$')

# Réorganiser les colonnes pour placer 'TYPE' juste après 'SECURITY_NAME'
colonnes_ordre = ['SECURITY_NAME', 'TYPE'] + [col for col in perf_fonds_ok.columns if col not in ['SECURITY_NAME', 'TYPE']]
perf_fonds_ok = perf_fonds_ok[colonnes_ordre]

colonnes_a_conserver1 = [
    'TYPE', 
    'ANALYSIS_DATE_FULL', 
    'TWR_1MC', 
]
perf_fonds_ok = perf_fonds_ok[colonnes_a_conserver1]

# Etape 2 : Catégorisation du dataset

# Créer un dataframe pour chaque année
perf_fonds_segmentation = {}

# Boucle pour filtrer les lignes selon l'année
for year in range(2020, 2025):  # Les années de 2020 à 2024 incluses
    df_annee = perf_fonds_ok[perf_fonds_ok['ANALYSIS_DATE_FULL'].dt.year == year].copy()
    
    # Étape 2 : Ajouter la colonne 'MONTH' avec le nom du mois en anglais
    df_annee['MONTH'] = df_annee['ANALYSIS_DATE_FULL'].dt.month_name()

    # Sauvegarde du dataframe pour chaque année dans un dictionnaire
    perf_fonds_segmentation[year] = df_annee

# Etape 1 : Nettoyage du dataset perf_bench

# Création d'un nouveau dataframe avec les colonnes sélectionnées
colonnes_a_conserver_bench = [
    'SECURITY_NAME', 
    'PERFORMANCE_START_DATE', 
    'ANALYSIS_DATE_FULL', 
    'TWR_NET_CUMULATED', 
    'TWR_1MC', 
    'TWR_YTD', 
    'REPORTING_CURRENCY'
]
perf_bench_reduit = perf_bench[colonnes_a_conserver_bench]

# Filtrer les lignes avec des valeurs spécifiques dans 'SECURITY_NAME'
valeurs_security_name_bench = [
    'EdR Fund Big Data A - EUR', 'EdR Fund Big Data A - USD', 'EdR Fund Big Data A - CHF',
    'EdR Fund Big Data I - EUR', 'EdR Fund Big Data I - USD', 'EdR Fund Big Data K - EUR',
    'EdR Fund Big Data N - EUR', 'EdR Fund Big Data N - USD', 'EdR Fund Big Data N CHF',
    'EdR Fund Big Data R - EUR', 'EdR Fund Big Data R - USD', 'EdR Fund Big Data CR - USD',
    'EdR Fund Big Data CRD - USD', 'EdR Fund Big Data CR - EUR', 'EdR Fund Big Data CRD - EUR',
    'EdR Fund Big Data B - EUR', 'EdR Fund Big Data J - USD', 'EdR Fund Big Data N2 - EUR',
    'EdR Fund Big Data P - EUR', 'EdR Fund Big Data P - USD', 'EdR Fund Big Data CRM - EUR'
]
perf_bench_ok = perf_bench_reduit[perf_bench_reduit['SECURITY_NAME'].isin(valeurs_security_name_bench)]

perf_bench_ok = perf_bench_ok.dropna(subset=['TWR_1MC'])

perf_bench_ok['ANALYSIS_DATE_FULL'] = pd.to_datetime(perf_bench_ok['ANALYSIS_DATE_FULL'])

# Filtrer les lignes en gardant uniquement celles avec une date à partir de septembre 2020 (inclus)
date_seuil = pd.Timestamp('2020-09-01')
perf_bench_ok = perf_bench_ok[perf_bench_ok['ANALYSIS_DATE_FULL'] >= date_seuil]

def filter_end_of_month_by_asset(df):
    # Ajouter une colonne avec l'année et le mois pour identifier les groupes
    df['YEAR_MONTH'] = df['ANALYSIS_DATE_FULL'].dt.to_period('M')
    
    # Grouper par type d'actif, année, et mois, puis choisir la date la plus proche du 30
    filtered_df = df.groupby(['SECURITY_NAME', 'YEAR_MONTH']).apply(
        lambda group: group[group['ANALYSIS_DATE_FULL'].dt.day.isin([30, 29, 28])]
        .sort_values(by='ANALYSIS_DATE_FULL', ascending=False)  # Trier pour privilégier le 30
        .head(1)  # Garder la date la plus proche du 30
    )
    
    # Enlever l'index supplémentaire créé par groupby
    filtered_df.reset_index(drop=True, inplace=True)
    # Supprimer la colonne temporaire YEAR_MONTH
    filtered_df.drop(columns=['YEAR_MONTH'], inplace=True)
    
    return filtered_df

# Appliquer le filtre
perf_bench_ok = filter_end_of_month_by_asset(perf_bench_ok)

# Permet de vérifier que tous les mois sont encore présents 

def obtenir_mois_2022(dataframe, colonne):
    # Filtrer les données de l'année 2022
    dataframe_2022 = dataframe[dataframe[colonne].dt.year == 2022]
    # Extraire les mois uniques
    mois_uniques = dataframe_2022[colonne].dt.month.unique()
    return sorted(mois_uniques)  # Trier pour une meilleure lisibilité

# Utilisation de la fonction
mois_2022 = obtenir_mois_2022(perf_bench_ok, 'ANALYSIS_DATE_FULL')

# Filtrer les lignes qui dont la devise est EUR 
perf_bench_ok = perf_bench_ok[perf_bench_ok['REPORTING_CURRENCY'] == 'EUR']

# Supprimer l'expression "- EUR", puis splitter la colonne 'SECURITY_NAME'
perf_bench_ok['SECURITY_NAME'] = perf_bench_ok['SECURITY_NAME'].str.replace(' - EUR', '', regex=False)

# Séparer la colonne 'SECURITY_NAME' en deux parties
perf_bench_ok[['SECURITY_NAME', 'TYPE']] = perf_bench_ok['SECURITY_NAME'].str.extract(r'^(.*) (\S+)$')

# Réorganiser les colonnes pour placer 'TYPE' juste après 'SECURITY_NAME'
colonnes_ordre = ['SECURITY_NAME', 'TYPE'] + [col for col in perf_bench_ok.columns if col not in ['SECURITY_NAME', 'TYPE']]
perf_bench_ok = perf_bench_ok[colonnes_ordre]

colonnes_a_conserver_bench1 = [
    'TYPE', 
    'ANALYSIS_DATE_FULL', 
    'TWR_1MC', 
]
perf_bench_ok = perf_bench_ok[colonnes_a_conserver_bench1]

# Etape 2 : Catégorisation du dataset

# Créer un dataframe pour chaque année
perf_bench_segmentation = {}

# Boucle pour filtrer les lignes selon l'année
for year in range(2020, 2025):  # Les années de 2020 à 2024 incluses
    df_annee_bench = perf_bench_ok[perf_bench_ok['ANALYSIS_DATE_FULL'].dt.year == year].copy()
    
    # Étape 2 : Ajouter la colonne 'MONTH' avec le nom du mois en anglais
    df_annee_bench['MONTH'] = df_annee_bench['ANALYSIS_DATE_FULL'].dt.month_name()

    # Sauvegarde du dataframe pour chaque année dans un dictionnaire
    perf_bench_segmentation[year] = df_annee_bench

# Renommer les colonnes 'TWR_1MC' et 'ANALYSIS_DATE_FULL' dans chaque dataframe
perf_fonds_ok.rename(columns={'TWR_1MC': 'R_M_FONDS', 'ANALYSIS_DATE_FULL': 'DATE'}, inplace=True)
perf_bench_ok.rename(columns={'TWR_1MC': 'R_M_BENCH', 'ANALYSIS_DATE_FULL': 'DATE'}, inplace=True)

# Convertir la colonne 'DATE' en datetime pour les deux dataframes
perf_fonds_ok['DATE'] = pd.to_datetime(perf_fonds_ok['DATE'])
perf_bench_ok['DATE'] = pd.to_datetime(perf_bench_ok['DATE'])

# Fusionner les deux dataframes sur 'TYPE' et 'DATE'
perf_all = pd.merge(
    perf_fonds_ok, 
    perf_bench_ok, 
    on=['TYPE', 'DATE'], 
    how='inner'  # Utilisation d'une jointure interne
)

# Ajouter les colonnes 'MONTH' et 'YEAR'
perf_all['MONTH'] = perf_all['DATE'].dt.strftime('%B')  # Mois écrit en toutes lettres
perf_all['YEAR'] = perf_all['DATE'].dt.year  # Année en chiffres

#Transformation de la colonne DATE en Mois et Année 
perf_all['DATE'] = perf_all['DATE'].dt.strftime('%Y-%m')

# Fusion du dataframe performance et du taux de l'indice de référence 

# Fusionner les deux dataframes à l’aide des colonnes ‘DATE’ et ‘Quote Date’
perf_all_merge = pd.merge(perf_all, indice_reference_merged_ok, on='DATE', how='right')

# Renommer la colonne 'Quote Close (EUR)’ en ‘INDICE_REFERENCE’
perf_all_merge.rename(columns={'Quote Close (EUR)': 'INDICE_REFERENCE'}, inplace=True)


perf_all_merge = perf_all_merge.drop(columns=["YEAR"])
perf_all_merge['DATE'] = pd.to_datetime(perf_all_merge['DATE'])
perf_all_merge['YEAR'] = perf_all_merge['DATE'].dt.year


# Etape 2 : Catégorisation du dataset 

# Affichage du dataframe final

# Liste des types d'actions
types_actions = ['A', 'B', 'CR', 'CRD', 'CRM', 'I', 'K', 'N', 'N2', 'P', 'R']

# Dictionnaire pour stocker les dataframes par type
perf_all_type = {}

# Création d'un dataframe pour chaque type
for type_action in types_actions:
    # Filtrer les lignes correspondant au type
    dataframe_type = perf_all_merge[perf_all_merge['TYPE'] == type_action]
    
    # Ajouter le dataframe au dictionnaire avec un nom dynamique
    perf_all_type[f'perf_{type_action}'] = dataframe_type
    
    # Si besoin, assigner dynamiquement le dataframe à une variable dans le namespace
    globals()[f'perf_{type_action}'] = dataframe_type  # Permet de créer des variables dynamiques

# Afficher les noms des dataframes créés pour vérification
print("Dataframes créés :", list(perf_all_type.keys()))
## RATIO ##

# Définir le taux sans risque (rendement annuel de l'obligation d'état 10 years USA en pourcentage)
risk_free_rate_annual = 4.401 
risk_free_rate_monthly = (1 + risk_free_rate_annual / 100) ** (1 / 12) - 1

# Fonction pour calculer les ratios financiers
def calculate_ratios(df):
    # 1. Volatilité mensuelle du fonds
    mean_fonds = df['R_M_FONDS'].mean()
    variance_fonds = ((df['R_M_FONDS'] - mean_fonds) ** 2)
    df['VOLATILITY_FONDS'] = np.sqrt(variance_fonds)
    
    # 2. Volatilité mensuelle du benchmark
    mean_bench = df['R_M_BENCH'].mean()
    variance_bench = ((df['R_M_BENCH'] - mean_bench) ** 2)
    df['VOLATILITY_BENCH'] = np.sqrt(variance_bench)
    
    # 3. Tracking error mensuel
    std_fonds = df['R_M_FONDS'].std()
    std_bench = df['R_M_BENCH'].std()
    df['TRACKING_ERROR'] = std_fonds - std_bench
    
    # 4. Ratio de Sharpe mensuel
    df['SHARPE_FONDS'] = (df['R_M_FONDS'] - df['INDICE_REFERENCE']) / std_fonds 
    df['SHARPE_BENCH'] = (df['R_M_BENCH'] - df['INDICE_REFERENCE']) / std_bench

    # 5. Ratio d'information mensuel
    df['INFO_RATIO'] = (df['R_M_FONDS'] - df['R_M_BENCH']) / (df['VOLATILITY_FONDS'] - df['VOLATILITY_BENCH'])
    
    # 6. Alpha
    covariance = np.cov(df['R_M_FONDS'], df['R_M_BENCH'])[0, 1]
    variance_bench_full = np.var(df['R_M_BENCH'])
    beta = covariance / variance_bench_full
    df['ALPHA'] = df['R_M_FONDS'] - df['INDICE_REFERENCE'] - beta * (df['R_M_BENCH'] - df['INDICE_REFERENCE']) 
    
    return df

# Appliquer la fonction sur chaque dataframe par type
for type_action in types_actions:
    globals()[f'perf_{type_action}'] = calculate_ratios(globals()[f'perf_{type_action}'])


# Liste des dataframes créés précédemment
dataframes_list = [globals()[f'perf_{type_action}'] for type_action in types_actions]

# Concaténer tous les dataframes en un seul
perf_all_ratios = pd.concat(dataframes_list, axis=0, ignore_index=True)


#Graphe année glissante
ratios = ["VOLATILITY_FONDS", "VOLATILITY_BENCH","SHARPE_FONDS" "SHARPE_BENCH", "INFO_RATIO", "ALPHA"]


# Créer un DataFrame pour chaque ratio avec la moyenne par 'DATE'
def create_ratio_df(df, ratio):
    """
    Crée un DataFrame pour un ratio donné avec la moyenne par 'DATE'.
    """
    return df.groupby('DATE')[ratio].mean().reset_index()

# Création des DataFrames pour chaque ratio et attribution à des variables explicites
VOLATILITY_FONDS_df = create_ratio_df(perf_all_ratios, "VOLATILITY_FONDS")
VOLATILITY_BENCH_df = create_ratio_df(perf_all_ratios, "VOLATILITY_BENCH")
TRACKING_ERROR_df = create_ratio_df(perf_all_ratios, "TRACKING_ERROR")
INFO_RATIO_df = create_ratio_df(perf_all_ratios, "INFO_RATIO")
ALPHA_df = create_ratio_df(perf_all_ratios, "ALPHA")
SHARPE_FONDS_df = create_ratio_df(perf_all_ratios, "SHARPE_FONDS")
SHARPE_BENCH_df = create_ratio_df(perf_all_ratios, "SHARPE_BENCH")

#ALPHA
# Conversion de la colonne DATE en datetime
ALPHA_df['DATE'] = pd.to_datetime(ALPHA_df['DATE'])

# Création d'une colonne pour les mois (format abrégé)
ALPHA_df['Mois'] = ALPHA_df['DATE'].dt.strftime('%B')

# Étape 1 : Trier les données par date
ALPHA_df = ALPHA_df.sort_values(by='DATE').reset_index(drop=True)

# Étape 2 : Définir les plages de 4 années glissantes
start_date = ALPHA_df['DATE'].min()  # Date de départ
gliding_periods = []  # Liste pour stocker chaque période d'un an

for i in range(4):  # 4 années glissantes
    end_date = start_date + pd.DateOffset(years=1) - pd.DateOffset(days=1)
    mask = (ALPHA_df['DATE'] >= start_date) & (ALPHA_df['DATE'] <= end_date)
    gliding_periods.append(ALPHA_df.loc[mask, ['Mois', 'ALPHA']])
    start_date = start_date + pd.DateOffset(years=1)  # Passer à l'année suivante

# Étape 3 : Fusionner les périodes dans un DataFrame final
# On crée un DataFrame de base avec les mois (de Septembre à Août)
mois_order = pd.date_range('2023-09-01', periods=12, freq='MS').strftime('%B')
final_df = pd.DataFrame({'Mois': mois_order})

# Ajouter les colonnes des 4 années glissantes
for i, period in enumerate(gliding_periods):
    period = period.groupby('Mois')['ALPHA'].mean().reset_index()  # Moyenne pour les doublons
    period = period.set_index('Mois').reindex(mois_order).reset_index()  # Reordonner selon les mois
    final_df[f"Année_{i+1}"] = period['ALPHA']

# Étape 4 : Structurer le DataFrame pour Streamlit

# Étape 1 : Moyenne des 4 années glissantes
final_df['Moyenne'] = final_df[['Année_1', 'Année_2', 'Année_3', 'Année_4']].mean(axis=1)
result_df_4 = final_df[['Mois', 'Moyenne']].rename(columns={'Moyenne': 'Moyenne_4_Ans'})

# Étape 2 : Moyenne des deux dernières années glissantes (Année_3 et Année_4)
last_two_years_df = final_df[['Mois', 'Année_3', 'Année_4']]
last_two_years_df['Moyenne'] = last_two_years_df[['Année_3', 'Année_4']].mean(axis=1)
result_df_2 = last_two_years_df[['Mois', 'Moyenne']].rename(columns={'Moyenne': 'Moyenne_2_Ans'})

# Étape 3 : Moyenne des deux dernières années glissantes (duplicat pour cohérence)
result_df_1 = last_two_years_df[["Mois",'Année_4']].rename(columns={'Année_4': 'Moyenne_1_Ans'})

# Étape 4 : Fusion des DataFrames sur la colonne "Mois"
merged_df_ALPHA_df = result_df_4.merge(result_df_2, on='Mois').merge(result_df_1, on='Mois')

merged_df_ALPHA_df = merged_df_ALPHA_df[['Mois', 'Moyenne_1_Ans', 'Moyenne_2_Ans', 'Moyenne_4_Ans']]

mean_values = merged_df_ALPHA_df[['Moyenne_1_Ans', 'Moyenne_2_Ans', 'Moyenne_4_Ans']].mean()

df_ALPHA_df_Glissant = pd.DataFrame({
    'Ratio': ['ALPHA'],
    'Moyenne_1_Ans': [mean_values['Moyenne_1_Ans']],
    'Moyenne_2_Ans': [mean_values['Moyenne_2_Ans']],
    'Moyenne_4_Ans': [mean_values['Moyenne_4_Ans']]
})


#VOLATILITY_BENCH
# Conversion de la colonne DATE en datetime
VOLATILITY_BENCH_df['DATE'] = pd.to_datetime(VOLATILITY_BENCH_df['DATE'])

# Création d'une colonne pour les mois (format abrégé)
VOLATILITY_BENCH_df['Mois'] = VOLATILITY_BENCH_df['DATE'].dt.strftime('%B')

# Étape 1 : Trier les données par date
VOLATILITY_BENCH_df = VOLATILITY_BENCH_df.sort_values(by='DATE').reset_index(drop=True)

# Étape 2 : Définir les plages de 4 années glissantes
start_date = VOLATILITY_BENCH_df['DATE'].min()  # Date de départ
gliding_periods = []  # Liste pour stocker chaque période d'un an

for i in range(4):  # 4 années glissantes
    end_date = start_date + pd.DateOffset(years=1) - pd.DateOffset(days=1)
    mask = (VOLATILITY_BENCH_df['DATE'] >= start_date) & (VOLATILITY_BENCH_df['DATE'] <= end_date)
    gliding_periods.append(VOLATILITY_BENCH_df.loc[mask, ['Mois', 'VOLATILITY_BENCH']])
    start_date = start_date + pd.DateOffset(years=1)  # Passer à l'année suivante

# Étape 3 : Fusionner les périodes dans un DataFrame final
# On crée un DataFrame de base avec les mois (de Septembre à Août)
mois_order = pd.date_range('2023-09-01', periods=12, freq='MS').strftime('%B')
final_df = pd.DataFrame({'Mois': mois_order})

# Ajouter les colonnes des 4 années glissantes
for i, period in enumerate(gliding_periods):
    period = period.groupby('Mois')['VOLATILITY_BENCH'].mean().reset_index()  # Moyenne pour les doublons
    period = period.set_index('Mois').reindex(mois_order).reset_index()  # Reordonner selon les mois
    final_df[f"Année_{i+1}"] = period['VOLATILITY_BENCH']

# Étape 4 : Structurer le DataFrame pour Streamlit

# Étape 1 : Moyenne des 4 années glissantes
final_df['Moyenne'] = final_df[['Année_1', 'Année_2', 'Année_3', 'Année_4']].mean(axis=1)
result_df_4 = final_df[['Mois', 'Moyenne']].rename(columns={'Moyenne': 'Moyenne_4_Ans'})

# Étape 2 : Moyenne des deux dernières années glissantes (Année_3 et Année_4)
last_two_years_df = final_df[['Mois', 'Année_3', 'Année_4']]
last_two_years_df['Moyenne'] = last_two_years_df[['Année_3', 'Année_4']].mean(axis=1)
result_df_2 = last_two_years_df[['Mois', 'Moyenne']].rename(columns={'Moyenne': 'Moyenne_2_Ans'})

# Étape 3 : Moyenne des deux dernières années glissantes (duplicat pour cohérence)
result_df_1 = last_two_years_df[["Mois",'Année_4']].rename(columns={'Année_4': 'Moyenne_1_Ans'})

# Étape 4 : Fusion des DataFrames sur la colonne "Mois"
merged_df_VOLATILITY_BENCH_df = result_df_4.merge(result_df_2, on='Mois').merge(result_df_1, on='Mois')
merged_df_VOLATILITY_BENCH_df = merged_df_VOLATILITY_BENCH_df[['Mois', 'Moyenne_1_Ans', 'Moyenne_2_Ans', 'Moyenne_4_Ans']]

mean_values = merged_df_VOLATILITY_BENCH_df[['Moyenne_1_Ans', 'Moyenne_2_Ans', 'Moyenne_4_Ans']].mean()

VOLATILITY_BENCH_df_Glissant = pd.DataFrame({
    'Ratio': ['VOLATILITY_BENCH'],
    'Moyenne_1_Ans': [mean_values['Moyenne_1_Ans']],
    'Moyenne_2_Ans': [mean_values['Moyenne_2_Ans']],
    'Moyenne_4_Ans': [mean_values['Moyenne_4_Ans']]
})


#VOLATILITY_FONDS 
# Conversion de la colonne DATE en datetime
VOLATILITY_FONDS_df['DATE'] = pd.to_datetime(VOLATILITY_FONDS_df['DATE'])

# Création d'une colonne pour les mois (format abrégé)
VOLATILITY_FONDS_df['Mois'] = VOLATILITY_FONDS_df['DATE'].dt.strftime('%B')

# Étape 1 : Trier les données par date
VOLATILITY_FONDS_df = VOLATILITY_FONDS_df.sort_values(by='DATE').reset_index(drop=True)

# Étape 2 : Définir les plages de 4 années glissantes
start_date = VOLATILITY_FONDS_df['DATE'].min()  # Date de départ
gliding_periods = []  # Liste pour stocker chaque période d'un an

for i in range(4):  # 4 années glissantes
    end_date = start_date + pd.DateOffset(years=1) - pd.DateOffset(days=1)
    mask = (VOLATILITY_FONDS_df['DATE'] >= start_date) & (VOLATILITY_FONDS_df['DATE'] <= end_date)
    gliding_periods.append(VOLATILITY_FONDS_df.loc[mask, ['Mois', 'VOLATILITY_FONDS']])
    start_date = start_date + pd.DateOffset(years=1)  # Passer à l'année suivante

# Étape 3 : Fusionner les périodes dans un DataFrame final
# On crée un DataFrame de base avec les mois (de Septembre à Août)
mois_order = pd.date_range('2023-09-01', periods=12, freq='MS').strftime('%B')
final_df = pd.DataFrame({'Mois': mois_order})

# Ajouter les colonnes des 4 années glissantes
for i, period in enumerate(gliding_periods):
    period = period.groupby('Mois')['VOLATILITY_FONDS'].mean().reset_index()  # Moyenne pour les doublons
    period = period.set_index('Mois').reindex(mois_order).reset_index()  # Reordonner selon les mois
    final_df[f"Année_{i+1}"] = period['VOLATILITY_FONDS']

# Étape 4 : Structurer le DataFrame pour Streamlit

# Étape 1 : Moyenne des 4 années glissantes
final_df['Moyenne'] = final_df[['Année_1', 'Année_2', 'Année_3', 'Année_4']].mean(axis=1)
result_df_4 = final_df[['Mois', 'Moyenne']].rename(columns={'Moyenne': 'Moyenne_4_Ans'})

# Étape 2 : Moyenne des deux dernières années glissantes (Année_3 et Année_4)
last_two_years_df = final_df[['Mois', 'Année_3', 'Année_4']]
last_two_years_df['Moyenne'] = last_two_years_df[['Année_3', 'Année_4']].mean(axis=1)
result_df_2 = last_two_years_df[['Mois', 'Moyenne']].rename(columns={'Moyenne': 'Moyenne_2_Ans'})

# Étape 3 : Moyenne des deux dernières années glissantes (duplicat pour cohérence)
result_df_1 = last_two_years_df[["Mois",'Année_4']].rename(columns={'Année_4': 'Moyenne_1_Ans'})

# Étape 4 : Fusion des DataFrames sur la colonne "Mois"
merged_df_VOLATILITY_FONDS_df = result_df_4.merge(result_df_2, on='Mois').merge(result_df_1, on='Mois')

merged_df_VOLATILITY_FONDS_df = merged_df_VOLATILITY_FONDS_df[['Mois', 'Moyenne_1_Ans', 'Moyenne_2_Ans', 'Moyenne_4_Ans']]

# Afficher le résultat final
mean_values = merged_df_VOLATILITY_FONDS_df[['Moyenne_1_Ans', 'Moyenne_2_Ans', 'Moyenne_4_Ans']].mean()

VOLATILITY_FONDS_df_Glissant = pd.DataFrame({
    'Ratio': ['VOLATILITY_FONDS'],
    'Moyenne_1_Ans': [mean_values['Moyenne_1_Ans']],
    'Moyenne_2_Ans': [mean_values['Moyenne_2_Ans']],
    'Moyenne_4_Ans': [mean_values['Moyenne_4_Ans']]
})


#SHARPE_FONDS_df
# Conversion de la colonne DATE en datetime
SHARPE_FONDS_df['DATE'] = pd.to_datetime(SHARPE_FONDS_df['DATE'])

# Création d'une colonne pour les mois (format abrégé)
SHARPE_FONDS_df['Mois'] = SHARPE_FONDS_df['DATE'].dt.strftime('%B')

# Étape 1 : Trier les données par date
SHARPE_FONDS_df = SHARPE_FONDS_df.sort_values(by='DATE').reset_index(drop=True)

# Étape 2 : Définir les plages de 4 années glissantes
start_date = SHARPE_FONDS_df['DATE'].min()  # Date de départ
gliding_periods = []  # Liste pour stocker chaque période d'un an

for i in range(4):  # 4 années glissantes
    end_date = start_date + pd.DateOffset(years=1) - pd.DateOffset(days=1)
    mask = (SHARPE_FONDS_df['DATE'] >= start_date) & (SHARPE_FONDS_df['DATE'] <= end_date)
    gliding_periods.append(SHARPE_FONDS_df.loc[mask, ['Mois', 'SHARPE_FONDS']])
    start_date = start_date + pd.DateOffset(years=1)  # Passer à l'année suivante

# Étape 3 : Fusionner les périodes dans un DataFrame final
# On crée un DataFrame de base avec les mois (de Septembre à Août)
mois_order = pd.date_range('2023-09-01', periods=12, freq='MS').strftime('%B')
final_df = pd.DataFrame({'Mois': mois_order})

# Ajouter les colonnes des 4 années glissantes
for i, period in enumerate(gliding_periods):
    period = period.groupby('Mois')['SHARPE_FONDS'].mean().reset_index()  # Moyenne pour les doublons
    period = period.set_index('Mois').reindex(mois_order).reset_index()  # Reordonner selon les mois
    final_df[f"Année_{i+1}"] = period['SHARPE_FONDS']

# Étape 4 : Structurer le DataFrame pour Streamlit

# Étape 1 : Moyenne des 4 années glissantes
final_df['Moyenne'] = final_df[['Année_1', 'Année_2', 'Année_3', 'Année_4']].mean(axis=1)
result_df_4 = final_df[['Mois', 'Moyenne']].rename(columns={'Moyenne': 'Moyenne_4_Ans'})

# Étape 2 : Moyenne des deux dernières années glissantes (Année_3 et Année_4)
last_two_years_df = final_df[['Mois', 'Année_3', 'Année_4']]
last_two_years_df['Moyenne'] = last_two_years_df[['Année_3', 'Année_4']].mean(axis=1)
result_df_2 = last_two_years_df[['Mois', 'Moyenne']].rename(columns={'Moyenne': 'Moyenne_2_Ans'})

# Étape 3 : Moyenne des deux dernières années glissantes (duplicat pour cohérence)
result_df_1 = last_two_years_df[["Mois",'Année_4']].rename(columns={'Année_4': 'Moyenne_1_Ans'})

# Étape 4 : Fusion des DataFrames sur la colonne "Mois"
merged_df_SHARPE_FONDS_df = result_df_4.merge(result_df_2, on='Mois').merge(result_df_1, on='Mois')

merged_df_SHARPE_FONDS_df = merged_df_SHARPE_FONDS_df[['Mois', 'Moyenne_1_Ans', 'Moyenne_2_Ans', 'Moyenne_4_Ans']]

mean_values = merged_df_SHARPE_FONDS_df[['Moyenne_1_Ans', 'Moyenne_2_Ans', 'Moyenne_4_Ans']].mean()

SHARPE_FONDS_df_Glissant = pd.DataFrame({
    'Ratio': ['SHARPE_FONDS_'],
    'Moyenne_1_Ans': [mean_values['Moyenne_1_Ans']],
    'Moyenne_2_Ans': [mean_values['Moyenne_2_Ans']],
    'Moyenne_4_Ans': [mean_values['Moyenne_4_Ans']]
})



#SHARPE_BENCH_df
# Conversion de la colonne DATE en datetime
SHARPE_BENCH_df['DATE'] = pd.to_datetime(SHARPE_BENCH_df['DATE'])

# Création d'une colonne pour les mois (format abrégé)
SHARPE_BENCH_df['Mois'] = SHARPE_BENCH_df['DATE'].dt.strftime('%B')

# Étape 1 : Trier les données par date
SHARPE_BENCH_df = SHARPE_BENCH_df.sort_values(by='DATE').reset_index(drop=True)

# Étape 2 : Définir les plages de 4 années glissantes
start_date = SHARPE_BENCH_df['DATE'].min()  # Date de départ
gliding_periods = []  # Liste pour stocker chaque période d'un an

for i in range(4):  # 4 années glissantes
    end_date = start_date + pd.DateOffset(years=1) - pd.DateOffset(days=1)
    mask = (SHARPE_BENCH_df['DATE'] >= start_date) & (SHARPE_BENCH_df['DATE'] <= end_date)
    gliding_periods.append(SHARPE_BENCH_df.loc[mask, ['Mois', 'SHARPE_BENCH']])
    start_date = start_date + pd.DateOffset(years=1)  # Passer à l'année suivante

# Étape 3 : Fusionner les périodes dans un DataFrame final
# On crée un DataFrame de base avec les mois (de Septembre à Août)
mois_order = pd.date_range('2023-09-01', periods=12, freq='MS').strftime('%B')
final_df = pd.DataFrame({'Mois': mois_order})

# Ajouter les colonnes des 4 années glissantes
for i, period in enumerate(gliding_periods):
    period = period.groupby('Mois')['SHARPE_BENCH'].mean().reset_index()  # Moyenne pour les doublons
    period = period.set_index('Mois').reindex(mois_order).reset_index()  # Reordonner selon les mois
    final_df[f"Année_{i+1}"] = period['SHARPE_BENCH']

# Étape 4 : Structurer le DataFrame pour Streamlit

# Étape 1 : Moyenne des 4 années glissantes
final_df['Moyenne'] = final_df[['Année_1', 'Année_2', 'Année_3', 'Année_4']].mean(axis=1)
result_df_4 = final_df[['Mois', 'Moyenne']].rename(columns={'Moyenne': 'Moyenne_4_Ans'})

# Étape 2 : Moyenne des deux dernières années glissantes (Année_3 et Année_4)
last_two_years_df = final_df[['Mois', 'Année_3', 'Année_4']]
last_two_years_df['Moyenne'] = last_two_years_df[['Année_3', 'Année_4']].mean(axis=1)
result_df_2 = last_two_years_df[['Mois', 'Moyenne']].rename(columns={'Moyenne': 'Moyenne_2_Ans'})

# Étape 3 : Moyenne des deux dernières années glissantes (duplicat pour cohérence)
result_df_1 = last_two_years_df[["Mois",'Année_4']].rename(columns={'Année_4': 'Moyenne_1_Ans'})

# Étape 4 : Fusion des DataFrames sur la colonne "Mois"
merged_df_SHARPE_BENCH_df = result_df_4.merge(result_df_2, on='Mois').merge(result_df_1, on='Mois')

merged_df_SHARPE_BENCH_df = merged_df_SHARPE_BENCH_df[['Mois', 'Moyenne_1_Ans', 'Moyenne_2_Ans', 'Moyenne_4_Ans']]

mean_values = merged_df_SHARPE_BENCH_df[['Moyenne_1_Ans', 'Moyenne_2_Ans', 'Moyenne_4_Ans']].mean()

SHARPE_BENCH_df_Glissant = pd.DataFrame({
    'Ratio': ['SHARPE_BENCH'],
    'Moyenne_1_Ans': [mean_values['Moyenne_1_Ans']],
    'Moyenne_2_Ans': [mean_values['Moyenne_2_Ans']],
    'Moyenne_4_Ans': [mean_values['Moyenne_4_Ans']]
})



#INFO_RATIO_df_Glissant
# Conversion de la colonne DATE en datetime
INFO_RATIO_df['DATE'] = pd.to_datetime(INFO_RATIO_df['DATE'])

# Création d'une colonne pour les mois (format abrégé)
INFO_RATIO_df['Mois'] = INFO_RATIO_df['DATE'].dt.strftime('%B')

# Étape 1 : Trier les données par date
INFO_RATIO_df = INFO_RATIO_df.sort_values(by='DATE').reset_index(drop=True)

# Étape 2 : Définir les plages de 4 années glissantes
start_date = INFO_RATIO_df['DATE'].min()  # Date de départ
gliding_periods = []  # Liste pour stocker chaque période d'un an

for i in range(4):  # 4 années glissantes
    end_date = start_date + pd.DateOffset(years=1) - pd.DateOffset(days=1)
    mask = (INFO_RATIO_df['DATE'] >= start_date) & (INFO_RATIO_df['DATE'] <= end_date)
    gliding_periods.append(INFO_RATIO_df.loc[mask, ['Mois', 'INFO_RATIO']])
    start_date = start_date + pd.DateOffset(years=1)  # Passer à l'année suivante

# Étape 3 : Fusionner les périodes dans un DataFrame final
# On crée un DataFrame de base avec les mois (de Septembre à Août)
mois_order = pd.date_range('2023-09-01', periods=12, freq='MS').strftime('%B')
final_df = pd.DataFrame({'Mois': mois_order})

# Ajouter les colonnes des 4 années glissantes
for i, period in enumerate(gliding_periods):
    period = period.groupby('Mois')['INFO_RATIO'].mean().reset_index()  # Moyenne pour les doublons
    period = period.set_index('Mois').reindex(mois_order).reset_index()  # Reordonner selon les mois
    final_df[f"Année_{i+1}"] = period['INFO_RATIO']

# Étape 4 : Structurer le DataFrame pour Streamlit

# Étape 1 : Moyenne des 4 années glissantes
final_df['Moyenne'] = final_df[['Année_1', 'Année_2', 'Année_3', 'Année_4']].mean(axis=1)
result_df_4 = final_df[['Mois', 'Moyenne']].rename(columns={'Moyenne': 'Moyenne_4_Ans'})

# Étape 2 : Moyenne des deux dernières années glissantes (Année_3 et Année_4)
last_two_years_df = final_df[['Mois', 'Année_3', 'Année_4']]
last_two_years_df['Moyenne'] = last_two_years_df[['Année_3', 'Année_4']].mean(axis=1)
result_df_2 = last_two_years_df[['Mois', 'Moyenne']].rename(columns={'Moyenne': 'Moyenne_2_Ans'})

# Étape 3 : Moyenne des deux dernières années glissantes (duplicat pour cohérence)
result_df_1 = last_two_years_df[["Mois",'Année_4']].rename(columns={'Année_4': 'Moyenne_1_Ans'})

# Étape 4 : Fusion des DataFrames sur la colonne "Mois"
merged_df_INFO_RATIO_df= result_df_4.merge(result_df_2, on='Mois').merge(result_df_1, on='Mois')

merged_df_INFO_RATIO_df = merged_df_INFO_RATIO_df[['Mois', 'Moyenne_1_Ans', 'Moyenne_2_Ans', 'Moyenne_4_Ans']]


mean_values = merged_df_INFO_RATIO_df[['Moyenne_1_Ans', 'Moyenne_2_Ans', 'Moyenne_4_Ans']].mean()

INFO_RATIO_df_Glissant = pd.DataFrame({
    'Ratio': ['INFO_RATIO'],
    'Moyenne_1_Ans': [mean_values['Moyenne_1_Ans']],
    'Moyenne_2_Ans': [mean_values['Moyenne_2_Ans']],
    'Moyenne_4_Ans': [mean_values['Moyenne_4_Ans']]
})

# Affichage du nouveau DataFrame avec les moyennes


#TRACKING ERROR
# Conversion de la colonne DATE en datetime
TRACKING_ERROR_df['DATE'] = pd.to_datetime(TRACKING_ERROR_df['DATE'])

# Création d'une colonne pour les mois (format abrégé)
TRACKING_ERROR_df['Mois'] = TRACKING_ERROR_df['DATE'].dt.strftime('%B')

# Étape 1 : Trier les données par date
TRACKING_ERROR_df = TRACKING_ERROR_df.sort_values(by='DATE').reset_index(drop=True)

# Étape 2 : Définir les plages de 4 années glissantes
start_date = TRACKING_ERROR_df['DATE'].min()  # Date de départ
gliding_periods = []  # Liste pour stocker chaque période d'un an

for i in range(4):  # 4 années glissantes
    end_date = start_date + pd.DateOffset(years=1) - pd.DateOffset(days=1)
    mask = (TRACKING_ERROR_df['DATE'] >= start_date) & (TRACKING_ERROR_df['DATE'] <= end_date)
    gliding_periods.append(TRACKING_ERROR_df.loc[mask, ['Mois', 'TRACKING_ERROR']])
    start_date = start_date + pd.DateOffset(years=1)  # Passer à l'année suivante

# Étape 3 : Fusionner les périodes dans un DataFrame final
# On crée un DataFrame de base avec les mois (de Septembre à Août)
mois_order = pd.date_range('2023-09-01', periods=12, freq='MS').strftime('%B')
final_df = pd.DataFrame({'Mois': mois_order})

# Ajouter les colonnes des 4 années glissantes
for i, period in enumerate(gliding_periods):
    period = period.groupby('Mois')['TRACKING_ERROR'].mean().reset_index()  # Moyenne pour les doublons
    period = period.set_index('Mois').reindex(mois_order).reset_index()  # Reordonner selon les mois
    final_df[f"Année_{i+1}"] = period['TRACKING_ERROR']

# Étape 4 : Structurer le DataFrame pour Streamlit

# Étape 1 : Moyenne des 4 années glissantes
final_df['Moyenne'] = final_df[['Année_1', 'Année_2', 'Année_3', 'Année_4']].mean(axis=1)
result_df_4 = final_df[['Mois', 'Moyenne']].rename(columns={'Moyenne': 'Moyenne_4_Ans'})

# Étape 2 : Moyenne des deux dernières années glissantes (Année_3 et Année_4)
last_two_years_df = final_df[['Mois', 'Année_3', 'Année_4']]
last_two_years_df['Moyenne'] = last_two_years_df[['Année_3', 'Année_4']].mean(axis=1)
result_df_2 = last_two_years_df[['Mois', 'Moyenne']].rename(columns={'Moyenne': 'Moyenne_2_Ans'})

# Étape 3 : Moyenne des deux dernières années glissantes (duplicat pour cohérence)
result_df_1 = last_two_years_df[["Mois",'Année_4']].rename(columns={'Année_4': 'Moyenne_1_Ans'})

# Étape 4 : Fusion des DataFrames sur la colonne "Mois"
merged_df_TRACKING_ERROR_df= result_df_4.merge(result_df_2, on='Mois').merge(result_df_1, on='Mois')

merged_df_TRACKING_ERROR_df = merged_df_TRACKING_ERROR_df[['Mois', 'Moyenne_1_Ans', 'Moyenne_2_Ans', 'Moyenne_4_Ans']]

# Afficher le résultat final
mean_values = merged_df_TRACKING_ERROR_df[['Moyenne_1_Ans', 'Moyenne_2_Ans', 'Moyenne_4_Ans']].mean()

TRACKING_ERROR_df_Glissant = pd.DataFrame({
    'Ratio': ['TRACKING_ERROR'],
    'Moyenne_1_Ans': [mean_values['Moyenne_1_Ans']],
    'Moyenne_2_Ans': [mean_values['Moyenne_2_Ans']],
    'Moyenne_4_Ans': [mean_values['Moyenne_4_Ans']]
})

# Affichage du nouveau DataFrame avec les moyennes




#ANALYSE DF_PASSIF
# Tablaux avec pourcentages pour déterminer la catégorie de clients la plus active et sa localisation 

# 1. Pourcentage d'ordres par pays
orders_by_country = passif_ok['Business Country (Business Relationship) (Business Relationship)'].value_counts(normalize=True) * 100
orders_by_country = orders_by_country.reset_index()
orders_by_country.columns = ['Country', 'Order Percentage']
orders_by_country = orders_by_country.sort_values(by='Order Percentage', ascending = False) 

# 2. Pourcentage d'ordres par segmentation
orders_by_segmentation = passif_ok['BR Segmentation (Business Relationship) (Business Relationship)'].value_counts(normalize=True) * 100
orders_by_segmentation = orders_by_segmentation.reset_index()
orders_by_segmentation.columns = ['Segmentation', 'Order Percentage']
orders_by_segmentation = orders_by_segmentation.sort_values(by='Order Percentage', ascending = False) 

# 3. Pourcentage du nombre de parts détenues par pays
quantity_by_country = (
    passif_ok.groupby('Business Country (Business Relationship) (Business Relationship)')['Quantity'].sum()
    / passif_ok['Quantity'].sum() * 100
).reset_index()
quantity_by_country.columns = ['Country', 'Quantity Percentage']
quantity_by_country = quantity_by_country.sort_values(by='Quantity Percentage', ascending = False)  

# 4. Pourcentage du nombre de parts détenues par segmentation
quantity_by_segmentation = (
    passif_ok.groupby('BR Segmentation (Business Relationship) (Business Relationship)')['Quantity'].sum()
    / passif_ok['Quantity'].sum() * 100
).reset_index()
quantity_by_segmentation.columns = ['Segmentation', 'Quantity Percentage']
quantity_by_segmentation = quantity_by_segmentation.sort_values(by='Quantity Percentage', ascending = False) 



# Etape 1 : Passage de passif_ok à passif_final 
passif_final = pd.DataFrame()
for annee in passif_annees:
    passif_final = pd.concat([passif_final, passif_annees[annee]], ignore_index=True)

colonnes_a_conserver2 = [
    'Share Type', 
    'Date',   
    'Net Inflows MTD (€)',
    'BR Segmentation (Business Relationship) (Business Relationship)',
    'Business Country (Business Relationship) (Business Relationship)',
    'Month'
]
passif_final = passif_final[colonnes_a_conserver2]


# Etape 2 : Merge entre passif et performance 
# Créer une colonne YearMonth dans les deux dataframes
# On créé un format Année-Mois (par exemple "2023-01") afin de pouvoir faire la jointure
passif_final['YEARMONTH'] = passif_final['Date'].dt.strftime('%Y-%m')
perf_all_ratios['YEARMONTH'] = perf_all_ratios['DATE'].dt.strftime('%Y-%m')

# Jointure sur YearMonth et TYPE/Share Type
# On veut joindre chaque enregistrement du passif_final (investisseur) aux ratios correspondants du fichier performance 
passif_perf = pd.merge(
    passif_final,
    perf_all_ratios[['TYPE', 'YEARMONTH', 'VOLATILITY_FONDS', 'VOLATILITY_BENCH', 'TRACKING_ERROR', 'SHARPE_FONDS',
       'INFO_RATIO', 'ALPHA']],
    left_on=['Share Type', 'YEARMONTH'],
    right_on=['TYPE', 'YEARMONTH'],
    how='inner'
)

colonnes_a_conserver3 = [ 
    'Date', 
    'Net Inflows MTD (€)',
    'BR Segmentation (Business Relationship) (Business Relationship)',
    'Business Country (Business Relationship) (Business Relationship)',
    'Month', 
    'YEARMONTH',
    'TYPE', 
    'VOLATILITY_FONDS', 
    'VOLATILITY_BENCH', 
    'TRACKING_ERROR', 
    'SHARPE_FONDS',
    'INFO_RATIO', 
    'ALPHA'
]

passif_perf = passif_perf[colonnes_a_conserver3]


# Etape 3 : Préparation du dataframe pour entrainement du modèle 
# Filtre pour le ratio de Sharpe
sharpe_passif_perf = [
    'TYPE', 
    'Date', 
    'YEARMONTH', 
    'Month', 
    'BR Segmentation (Business Relationship) (Business Relationship)', 
    'Business Country (Business Relationship) (Business Relationship)', 
    'Net Inflows MTD (€)', 
    'SHARPE_FONDS', 
]
passif_perf_sharpe = passif_perf[sharpe_passif_perf]

# Pivot multi-index : (YearMonth, Share Type) en index, catégories en colonnes
sharpe_pivot = passif_perf_sharpe.pivot_table(
    index=['YEARMONTH', 'TYPE'], 
    columns='BR Segmentation (Business Relationship) (Business Relationship)', 
    values='Net Inflows MTD (€)',
    aggfunc='sum' # Au cas où plusieurs lignes par (YearMonth, Share Type, catégorie)
)

# Maintenant, on a un DF avec un index à deux niveaux (YEARMONTH, TYPE) et une colonne par catégorie d'investisseur.
# On veut joindre le ratio de Sharpe correspondant
# Le ratio de Sharpe dépend aussi de (YEARMONTH, TYPE)
# On crée un DataFrame tracking error avec le même index

passif_perf_sharpe = passif_perf_sharpe[['YEARMONTH', 'TYPE', 'SHARPE_FONDS']].drop_duplicates()
passif_perf_sharpe = passif_perf_sharpe.set_index(['YEARMONTH', 'TYPE'])

# Joindre le ratio de Sharpe
sharpe_merged = sharpe_pivot.join(passif_perf_sharpe)


# Etape 4 : Entrainement du modèle 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt

# Liste des catégories d'investisseurs
investor_categories = [
    "AM company / EdR Fund of Funds",
    "AM company / Fund of Funds",
    "AM company / Independant",
    "AM company / Own account",
    "Bank / EdR Private Banking",
    "Bank / Online Broker",
    "Bank / Own Account",
    "Bank / Private Banking",
    "Bank / Retail",
    "Bank / Trading Platform",
    "Independant Financial Advisor",
    "Insurance Company / Mutuelle",
    "Insurance Company / Own Account",
    "Insurance Company / Retirement Scheme",
    "Insurance Company / Unit Link",
    "Multi Familly Office / Wealth manager",
    "Pension Fund / Employee Saving Scheme",
    "Pension Fund / Other",
    "Single Family Office",
    "Sovereign / Pension Fund",
    "Unknown"
]

# Dictionnaire pour stocker les modèles et les résultats
trained_models = {}
sharpe_merged = sharpe_merged.reset_index()

# Boucle sur chaque catégorie d'investisseurs
for investor_category in investor_categories:
    print(f"\nTraitement de la catégorie : {investor_category}")

    if investor_category not in sharpe_merged.columns:
        print(f"La catégorie {investor_category} n'existe pas dans les colonnes du dataframe.")
        continue

    # Création de la cible binaire
    sharpe_merged['target'] = (sharpe_merged[investor_category] > 0).astype(int)

    # Vérification de la distribution des classes
    if sharpe_merged['target'].nunique() < 2:
        print(f"Skipping {investor_category} : une seule classe trouvée.")
        continue

    # Variables explicatives
    X = sharpe_merged[['SHARPE_FONDS', 'TYPE']]
    X = pd.get_dummies(X, columns=['TYPE'], prefix='TYPE')
    y = sharpe_merged['target']

    # Séparation en jeu d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modèle LightGBM
    model = lgb.LGBMClassifier(random_state=42, n_estimators=100, learning_rate=0.1)

    # Entraînement
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')


    # Sauvegarde du modèle
    trained_models[investor_category] = {
        'model': model,
        'X_test': X_test,
        'y_test': y_test
    }


# Etape 5 : Fonction de prédictions 
def predict_category(category, input_data):
    if category not in trained_models:
        print(f"Modèle pour {category} non trouvé.")
        return
    model = trained_models[category]['model']
    prediction = model.predict(input_data)
    return prediction

# Créer les onglets
tab1, tab2, tab3 = st.tabs(["Vue d'ensemble", "Analyse performance", "Analyse prédictive"])

# Contenu de l'onglet "Vue d'ensemble"
with tab1:
        # ---- PRÉTRAITEMENT DES DONNÉES ----
    positions['INVENTORY_DATE'] = pd.to_datetime(positions['INVENTORY_DATE'])
    positions['YEAR'] = positions['INVENTORY_DATE'].dt.year

    # Fonction pour extraire la région depuis SECURITY_ID
    def extract_region(security_id):
        match = re.search(r'\s([A-Z]{2})$', str(security_id))
        return match.group(1) if match else None

    positions['REGION'] = positions['SECURITY_ID'].apply(extract_region)
    df_pos_valid = positions.dropna(subset=['REGION'])

    # ---- SECTION 1 : Vue d'ensemble ----
    st.header("📈 Vue d'ensemble des Securities")

        # Agréger les données pour éviter les doublons
    aggregated_data = positions.groupby(['INVENTORY_DATE', 'SECURITY_NAME'])['DIRTY_VALUE_TOTAL_PC'].sum().reset_index()

    # Identification des Top 5 et Bottom 5
    performance = aggregated_data.groupby('SECURITY_NAME')['DIRTY_VALUE_TOTAL_PC'].sum().reset_index()
    performance = performance.sort_values(by='DIRTY_VALUE_TOTAL_PC', ascending=False)
    top_5 = performance.head(5)['SECURITY_NAME']
    bottom_5 = performance.tail(5)['SECURITY_NAME']

    # Combiner les Top 5 et Bottom 5 sans doublons
    combined_securities = pd.concat([top_5, bottom_5]).drop_duplicates()

    # Initialisation de la figure
    fig = go.Figure()

    # Indices de visibilité
    visibility_top5 = []
    visibility_bottom5 = []
    visibility_combined = []
    visibility_all = []

    # Ajouter les Top 5 (visibles par défaut)
    for security in top_5:
        data_filtered = aggregated_data[aggregated_data['SECURITY_NAME'] == security]
        fig.add_trace(go.Scatter(
            x=data_filtered['INVENTORY_DATE'],
            y=data_filtered['DIRTY_VALUE_TOTAL_PC'],
            mode='lines',
            name=f"{security}",
            visible=True  # Visible par défaut
        ))
        visibility_top5.append(True)
        visibility_bottom5.append(False)
        visibility_combined.append(True)
        visibility_all.append(True)

    # Ajouter les Bottom 5 (non visibles par défaut)
    for security in bottom_5:
        data_filtered = aggregated_data[aggregated_data['SECURITY_NAME'] == security]
        fig.add_trace(go.Scatter(
            x=data_filtered['INVENTORY_DATE'],
            y=data_filtered['DIRTY_VALUE_TOTAL_PC'],
            mode='lines',
            name=f"{security}",
            visible=False  # Caché par défaut
        ))
        visibility_top5.append(False)
        visibility_bottom5.append(True)
        visibility_combined.append(True)
        visibility_all.append(True)

    # Ajouter toutes les autres securities (légende dynamique)
    remaining_securities = performance['SECURITY_NAME'][~performance['SECURITY_NAME'].isin(combined_securities)]
    for security in remaining_securities:
        data_filtered = aggregated_data[aggregated_data['SECURITY_NAME'] == security]
        fig.add_trace(go.Scatter(
            x=data_filtered['INVENTORY_DATE'],
            y=data_filtered['DIRTY_VALUE_TOTAL_PC'],
            mode='lines',
            name=f"{security}",
            visible='legendonly'  # Caché mais activable via la légende
        ))
        visibility_top5.append(False)
        visibility_bottom5.append(False)
        visibility_combined.append(False)
        visibility_all.append(True)

    # Ajouter des boutons pour la liste déroulante
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[{'visible': visibility_top5}],
                        label="Top 5",
                        method="update"
                    ),
                    dict(
                        args=[{'visible': visibility_bottom5}],
                        label="Bottom 5",
                        method="update"
                    ),
                    dict(
                        args=[{'visible': visibility_combined}],
                        label="Top 5 + Bottom 5",
                        method="update"
                    ),
                    dict(
                        args=[{'visible': visibility_all}],
                        label="Tout afficher",
                        method="update"
                    ),
                ]),
                direction="down",
                showactive=True,
                x=0.1,
                y=1.15
            ),
        ],
        title="Évolution des Securities : Top 5, Bottom 5 et Sélection Dynamique",
        xaxis_title="Date",
        yaxis_title="Valeur Totale (DIRTY_VALUE_TOTAL_PC)",
        template="plotly_white",
        legend_title="Securities"
    )

    
# Afficher le graphique
    st.plotly_chart(fig)

    st.markdown("---")  # Séparateur visuel

    # ---- SECTION 2 : Fluctuations par Région ----
    st.header("🌍 Fluctuations par Région et Année")

    # Colonnes pour les deux graphiques
    col1, col2, = st.columns([0.7,0.3])

    with col1:
        st.subheader("Volumes détenus")
        fluctuations_volume = df_pos_valid.groupby(['YEAR', 'REGION'])['BALANCE_NOMINAL_OR_NUMBER'].sum().reset_index()
        fig1 = px.bar(
            fluctuations_volume,
            x='REGION',
            y='BALANCE_NOMINAL_OR_NUMBER',
            color='YEAR',
            title="Volumes détenus par Région (Logarithmique)",
            barmode='group'
        )
        fig1.update_yaxes(type='log', title_text="Volumes détenus (log scale)")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Valeurs des Positions")
        fluctuations_value = df_pos_valid.groupby(['YEAR', 'REGION'])['DIRTY_VALUE_TOTAL_PC'].sum().reset_index()
        fig2 = px.bar(
            fluctuations_value,
            x='REGION',
            y='DIRTY_VALUE_TOTAL_PC',
            color='YEAR',
            title="Valeurs des Positions par Région (Logarithmique)",
            barmode='group'
        )
        fig2.update_yaxes(type='log', title_text="Valeur Totale (log scale)")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("### Légende des Abréviations")

        # Code HTML pour un tableau stylisé
        table_html = """
        <style>
            table {
                width: 20%;
                margin-left: auto;
                margin-right: auto;
                border-collapse: collapse;
            }
            th, td {
                border: 1px solid #ddd;
                text-align: left;
                padding: 8px;
            }
            th {
                background-color: #4CAF50;
                color: white;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            tr:nth-child(odd) {
                background-color: #ffffff;
            }
        </style>

        <table>
            <tr>
                <th>Abréviation</th>
                <th>Région correspondante</th>
            </tr>
            <tr>
                <td>BB</td>
                <td>Brésil (Bovespa/B3)</td>
            </tr>
            <tr>
                <td>FP</td>
                <td>France (Paris - Euronext)</td>
            </tr>
            <tr>
                <td>GR</td>
                <td>Allemagne (Germany - XETRA)</td>
            </tr>
            <tr>
                <td>HK</td>
                <td>Hong Kong</td>
            </tr>
            <tr>
                <td>JP</td>
                <td>Japon (Japan - TSE)</td>
            </tr>
            <tr>
                <td>KS</td>
                <td>Corée du Sud (KOSPI)</td>
            </tr>
            <tr>
                <td>NA</td>
                <td>Amérique du Nord (North America)</td>
            </tr>
            <tr>
                <td>SM</td>
                <td>Espagne (Spain - Madrid Exchange)</td>
            </tr>
            <tr>
                <td>SP</td>
                <td>Singapour</td>
            </tr>
            <tr>
                <td>TT</td>
                <td>Taïwan (Taipei Exchange)</td>
            </tr>
            <tr>
                <td>US</td>
                <td>États-Unis (USA - NYSE/NASDAQ)</td>
            </tr>
            <tr>
                <td>CN</td>
                <td>Chine (China - Shanghai/Shenzhen)</td>
            </tr>
            <tr>
                <td>SS</td>
                <td>Suisse (Switzerland - SIX)</td>
            </tr>
            <tr>
                <td>LX</td>
                <td>Luxembourg</td>
            </tr>
        </table>
        """

        # Afficher le tableau en Streamlit avec HTML
        st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("---")  # Séparateur visuel

    # ---- SECTION 3 : Répartition en Camemberts ----
    st.header("🥧 Répartition des Pourcentages par Pays et Segmentation")

    # Colonnes pour les camemberts
    col1, col2 = st.columns(2)

    threshold = 1.5  # Seuil pour grouper les petites valeurs

    with col1:
        st.subheader("Répartition par Pays")
        orders_by_country['Category'] = orders_by_country['Country']
        orders_by_country.loc[orders_by_country['Order Percentage'] < threshold, 'Category'] = 'AUTRE'
        grouped_df = orders_by_country.groupby('Category')['Order Percentage'].sum().reset_index()
        fig_pie1 = px.pie(
            grouped_df,
            names='Category',
            values='Order Percentage',
            title="Répartition des pourcentages par pays",
            hole=0.3
        )
        st.plotly_chart(fig_pie1, use_container_width=True)

    with col2:
        st.subheader("Répartition par Segmentation")
        orders_by_segmentation['Category'] = orders_by_segmentation['Segmentation']
        orders_by_segmentation.loc[orders_by_segmentation['Order Percentage'] < threshold, 'Category'] = 'AUTRE'
        grouped_df2 = orders_by_segmentation.groupby('Category')['Order Percentage'].sum().reset_index()
        fig_pie2 = px.pie(
            grouped_df2,
            names='Category',
            values='Order Percentage',
            title="Répartition des pourcentages par segmentation",
            hole=0.3
        )
        st.plotly_chart(fig_pie2, use_container_width=True)

    st.markdown("---")  # Séparateur visuel

    # ---- SECTION 4 : Quantités par Pays et Segmentation ----
    st.header("📊 Répartition des Quantités")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Quantités par Pays")
        quantity_by_country['Category'] = quantity_by_country['Country']
        quantity_by_country.loc[quantity_by_country['Quantity Percentage'] < threshold, 'Category'] = 'AUTRE'
        grouped_df3 = quantity_by_country.groupby('Category')['Quantity Percentage'].sum().reset_index()
        fig_pie3 = px.pie(
            grouped_df3,
            names='Category',
            values='Quantity Percentage',
            title="Répartition des quantités par pays",
            hole=0.3
        )
        st.plotly_chart(fig_pie3, use_container_width=True)

    with col2:
        st.subheader("Quantités par Segmentation")
        quantity_by_segmentation['Category'] = quantity_by_segmentation['Segmentation']
        quantity_by_segmentation.loc[quantity_by_segmentation['Quantity Percentage'] < threshold, 'Category'] = 'AUTRE'
        grouped_df4 = quantity_by_segmentation.groupby('Category')['Quantity Percentage'].sum().reset_index()
        fig_pie4 = px.pie(
            grouped_df4,
            names='Category',
            values='Quantity Percentage',
            title="Répartition des quantités par segmentation",
            hole=0.3
        )
        st.plotly_chart(fig_pie4, use_container_width=True)

# Contenu de l'onglet "Analyse performance"
with tab2:
    # Ajoute ici ton contenu spécifique à l'Analyse performance

    # --- Fusion des DataFrames ---
    datasets = [
        INFO_RATIO_df_Glissant, SHARPE_BENCH_df_Glissant, SHARPE_FONDS_df_Glissant,
        VOLATILITY_FONDS_df_Glissant, VOLATILITY_BENCH_df_Glissant, df_ALPHA_df_Glissant
    ]
    final_df_ = pd.concat(datasets, ignore_index=True)

    # --- Renommage des colonnes ---
    final_df_.rename(columns={
        'Moyenne_1_Ans': '1 an',
        'Moyenne_2_Ans': '2 ans',
        'Moyenne_4_Ans': '4 ans'
    }, inplace=True)

    # --- Affichage amélioré du DataFrame ---
    st.title("Analyse de Performance")
    st.markdown("### Résumé des ratios par période d'observation")

    # Mise en forme visuelle avec `st.dataframe` et des options supplémentaires
    st.dataframe(
        final_df_.style.format({
            "1 an": "{:.4f}", 
            "2 ans": "{:.4f}", 
            "4 ans": "{:.4f}"
        }).set_properties(**{
            'text-align': 'center',
            'background-color': '#f9f9f9',
            'font-size': '14px'
        }).set_table_styles([
            {'selector': 'th', 'props': [('font-size', '14px'), ('text-align', 'center'), ('background-color', '#e3e3e3')]},
            {'selector': 'td', 'props': [('text-align', 'center')]}
        ])
    )

    st.markdown("---")  # Séparateur visuel

    st.subheader("Évolution de la collecte et de la performance du fonds")


    # MOUVEMENT COLLECTE
    # Sélectionner uniquement les colonnes "Net Inflows YTD (€)" et "Date"
    passif_filtered = passif[['Date','Net Inflows MTD (€)']]
    passif_filtered['Date'] = pd.to_datetime(passif_filtered['Date']).dt.strftime('%Y-%m')

    # Trier le DataFrame par la colonne "Date" dans l'ordre croissant
    passif_filtered = passif_filtered.sort_values(by='Date', ascending=True)
    passif_filtered = passif_filtered.rename(columns={'Date': 'DATE'})


    #RATIO 
    # Transformation de la colonne 'DATE' pour chaque DataFrame
    VOLATILITY_FONDS_df['DATE'] = pd.to_datetime(VOLATILITY_FONDS_df['DATE']).dt.strftime('%Y-%m')
    ALPHA_df['DATE'] = pd.to_datetime(ALPHA_df['DATE']).dt.strftime('%Y-%m')
    SHARPE_FONDS_df['DATE'] = pd.to_datetime(SHARPE_FONDS_df['DATE']).dt.strftime('%Y-%m')
    INFO_RATIO_df['DATE'] = pd.to_datetime(INFO_RATIO_df['DATE']).dt.strftime('%Y-%m')

    # Renommage des DataFrames après transformation
    VOLATILITY_FONDS_transformed = VOLATILITY_FONDS_df.sort_values(by='DATE', ascending=True)
    ALPHA_transformed = ALPHA_df.sort_values(by='DATE', ascending=True)
    SHARPE_FONDS_transformed = SHARPE_FONDS_df.sort_values(by='DATE', ascending=True)
    SHARPE_FONDS_transformed = SHARPE_FONDS_transformed.drop(columns=['Mois'], errors='ignore')
    INFO_RATIO_transformed = INFO_RATIO_df.sort_values(by='DATE', ascending=True)

    #MERGE DES DES DF RATIO ET PASSIF_FIlter
    # Fusionner passif_filtered avec VOLATILITY_FONDS_transformed
    merged_passif_volatility = passif_filtered.merge(VOLATILITY_FONDS_transformed, on='DATE', how='left')

    # Fusionner passif_filtered avec ALPHA_transformed
    merged_passif_alpha = passif_filtered.merge(ALPHA_transformed, on='DATE', how='left')

    # Fusionner passif_filtered avec SHARPE_FONDS_transformed
    merged_passif_sharpe = passif_filtered.merge(SHARPE_FONDS_transformed, on='DATE', how='left')

    # Fusionner passif_filtered avec INFO_RATIO_transformed
    merged_passif_info_ratio = passif_filtered.merge(INFO_RATIO_transformed, on='DATE', how='left')

    # Normaliser les données des ratios pour les rendre comparables à Net Inflows YTD (€)

    # Utilisation de la normalisation min-max pour les ratios
    normalized_volatility = (merged_passif_volatility['VOLATILITY_FONDS'] - merged_passif_volatility['VOLATILITY_FONDS'].min()) / (merged_passif_volatility['VOLATILITY_FONDS'].max() - merged_passif_volatility['VOLATILITY_FONDS'].min())
    normalized_alpha = (merged_passif_alpha['ALPHA'] - merged_passif_alpha['ALPHA'].min()) / (merged_passif_alpha['ALPHA'].max() - merged_passif_alpha['ALPHA'].min())
    normalized_sharpe = (merged_passif_sharpe['SHARPE_FONDS'] - merged_passif_sharpe['SHARPE_FONDS'].min()) / (merged_passif_sharpe['SHARPE_FONDS'].max() - merged_passif_sharpe['SHARPE_FONDS'].min())
    normalized_info_ratio = (merged_passif_info_ratio['INFO_RATIO'] - merged_passif_info_ratio['INFO_RATIO'].min()) / (merged_passif_info_ratio['INFO_RATIO'].max() - merged_passif_info_ratio['INFO_RATIO'].min())

    # Créer un graphique avec deux axes y
    fig = go.Figure()

    # Ajouter les courbes des ratios normalisés
    fig.add_trace(go.Scatter(
        x=merged_passif_volatility['DATE'], 
        y=normalized_volatility, 
        mode='lines', 
        name='Volatilité (Normalisée)', 
        line=dict(color='blue'),
        yaxis='y1'
    ))

    fig.add_trace(go.Scatter(
        x=merged_passif_alpha['DATE'], 
        y=normalized_alpha, 
        mode='lines', 
        name='Alpha (Normalisé)', 
        line=dict(color='green'),
        yaxis='y1'
    ))

    fig.add_trace(go.Scatter(
        x=merged_passif_sharpe['DATE'], 
        y=normalized_sharpe, 
        mode='lines', 
        name='Sharpe Ratio (Normalisé)', 
        line=dict(color='orange'),
        yaxis='y1'
    ))

    fig.add_trace(go.Scatter(
        x=merged_passif_info_ratio['DATE'], 
        y=normalized_info_ratio, 
        mode='lines', 
        name='Info Ratio (Normalisé)', 
        line=dict(color='purple'),
        yaxis='y1'
    ))

    # Ajouter la courbe Net Inflows YTD (€)
    fig.add_trace(go.Scatter(
        x=merged_passif_volatility[merged_passif_volatility['Net Inflows MTD (€)'] >= 0]['DATE'],
        y=merged_passif_volatility[merged_passif_volatility['Net Inflows MTD (€)'] >= 0]['Net Inflows MTD (€)'],
        mode='lines',
        name='Net Inflows MTD (€)',
        line=dict(color='black'),
        yaxis='y2'
    ))

    fig.add_trace(go.Scatter(
        x=merged_passif_volatility[merged_passif_volatility['Net Inflows MTD (€)'] < 0]['DATE'],
        y=merged_passif_volatility[merged_passif_volatility['Net Inflows MTD (€)'] < 0]['Net Inflows MTD (€)'],
        mode='lines',
        name='Net Inflows MTD (€) (Negative)',
        line=dict(color='red'),
        yaxis='y2'
    ))

    # Mise en forme de l'axe y2
    fig.update_layout(
        title="Ratios Financiers et Net Inflows MTD (€)",
        xaxis_title="Date",
        yaxis_title="Ratios Normalisés",
        yaxis2=dict(
            title="Net Inflows MTD (€)",
            overlaying='y',
            side='right'
        ),
        legend_title="Ratios et Net Inflows",
        template="plotly_dark",
        hovermode='closest'
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)


    merged_perf = pd.merge(perf_bench_ok, perf_fonds_ok, on=['TYPE', 'DATE'])

    # Formatage des dates pour affichage propre
    merged_perf['DATE'] = merged_perf['DATE'].dt.strftime('%b %Y')  # Format 'Jan 2021'

    # --- Streamlit Layout ---
    st.header("Comparaison des Performances : Fonds vs Benchmark")

    # Créer une figure avec Plotly
    fig = go.Figure()

    # Ajouter les barres pour le Fonds
    fig.add_trace(go.Bar(
        x=merged_perf['DATE'],
        y=merged_perf['R_M_FONDS'],
        name="Performance du Fonds",
        marker=dict(color='blue', opacity=0.7),
    ))

    # Ajouter les barres pour le Benchmark
    fig.add_trace(go.Bar(
        x=merged_perf['DATE'],
        y=merged_perf['R_M_BENCH'],
        name="Performance du Benchmark",
        marker=dict(color='orange', opacity=0.7),
    ))

    # Ajustements de la mise en page pour plus de lisibilité
    fig.update_layout(
        barmode='group',  # Affichage côte à côte
        title="Comparaison des Performances : Fonds vs Benchmark",
        xaxis_title="Date",
        yaxis_title="Performance",
        xaxis=dict(tickangle=45),  # Rotation des dates
        legend=dict(x=0.7, y=1.1, orientation='h'),  # Légende horizontale
        margin=dict(l=50, r=50, t=50, b=50),
        template='plotly_white'
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)


    #VOLATILITE FONDS 
    # Trier par DATE pour préparer les calculs glissants
    perf_all_ratios = perf_all_ratios.sort_values(by='DATE').reset_index(drop=True)

    # Étape 1: Initialiser un DataFrame pour stocker les moyennes glissantes
    final_data = []

    # Définir les fenêtres de temps pour 1 an, 2 ans, 3 ans et 4 ans
    time_windows = {'1an': 1, '2an': 2, '3an': 3, '4an': 4}

    # Calcul des moyennes glissantes pour chaque TYPE
    for type_action in perf_all_ratios['TYPE'].unique():
        df_type = perf_all_ratios[perf_all_ratios['TYPE'] == type_action]
        

        for start_date in pd.date_range(df_type['DATE'].min(), df_type['DATE'].max(), freq='MS'):
            row = {'MONTH': start_date.strftime('%B'), 'TYPE': type_action}

            # Calculer les moyennes glissantes pour chaque fenêtre de temps

            for window_name, years in time_windows.items():
                end_date = start_date + pd.DateOffset(years=years) - pd.DateOffset(days=1)
                mask = (df_type['DATE'] >= start_date) & (df_type['DATE'] <= end_date)
                row[window_name] = df_type.loc[mask, 'VOLATILITY_FONDS'].mean() if not df_type.loc[mask].empty else None
            
            final_data.append(row)


    # Étape 3: Convertir en DataFrame et ajouter les moyennes des périodes
    volatility_df = pd.DataFrame(final_data)

    # Ajouter les colonnes moyennes pour 1_an, 2_an, 4_an
    volatility_df['1_an'] = volatility_df['1an']
    volatility_df['2_an'] = volatility_df[['1an', '2an']].mean(axis=1)
    volatility_df['4_an'] = volatility_df[['1an', '2an', '3an', '4an']].mean(axis=1)

    # Garder uniquement les colonnes nécessaires
    volatility_df = volatility_df[['MONTH', 'TYPE', '1_an', '2_an', '4_an']]

    # Étape 4: Regrouper par TYPE pour obtenir une seule valeur moyenne par TYPE
    volatility_final = volatility_df.groupby('TYPE', as_index=False)[['1_an', '2_an', '4_an']].mean()

    # Étape 1: Convertir DATE en datetime et extraire le mois
    perf_fonds_ok['DATE'] = pd.to_datetime(perf_fonds_ok['DATE'])
    perf_fonds_ok['MONTH'] = perf_fonds_ok['DATE'].dt.strftime('%B')  # Mois complet (Septembre, etc.)


    # Trier par DATE pour préparer les calculs glissants
    perf_fonds_ok = perf_fonds_ok.sort_values(by='DATE').reset_index(drop=True)

    # Étape 2: Initialiser un DataFrame pour stocker les moyennes glissantes
    final_data = []

    # Définir les plages de 1 an, 2 ans, 3 ans, et 4 ans
    time_windows = {'1an': 1, '2an': 2, '3an': 3, '4an': 4}

    # Étape 3: Calcul des moyennes glissantes pour chaque TYPE d'actions
    for type_action in perf_fonds_ok['TYPE'].unique():
        df_type = perf_fonds_ok[perf_fonds_ok['TYPE'] == type_action]
        for start_date in pd.date_range(df_type['DATE'].min(), df_type['DATE'].max(), freq='MS'):
            row = {'MONTH': start_date.strftime('%B'), 'TYPE': type_action}

            for window_name, years in time_windows.items():
                end_date = start_date + pd.DateOffset(years=years) - pd.DateOffset(days=1)
                mask = (df_type['DATE'] >= start_date) & (df_type['DATE'] <= end_date)
                row[window_name] = df_type.loc[mask, 'R_M_FONDS'].mean() if not df_type.loc[mask].empty else None
            
            final_data.append(row)

    # Étape 4: Convertir les résultats en DataFrame avec les colonnes 1an, 2an, 3an, 4an
    perf_df = pd.DataFrame(final_data)

    # Étape 5: Créer un DataFrame final avec les moyennes glissantes sur 1_an, 2_an, et 4_an
    perf_final = perf_df.copy()
    perf_final['1_an'] = perf_final['1an']
    perf_final['2_an'] = perf_final[['1an', '2an']].mean(axis=1)
    perf_final['4_an'] = perf_final[['1an', '2an', '3an', '4an']].mean(axis=1)

    # Garder uniquement les colonnes nécessaires
    perf_final = perf_final[['MONTH', 'TYPE', '1_an', '2_an', '4_an']]

    # Étape 6: Regrouper par TYPE pour obtenir une seule valeur moyenne par TYPE
    perf_final = perf_final.groupby('TYPE', as_index=False)[['1_an', '2_an', '4_an']].mean()

    # Étape 1: Fusionner les DataFrames volatility_final et perf_final par 'TYPE'
    merged_df = pd.merge(volatility_final, perf_final, on='TYPE', suffixes=('_vol', '_perf'))

    st.header("Performance et volatilité du fonds")

    # Création de la mise en page avec 3 colonnes
    col1, col2, col3 = st.columns(3)

    # Graphique pour la période glissante de 1 an
    with col1:
        fig_1an = px.scatter(
            merged_df, 
            x='1_an_vol', 
            y='1_an_perf', 
            color='TYPE',  # Différencier les types par couleur
            labels={
                '1_an_vol': 'Volatilité Annualisée (1 an)', 
                '1_an_perf': 'Performance Annualisée (1 an)'
            },
            title="Performance vs Volatilité - Période 1 an"
        )
        st.plotly_chart(fig_1an)

    # Graphique pour la période glissante de 2 ans

    with col2:
        fig_2an = px.scatter(
            merged_df, 
            x='2_an_vol', 
            y='2_an_perf', 
            color='TYPE',  # Différencier les types par couleur
            labels={
                '2_an_vol': 'Volatilité Annualisée (2 ans)', 
                '2_an_perf': 'Performance Annualisée (2 ans)'
            },
            title="Performance vs Volatilité - Période 2 ans"
        )
        st.plotly_chart(fig_2an)

    # Graphique pour la période glissante de 4 ans
    with col3:
        fig_4an = px.scatter(
            merged_df, 
            x='4_an_vol', 
            y='4_an_perf', 
            color='TYPE',  # Différencier les types par couleur
            labels={
                '4_an_vol': 'Volatilité Annualisée (4 ans)', 
                '4_an_perf': 'Performance Annualisée (4 ans)'
            },
            title="Performance vs Volatilité - Période 4 ans"
        )
        st.plotly_chart(fig_4an)

    st.markdown("---") 

        # ---- Données : Correction de la temporalité ----
    # Conversion de la colonne DATE en datetime
    SHARPE_FONDS_df['DATE'] = pd.to_datetime(SHARPE_FONDS_df['DATE'])

    # Création d'une colonne pour les mois dans un ordre spécifique (Septembre → Août)
    SHARPE_FONDS_df['Mois'] = SHARPE_FONDS_df['DATE'].dt.month_name(

    )

    # Assurer l'ordre des mois de Septembre à Août
    mois_order = ['September', 'October', 'November', 'December', 
                'January', 'February', 'March', 'April', 
                'May', 'June', 'July', 'August']

    # Étape 1 : Définir les plages glissantes
    start_date = SHARPE_FONDS_df['DATE'].min()
    gliding_periods = []

    for i in range(4):  # 4 années glissantes
        end_date = start_date + pd.DateOffset(years=1) - pd.DateOffset(days=1)
        mask = (SHARPE_FONDS_df['DATE'] >= start_date) & (SHARPE_FONDS_df['DATE'] <= end_date)
        temp_df = SHARPE_FONDS_df.loc[mask, ['Mois', 'SHARPE_FONDS']]
        temp_df = temp_df.groupby('Mois')['SHARPE_FONDS'].mean().reindex(mois_order).reset_index()
        temp_df['SHARPE_FONDS'] = temp_df['SHARPE_FONDS'].fillna(0)  # Remplacer valeurs manquantes par 0
        temp_df.rename(columns={'SHARPE_FONDS': f'Année_{i+1}'}, inplace=True)
        gliding_periods.append(temp_df)
        start_date = start_date + pd.DateOffset(years=1)

    # Étape 2 : Fusionner les périodes dans un DataFrame final
    final_df = gliding_periods[0][['Mois']].copy()
    for i, period in enumerate(gliding_periods):
        final_df[f"Année_{i+1}"] = period[f"Année_{i+1}"]

    # ---- Affichage Streamlit ----
    st.subheader("📊 Évolution du SHARPE_FONDS sur 4 années glissantes")

    st.write("""
    Ce graphique interactif affiche les valeurs de **SHARPE_FONDS** sur 4 années glissantes.
    Les valeurs manquantes sont remplacées par 0 pour éviter les ruptures dans le graphique.
    """)

    # ---- Bar chart interactif ----
    fig = px.bar(
        final_df,
        x='Mois',
        y=['Année_1', 'Année_2', 'Année_3', 'Année_4'],
        title="Évolution du SHARPE_FONDS par mois (4 années glissantes)",
        labels={"value": "SHARPE_FONDS", "variable": "Année"},
        barmode='group'
    )

    # Ajustement de l'ordre des mois
    fig.update_xaxes(categoryorder='array', categoryarray=mois_order)

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)


            # ---- Données : Correction de la temporalité ----
    # Conversion de la colonne DATE en datetime
    SHARPE_BENCH_df['DATE'] = pd.to_datetime(SHARPE_BENCH_df['DATE'])

    # Création d'une colonne pour les mois dans un ordre spécifique (Septembre → Août)
    SHARPE_BENCH_df['Mois'] = SHARPE_BENCH_df['DATE'].dt.month_name()

    # Assurer l'ordre des mois de Septembre à Août
    mois_order = ['September', 'October', 'November', 'December', 
                'January', 'February', 'March', 'April', 
                'May', 'June', 'July', 'August']

    # Étape 1 : Définir les plages glissantes
    start_date = SHARPE_BENCH_df['DATE'].min()
    gliding_periods = []

    for i in range(4):  # 4 années glissantes
        end_date = start_date + pd.DateOffset(years=1) - pd.DateOffset(days=1)
        mask = (SHARPE_BENCH_df['DATE'] >= start_date) & (SHARPE_BENCH_df['DATE'] <= end_date)
        temp_df = SHARPE_BENCH_df.loc[mask, ['Mois', 'SHARPE_BENCH']]
        temp_df = temp_df.groupby('Mois')['SHARPE_BENCH'].mean().reindex(mois_order).reset_index()
        temp_df['SHARPE_BENCH'] = temp_df['SHARPE_BENCH'].fillna(0)  # Remplacer valeurs manquantes par 0
        temp_df.rename(columns={'SHARPE_BENCH': f'Année_{i+1}'}, inplace=True)
        gliding_periods.append(temp_df)
        start_date = start_date + pd.DateOffset(years=1)

    # Étape 2 : Fusionner les périodes dans un DataFrame final
    final_df = gliding_periods[0][['Mois']].copy()
    for i, period in enumerate(gliding_periods):
        final_df[f"Année_{i+1}"] = period[f"Année_{i+1}"]

    # ---- Affichage Streamlit ----
    st.subheader("📊 Évolution du SHARPE_BENCH sur 4 années glissantes")

    st.write("""
    Ce graphique interactif affiche les valeurs de **SHARPE_BENCH** sur 4 années glissantes.
    Les valeurs manquantes sont remplacées par 0 pour éviter les ruptures dans le graphique.
    """)

    # ---- Bar chart interactif ----
    fig = px.bar(
        final_df,
        x='Mois',
        y=['Année_1', 'Année_2', 'Année_3', 'Année_4'],
        title="Évolution du SHARPE_BENCH par mois (4 années glissantes)",
        labels={"value": "SHARPE_BENCH", "variable": "Année"},
        barmode='group'
    )

    # Ajustement de l'ordre des mois
    fig.update_xaxes(categoryorder='array', categoryarray=mois_order)

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)


            # ---- Données : Correction de la temporalité ----
    # Conversion de la colonne DATE en datetime
    VOLATILITY_FONDS_df['DATE'] = pd.to_datetime(VOLATILITY_FONDS_df['DATE'])

    # Création d'une colonne pour les mois dans un ordre spécifique (Septembre → Août)
    VOLATILITY_FONDS_df['Mois'] = VOLATILITY_FONDS_df['DATE'].dt.month_name()

    # Assurer l'ordre des mois de Septembre à Août
    mois_order = ['September', 'October', 'November', 'December', 
                'January', 'February', 'March', 'April', 
                'May', 'June', 'July', 'August']

    # Étape 1 : Définir les plages glissantes
    start_date = VOLATILITY_FONDS_df['DATE'].min()
    gliding_periods = []

    for i in range(4):  # 4 années glissantes
        end_date = start_date + pd.DateOffset(years=1) - pd.DateOffset(days=1)
        mask = (VOLATILITY_FONDS_df['DATE'] >= start_date) & (VOLATILITY_FONDS_df['DATE'] <= end_date)
        temp_df = VOLATILITY_FONDS_df.loc[mask, ['Mois', 'VOLATILITY_FONDS']]
        temp_df = temp_df.groupby('Mois')['VOLATILITY_FONDS'].mean().reindex(mois_order).reset_index()
        temp_df['VOLATILITY_FONDS'] = temp_df['VOLATILITY_FONDS'].fillna(0)  # Remplacer valeurs manquantes par 0
        temp_df.rename(columns={'VOLATILITY_FONDS': f'Année_{i+1}'}, inplace=True)
        gliding_periods.append(temp_df)
        start_date = start_date + pd.DateOffset(years=1)

    # Étape 2 : Fusionner les périodes dans un DataFrame final
    final_df = gliding_periods[0][['Mois']].copy()
    for i, period in enumerate(gliding_periods):
        final_df[f"Année_{i+1}"] = period[f"Année_{i+1}"]

    # ---- Affichage Streamlit ----
    st.subheader("📊 Évolution de la VOLATILITY_FONDS sur 4 années glissantes")

    st.write("""
    Ce graphique interactif affiche les valeurs de **VOLATILITY_FONDS** sur 4 années glissantes.
    Les valeurs manquantes sont remplacées par 0 pour éviter les ruptures dans le graphique.
    """)

    # ---- Bar chart interactif ----
    fig = px.bar(
        final_df,
        x='Mois',
        y=['Année_1', 'Année_2', 'Année_3', 'Année_4'],
        title="Évolution de la VOLATILITY_FONDS par mois (4 années glissantes)",
        labels={"value": "VOLATILITY_FONDS", "variable": "Année"},
        barmode='group'
    )

    # Ajustement de l'ordre des mois
    fig.update_xaxes(categoryorder='array', categoryarray=mois_order)

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)



            # ---- Données : Correction de la temporalité ----
    # Conversion de la colonne DATE en datetime
    VOLATILITY_BENCH_df['DATE'] = pd.to_datetime(VOLATILITY_BENCH_df['DATE'])

    # Création d'une colonne pour les mois (en anglais par défaut)
    VOLATILITY_BENCH_df['Mois'] = VOLATILITY_BENCH_df['DATE'].dt.month_name()

    # Assurer l'ordre des mois de Septembre à Août
    mois_order = ['September', 'October', 'November', 'December', 
                'January', 'February', 'March', 'April', 
                'May', 'June', 'July', 'August']

    # Étape 1 : Définir les plages glissantes
    start_date = VOLATILITY_BENCH_df['DATE'].min()
    gliding_periods = []

    for i in range(4):  # 4 années glissantes
        end_date = start_date + pd.DateOffset(years=1) - pd.DateOffset(days=1)
        mask = (VOLATILITY_BENCH_df['DATE'] >= start_date) & (VOLATILITY_BENCH_df['DATE'] <= end_date)
        temp_df = VOLATILITY_BENCH_df.loc[mask, ['Mois', 'VOLATILITY_BENCH']]
        temp_df = temp_df.groupby('Mois')['VOLATILITY_BENCH'].mean().reindex(mois_order).reset_index()
        temp_df['VOLATILITY_BENCH'] = temp_df['VOLATILITY_BENCH'].fillna(0)  # Remplacer valeurs manquantes par 0
        temp_df.rename(columns={'VOLATILITY_BENCH': f'Année_{i+1}'}, inplace=True)
        gliding_periods.append(temp_df)
        start_date = start_date + pd.DateOffset(years=1)

    # Étape 2 : Fusionner les périodes dans un DataFrame final
    final_df = gliding_periods[0][['Mois']].copy()
    for i, period in enumerate(gliding_periods):
        final_df[f"Année_{i+1}"] = period[f"Année_{i+1}"]

    # ---- Affichage Streamlit ----
    st.subheader("📊 Évolution de la VOLATILITY_BENCH sur 4 années glissantes")

    st.write("""
    Ce graphique interactif affiche les valeurs de **VOLATILITY_BENCH** sur 4 années glissantes.
    Les valeurs manquantes sont remplacées par 0 pour éviter les ruptures dans le graphique.
    """)

    # ---- Bar chart interactif ----
    fig = px.bar(
        final_df,
        x='Mois',
        y=['Année_1', 'Année_2', 'Année_3', 'Année_4'],
        title="Évolution de la VOLATILITY_BENCH par mois (4 années glissantes)",
        labels={"value": "VOLATILITY_BENCH", "variable": "Année"},
        barmode='group'
    )

    # Ajustement de l'ordre des mois
    fig.update_xaxes(categoryorder='array', categoryarray=mois_order)

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)



            # ---- Données : Correction de la temporalité ----
    # Conversion de la colonne DATE en datetime
    ALPHA_df['DATE'] = pd.to_datetime(ALPHA_df['DATE'])

    # Création d'une colonne pour les mois dans un ordre spécifique (Septembre → Août)
    ALPHA_df['Mois'] = ALPHA_df['DATE'].dt.month_name()

    # Assurer l'ordre des mois de Septembre à Août
    mois_order = ['September', 'October', 'November', 'December', 
                'January', 'February', 'March', 'April', 
                'May', 'June', 'July', 'August']

    # Étape 1 : Définir les plages glissantes
    start_date = ALPHA_df['DATE'].min()
    gliding_periods = []

    for i in range(4):  # 4 années glissantes
        end_date = start_date + pd.DateOffset(years=1) - pd.DateOffset(days=1)
        mask = (ALPHA_df['DATE'] >= start_date) & (ALPHA_df['DATE'] <= end_date)
        temp_df = ALPHA_df.loc[mask, ['Mois', 'ALPHA']]
        temp_df = temp_df.groupby('Mois')['ALPHA'].mean().reindex(mois_order).reset_index()
        temp_df['ALPHA'] = temp_df['ALPHA'].fillna(0)  # Remplacer valeurs manquantes par 0
        temp_df.rename(columns={'ALPHA': f'Année_{i+1}'}, inplace=True)
        gliding_periods.append(temp_df)
        start_date = start_date + pd.DateOffset(years=1)

    # Étape 2 : Fusionner les périodes dans un DataFrame final
    final_df = gliding_periods[0][['Mois']].copy()
    for i, period in enumerate(gliding_periods):
        final_df[f"Année_{i+1}"] = period[f"Année_{i+1}"]

    # ---- Affichage Streamlit ----
    st.subheader("📊 Évolution des ALPHA sur 4 années glissantes")

    st.write("""
    Ce graphique interactif affiche les valeurs de **ALPHA** sur 4 années glissantes.
    Les valeurs manquantes sont remplacées par 0 pour éviter les ruptures dans le graphique.
    """)

    # ---- Bar chart interactif ----
    fig = px.bar(
        final_df,
        x='Mois',
        y=['Année_1', 'Année_2', 'Année_3', 'Année_4'],
        title="Évolution des ALPHA par mois (4 années glissantes)",
        labels={"value": "ALPHA", "variable": "Année"},
        barmode='group'
    )

    # Ajustement de l'ordre des mois
    fig.update_xaxes(categoryorder='array', categoryarray=mois_order)

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)

            # ---- Données : Correction de la temporalité ----
    # Conversion de la colonne DATE en datetime
    INFO_RATIO_df['DATE'] = pd.to_datetime(INFO_RATIO_df['DATE'])

    # Création d'une colonne pour les mois (ordre spécifique Septembre → Août)
    INFO_RATIO_df['Mois'] = INFO_RATIO_df['DATE'].dt.month_name()

    # Assurer l'ordre des mois de Septembre à Août
    mois_order = ['September', 'October', 'November', 'December', 
                'January', 'February', 'March', 'April', 
                'May', 'June', 'July', 'August']

    # Étape 1 : Définir les plages glissantes
    start_date = INFO_RATIO_df['DATE'].min()
    gliding_periods = []

    for i in range(4):  # 4 années glissantes
        end_date = start_date + pd.DateOffset(years=1) - pd.DateOffset(days=1)
        mask = (INFO_RATIO_df['DATE'] >= start_date) & (INFO_RATIO_df['DATE'] <= end_date)
        temp_df = INFO_RATIO_df.loc[mask, ['Mois', 'INFO_RATIO']]
        temp_df = temp_df.groupby('Mois')['INFO_RATIO'].mean().reindex(mois_order).reset_index()
        temp_df['INFO_RATIO'] = temp_df['INFO_RATIO'].fillna(0)  # Remplacer valeurs manquantes par 0
        temp_df.rename(columns={'INFO_RATIO': f'Année_{i+1}'}, inplace=True)
        gliding_periods.append(temp_df)
        start_date = start_date + pd.DateOffset(years=1)

    # Étape 2 : Fusionner les périodes dans un DataFrame final
    final_df = gliding_periods[0][['Mois']].copy()
    for i, period in enumerate(gliding_periods):
        final_df[f"Année_{i+1}"] = period[f"Année_{i+1}"]

    # ---- Affichage Streamlit ----
    st.subheader("📊 Évolution des INFO_RATIO sur 4 années glissantes")

    st.write("""
    Ce graphique interactif affiche les valeurs de **INFO_RATIO** sur 4 années glissantes.
    Les valeurs manquantes sont remplacées par 0 pour éviter les ruptures dans le graphique.
    """)

    # ---- Bar chart interactif ----
    fig = px.bar(
        final_df,
        x='Mois',
        y=['Année_1', 'Année_2', 'Année_3', 'Année_4'],
        title="Évolution des INFO_RATIO par mois (4 années glissantes)",
        labels={"value": "INFO_RATIO", "variable": "Année"},
        barmode='group'
    )

    # Ajustement de l'ordre des mois
    fig.update_xaxes(categoryorder='array', categoryarray=mois_order)

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)




# Contenu de l'onglet "Analyse prédictive"
with tab3:
    # Titre et sous-titre
    col1, col2 = st.columns([0.2,0.8])
    col1.image("Edmind.png", width=200)
    col2.title("EDMIND - AI for Asset Management")
    st.subheader("""Analyse prédictive : algorithme de classification binaire capable de déterminer, pour chaque catégorie d’investisseurs, les chances qu’elle investisse en fonction de l’évolution du ratio de Sharpe. 
                 \n \tClassificateur entraîné par LightGBM (Merci de saisir 6 chiffres après la virgule).
                 """)

    # Zone de saisie du ratio de Sharpe
    ratio_input = st.text_input("Veuillez saisir le ratio de Sharpe :")

    # Bouton pour lancer la prédiction
    if st.button("Prédire"):
        try:
            ratio_val = float(ratio_input)
        except ValueError:
            st.error("Veuillez saisir une valeur numérique valide pour le ratio de Sharpe.")
        else:
            # On recrée les données de la même façon que lors de l'entraînement
            # X = sharpe_merged[['SHARPE_FONDS', 'TYPE']]
            # X = pd.get_dummies(X, columns=['TYPE'], prefix='TYPE']

            # Choix d'un type par défaut (à adapter)
            default_type = "TYPE_A"  
            
            # Création de l'input avec les mêmes colonnes que lors du training
            input_data = pd.DataFrame({'SHARPE_FONDS': [ratio_val], 'TYPE': [default_type]})
            input_data = pd.get_dummies(input_data, columns=['TYPE'], prefix='TYPE')

            # Récupération de la référence des colonnes depuis un modèle entraîné
            # (ici on prend comme exemple la catégorie "AM company / Fund of Funds", adaptez si besoin)
            reference_cols = trained_models["AM company / Fund of Funds"]['X_test'].columns

            # Ajout de toute colonne manquante dans input_data
            for col in reference_cols:
                if col not in input_data.columns:
                    input_data[col] = 0

            # Tri des colonnes pour qu'elles correspondent exactement à l'ordre du modèle
            input_data = input_data[reference_cols]

            # Liste des catégories d’investisseurs
            categories = [
                "AM company / EdR Fund of Funds",
                "AM company / Fund of Funds",
                "AM company / Independant",
                "AM company / Own account",
                "Bank / EdR Private Banking",
                "Bank / Online Broker",
                "Bank / Own Account",
                "Bank / Private Banking",
                "Bank / Retail",
                "Bank / Trading Platform",
                "Independant Financial Advisor",
                "Insurance Company / Mutuelle",
                "Insurance Company / Own Account",
                "Insurance Company / Retirement Scheme",
                "Insurance Company / Unit Link",
                "Multi Familly Office / Wealth manager",
                "Pension Fund / Employee Saving Scheme",
                "Pension Fund / Other",
                "Single Family Office",
                "Sovereign / Pension Fund",
                "Unknown"
            ]

            # Récupération des prédictions pour chaque catégorie
            predictions = {}
            for cat in categories:
                pred = predict_category(cat, input_data)
                if pred is not None and len(pred) > 0:
                    predictions[cat] = pred[0]
                else:
                    predictions[cat] = None

            # Création d'un DataFrame avec les résultats
            df_pred = pd.DataFrame(list(predictions.items()), columns=["Catégorie d'investisseur", "Prédiction"])

            # Fonction pour mettre en couleur les prédictions
            def highlight_pred(val):
                if val == 1:
                    color = 'lightgreen'
                elif val == 0:
                    color = 'salmon'
                else:
                    color = 'lightgrey'
                return f'background-color: {color}'



        # Trier les données par la colonne 'Prédiction'
        df_pred_sorted = df_pred.sort_values(by='Prédiction', ascending=False)

        # Appliquer le style
        styled_df = df_pred_sorted.style.applymap(highlight_pred, subset=['Prédiction'])

        # Afficher la table dans Streamlit
        st.dataframe(styled_df, height=600)

    


## AVOIR POUR CHAQUE COLONNE DE RATIO UN DF SUR LA MOYENNE TOTALE DU RATIO SUR LES 4 ANNEES GLISSANTES