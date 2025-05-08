## 📊 Analyse de la Performance des Fonds Edmond de Rothschild

Ce projet a pour objectif d’analyser et de visualiser les performances des fonds d’investissement d’**Edmond de Rothschild**, à travers des indicateurs financiers calculés sur des **années glissantes**, afin de mieux comprendre l’évolution des fonds et leur gestion dans le temps.

### 🔍 Objectifs

*  Suivre l’évolution mensuelle de ratios clés (Sharpe, Alpha, Volatilité, etc.) sur des périodes mobiles de 12 mois.
*  Visualiser l’historique de performance par type d’action ou de fonds.
*  Intégrer des analyses prédictives pour simuler les trajectoires futures des positions.
*  Construire des dashboards interactifs pour faciliter l’interprétation des résultats.

### 🧩 Données utilisées

* **Positions journalières des fonds**
* **Historique de performance des fonds et des benchmarks**
* **Données de passifs et flux de souscriptions/rachats**
* **Données de marché : taux, devises, indices de référence**

### 🛠️ Fonctions et outils développés

* Attribution des années glissantes (Septembre à Août).
* Moyennes mensuelles calculées par ratio et par gliding year.
* Visualisation interactive via **Plotly** et **Streamlit**.
* Segmentation automatique des colonnes par catégories (`1`, `2`, etc.) pour une organisation claire dans le dashboard.
* Intégration possible de modèles prédictifs (LSTM) pour la prévision de mouvements de position.


### 🚀 Technologies

* Python (Pandas, NumPy, Matplotlib, Plotly)
* Streamlit pour l’interface utilisateur
* Scikit-learn pour la modélisation
* Git pour le versioning

### Captures d'écran 
*![image](https://github.com/user-attachments/assets/efefd1b7-5238-4393-babf-66514ed65dfb)*
*![image](https://github.com/user-attachments/assets/ac1d06a8-7312-404d-b350-f01ccd782288)*
