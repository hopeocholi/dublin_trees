# Predicting Urban Tree Health with Machine Learning

**A multi-class classification system that flags at-risk public trees in Dublin, enabling proactive inspection scheduling and safer streets.**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)
![Domain](https://img.shields.io/badge/Domain-Smart%20City%20%7C%20Public%20Sector-2e7d32.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## Overview

Urban tree management is a growing responsibility for city governments. Unattended trees raise maintenance costs and public-safety risk: branches break, and ageing or structurally compromised trees can fall. Dublin City Council currently relies on reactive reporting and manual inspection scheduling, which is slow and resource-intensive.

This project builds a predictive model that classifies the **health condition** of a public tree from attributes describing its species, age, size, and surroundings. By flagging trees likely to be in poor condition, the council can prioritise inspections, cut emergency maintenance spend, and improve public safety, a concrete example of predictive analytics applied to smart-city asset management.

The task is framed as a **multi-class classification** problem on real, open Dublin City Council tree data.

---

## Key Results

Two models were trained on an identical preprocessing pipeline and evaluated with **accuracy** and **F1-macro** (the appropriate headline metric for imbalanced multi-class data, since it weights every class equally).

| Model | Accuracy | F1-macro |
|---|---|---|
| Logistic Regression (baseline) | 0.803 | 0.399 |
| **Random Forest (final)** | 0.745 | **0.486** |

Although Logistic Regression posted higher raw accuracy, that figure is inflated by the dominant "Fair" class. On the metric that matters for imbalanced data, **Random Forest lifted F1-macro by roughly 22%** and delivered materially better recall on minority classes such as *Good*, *Poor*, *Dead*, and *Monolith*, exactly the trees the council most needs to catch. This trade-off is the central finding of the project.

---

## Dataset

- **Source:** [Dublin City Council Tree Dataset](https://data.smartdublin.ie/dataset/tree-dcc) (Ireland Open Data / Smart Dublin)
- **Size:** 15,962 records, one per council-managed tree
- **Target:** `Condition` (tree health status, multi-class)
- **Key features:** Species, Age group, Height range, Spread range, Stem diameter range, Proximity to infrastructure, Area

The raw data is tabular and mixes categorical fields with text-encoded numeric ranges (for example `"5 to 10 m"`). Several fields carry substantial missingness (Area, Proximity, and BuildingNumber are the worst affected).

> The dataset is **not** committed to this repo. Download it from the link above and place `total-tree-population.csv` in `data/`. See [`data/README.md`](data/README.md).

---

## Approach

**1. Exploratory Data Analysis**
Confirmed a strong class imbalance in the target, a small number of species dominating the population, and revealed the distribution of age groups, proximity categories, and size ranges. These findings drove the preprocessing and metric choices.

**2. Feature Engineering**
Text range fields (`Height`, `Spread`, `StemDiameter`) were parsed into numeric **midpoint** values so the models could learn from continuous size signals. Rare `Condition` categories were collapsed into a single `Other` class to reduce noise and stabilise training.

**3. Preprocessing Pipeline** (shared across both models for a fair comparison)
- Numerical features: median imputation, then standard scaling
- Categorical features: most-frequent imputation, then one-hot encoding
- Wrapped in a single scikit-learn `ColumnTransformer` + `Pipeline`

**4. Modelling**
- **Logistic Regression** as an interpretable, efficient baseline (`max_iter=1000`)
- **Random Forest** with 200 trees and balanced class weighting to counter imbalance
- 80/20 train/test split with **stratified sampling** to preserve class proportions

**5. Evaluation**
Both models compared on accuracy and F1-macro, with full per-class classification reports.

---

## Repository Structure

```
dublin-tree-health/
├── README.md                     # You are here
├── notebooks/
│   └── dublin_tree_health_classification.ipynb   # Full analysis, EDA to evaluation
├── src/
│   └── tree_health_pipeline.py   # Reusable, script-form version of the pipeline
├── reports/
│   └── Executive_Summary.pdf     # Business-facing write-up
├── data/
│   └── README.md                 # How to obtain the dataset
├── images/                       # Exported figures (generated on run)
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## Getting Started

```bash
# 1. Clone
git clone https://github.com/<your-username>/dublin-tree-health.git
cd dublin-tree-health

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the data
#    Download total-tree-population.csv from the Smart Dublin link above
#    and place it in the data/ folder.

# 4a. Run the notebook
jupyter notebook notebooks/dublin_tree_health_classification.ipynb

# 4b. Or run the script version end-to-end
python src/tree_health_pipeline.py --data data/total-tree-population.csv
```

---

## Business Impact

A deployed version of this model would let Dublin City Council:

- **Prioritise inspections** toward trees flagged as likely-poor rather than working a manual queue
- **Reduce emergency maintenance costs** by catching decline before failure
- **Improve public safety** through earlier intervention on high-risk trees

It demonstrates that machine learning can serve as practical decision support for smart-city asset management.

---

## Future Work

- Incorporate **geospatial and weather data** to capture environmental drivers of tree health
- Test **gradient boosting** (XGBoost / LightGBM) for further accuracy gains
- Apply **cost-sensitive learning** to prioritise high-risk misclassifications
- Wrap the model in a **real-time decision-support tool** for inspection teams

---

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib` · `seaborn` · `Jupyter`

---

## License

Released under the MIT License. See [`LICENSE`](LICENSE).

*Data © Dublin City Council, provided under the Smart Dublin open data terms.*
