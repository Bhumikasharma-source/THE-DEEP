# 🌊 The-DEEP: Machine Learning Water Quality Prediction

An intelligent environmental monitoring web application that predicts **Biochemical Oxygen Demand (B.O.D.)** and evaluates water pollution levels based on physical, chemical, and biological water parameters.

---

## 📌 Project Overview
* **Target Output**: Biochemical Oxygen Demand (B.O.D. in mg/L) & 5-tier pollution classification.
* **Core Algorithm**: **Random Forest Regressor** (`sklearn.ensemble.RandomForestRegressor`) with 100 estimators.
* **Dataset**: Indian Central Pollution Control Board (**CPCB**) National Water Quality Monitoring Programme (NWMP) dataset (`water_dataX_updated.csv`).
* **Interface**: Interactive **Streamlit** dashboard (`app.py`).
* **Notebook**: Complete EDA, data cleaning, feature engineering, and model training in `TISD.ipynb`.

---

## 🚀 Quick Start

### 1. Installation
```bash
cd The-DEEP
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 📖 Detailed Documentation
For comprehensive technical details, data preprocessing steps, feature importance weights, hyperparameter tuning, and architectural diagrams, refer to:
👉 **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)**
Maintained by Bhumika
