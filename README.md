# 🛡️ Cyber Breach Prediction Across the Enterprise Kill Chain

A machine learning system that predicts whether an in-progress cyber intrusion will result in a successful breach or be contained — using only pre-breach signals like defensive posture and kill-chain progression.

---

## 🔍 Problem Statement

Security Operations Center (SOC) analysts face constant pressure to assess breach likelihood in real time, often relying on scattered signals and manual judgment. This project automates that assessment with a **Random Forest classifier** trained on 100,000 enterprise incident records, enabling faster and more consistent threat prioritization.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **86.4%** |
| F1-Score (breach class) | **0.77** |
| ROC-AUC | **0.93** |
| Training Records | 100,000 |
| Input Features | 40 |

### Kill-Chain Stage vs. Breach Rate

The strongest predictor: the further an attacker progresses through the kill chain, the more likely the intrusion succeeds.

| Stage | Breach Rate |
|-------|------------|
| Reconnaissance | 4.7% |
| Initial Access | 38.6% |
| Execution | 56.1% |
| Persistence | 73.3% |
| Impact | 93.8% |

---

## 🗂️ Project Structure

```
├── app.py                          # Streamlit dashboard (entry point)
├── requirements.txt                # Python dependencies
│
├── model/
│   └── cyber_model.joblib          # Trained scikit-learn pipeline
│
├── frontend/
│   └── index.html                  # Standalone HTML/JS prototype
│
├── notebooks/
│   └── cyber-attacks-analysis.ipynb    # EDA + model training pipeline
│
└── data/
    └── Enterprise_Cyber_Kill_Chain_Dataset.csv
```

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/PranavP2207/Cyber-Breach-Prediction.git
cd Cyber-Breach-Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py
```

The app opens at **http://localhost:8501**. Use the **Quick scenario** dropdown to load demo presets:

- **Case A** — Contained attack (Reconnaissance, hardened defenses) → **LOW** risk
- **Case B** — Critical breach (Impact stage, weak defenses) → **CRITICAL** risk

---

## ⚙️ How It Works

1. **Data Cleaning** — Removed duplicates and dropped 7 post-breach columns (financial loss, downtime, etc.) to prevent data leakage
2. **Feature Engineering** — 40 features across enterprise profile, defensive controls, and kill-chain telemetry are one-hot encoded and fed into the model
3. **Training** — A Random Forest (150 trees, max depth 12) trained on an 80/20 stratified split, benchmarked against a Logistic Regression baseline
4. **Prediction** — The Streamlit dashboard loads the trained model and returns a live breach probability with a LOW / MEDIUM / HIGH / CRITICAL risk label

---

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Streamlit` · `Matplotlib` · `Seaborn` · `joblib`

---

## 📄 License

This project is for academic and educational purposes.
