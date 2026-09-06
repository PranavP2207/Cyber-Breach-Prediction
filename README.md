# 🛡️ Cyber Breach Prediction Across the Enterprise Kill Chain

**Using Machine Learning**

---

## Overview

SOC analysts manually assess breach likelihood under time pressure, using scattered signals — a slow, error-prone process. This project builds a **supervised ML classifier** that ingests live defensive-posture and kill-chain telemetry and outputs a **real-time breach probability**, flagging high-risk incidents for immediate escalation.

**Core question:** Given an enterprise's defenses and how far an attacker has progressed through the Cyber Kill Chain, will this intrusion end in a **successful breach** or be **contained/mitigated**?

---

## Model at a Glance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest (150 trees, max depth 12) |
| Training data | 100,000 incidents (after cleaning) |
| Input features | 40 (enterprise profile + defenses + kill-chain telemetry) |
| Test accuracy | **86.4%** |
| F1-score (breach class) | **0.77** |
| ROC-AUC | **0.93** |

### Key Finding — Kill-Chain Depth is Highly Predictive

| Furthest Stage Reached | Breach Success Rate |
|------------------------|-------------------|
| Reconnaissance | 4.7% |
| Initial Access | 38.6% |
| Execution | 56.1% |
| Persistence | 73.3% |
| Impact | 93.8% |

---

## Project Structure

```
Cyber-Breach-Prediction/
│
├── app.py                      # 🚀 Streamlit dashboard (entry point)
├── requirements.txt            # Python dependencies
├── README.md
├── .gitignore
│
├── model/
│   └── cyber_model.joblib      # Trained scikit-learn pipeline (Random Forest)
│
├── frontend/
│   └── index.html              # Standalone HTML/CSS/JS prototype (client-side)
│
├── notebooks/
│   └── cyber-attacks-analysis.ipynb   # Full EDA + modeling pipeline
│
├── data/
│   └── Enterprise_Cyber_Kill_Chain_Dataset.csv   # Raw dataset (100,500 × 50)
│
└── docs/
    ├── project_brief.txt               # Technical hand-off document
    ├── aiml_lab_poster_16x9.png        # Academic poster (image)
    └── aiml_lab_poster_16x9.pptx       # Academic poster (editable)
```

| Folder | Contents |
|--------|----------|
| **`model/`** | Trained ML model (`cyber_model.joblib`) — loaded by `app.py` at runtime |
| **`frontend/`** | Standalone HTML prototype with a client-side breach calculator |
| **`notebooks/`** | Jupyter notebook with complete EDA, data cleaning, and model training |
| **`data/`** | Raw dataset used for training and analysis |
| **`docs/`** | Project brief, academic poster, and other documentation |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the dashboard

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### 3. Try the demo presets

Use the **"Quick scenario"** dropdown at the top of the dashboard:
- **Case A: Contained attack** — Reconnaissance stage, hardened defenses → LOW risk (~1%)
- **Case B: Critical breach** — Impact stage, weak defenses → CRITICAL risk (~90%)

---

## Data Cleaning & Leakage Removal

1. Removed 500 exact duplicate rows
2. **Dropped 7 post-breach leakage columns** (Recovery_Cost_USD, Records_Compromised, Downtime_Hours, Financial_Loss_USD, Cyber_Risk_Score, Risk_Level, Incident_Severity) — these are only known *after* a breach, so including them would inflate accuracy artificially
3. Imputed missing `Compliance` values as `"None"` (unregulated sectors)
4. Imputed missing `Detection_Time_Min` with the column median
5. Dropped `Incident_ID` and `Timestamp` (non-predictive / already decomposed)

---

## Tech Stack

- **Python** · pandas · NumPy · scikit-learn · matplotlib · seaborn
- **Streamlit** — interactive dashboard
- **joblib** — model persistence
- **HTML / CSS / JS** — standalone prototype

---

## Disclaimer

This is an academic decision-support prototype for demonstration purposes. It does not replace analyst judgment or an organization's incident response process.
