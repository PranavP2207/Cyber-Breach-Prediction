# Cyber Breach Prediction Using Machine Learning

An academic Streamlit application that assesses incident telemetry with **Random Forest** and **Logistic Regression**. Enter 16 features or load one of 14 demonstration scenarios to compare both models' estimated probability of `Attack_Success = 1`.

The project demonstrates **incident classification**. Some inputs describe observed harm, such as encryption or exfiltration; prospective early-warning performance and real-world probability calibration have not been established.

## Quick start

Tested with **Python 3.11**, Streamlit 1.63.0 and scikit-learn 1.8.0. Use the versions in `requirements.txt` for compatibility with the saved pipelines.

```bash
git clone https://github.com/PranavP2207/Cyber-Breach-Prediction.git
cd Cyber-Breach-Prediction
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL shown by Streamlit, normally `http://localhost:8501`. A virtual environment is recommended. The included model files are ready to use: **training and notebook execution are not required to run the app**.

## What the app provides

- **Predictor:** grouped incident inputs, 14 editable scenarios, equally sized model cards with semicircle gauges, and an observed kill-chain timeline. Both predictions update when an input changes.
- **Model comparison:** holdout metrics, confusion matrices, grouped forest feature importance, and measured predictions for all scenarios.
- **JSON downloads:** save the current inputs and both predictions, or the stored evaluation and scenario comparison. Downloads do not train or alter the models.
- **Separate project guide:** open [docs/project-guide.html](docs/project-guide.html) in a browser for feature explanations, data flow, training details and worked calculations. It is not an app tab. Its tables and calculations work offline; Google Fonts fall back to system fonts.

The interface uses a dark teal theme, responsive cards and horizontal scrolling for the kill-chain stages on small screens. The timeline shows the selected stage; it does not infer that every earlier stage occurred.

## The 16 inputs

The ordered feature schema, defaults, explanations and scenario values live in [project_config.py](project_config.py).

| Feature | Meaning |
|---|---|
| `Attack_Stage` | Observed phase: Reconnaissance, Initial Access, Execution, Persistence or Impact. |
| `Company_Size` | Small, Medium or Large organization category. |
| `Lateral_Movement` | Whether movement between systems was observed. |
| `Privilege_Escalation` | Whether elevated access was obtained. |
| `Persistence` | Whether the attacker established continued access. |
| `Credential_Stolen` | Whether credential theft was observed. |
| `Data_Exfiltration_GB` | Recorded volume of data leaving the organization, in GB. |
| `Data_Encrypted` | Whether attacker-driven data encryption was observed. |
| `Phishing_Click` | Whether a phishing link was clicked. |
| `Detection_Time_Min` | Recorded delay before detection, in minutes. |
| `Response_Time_Min` | Recorded response delay, in minutes. |
| `Firewall` | Whether a firewall is enabled. |
| `MFA` | Whether multifactor authentication is enabled. |
| `EDR` | Whether endpoint detection and response is enabled. |
| `CVSS_Score` | Vulnerability severity on a 0–10 scale. |
| `Patch_Age_Days` | Recorded patch age in days. |

These are two categorical, nine binary and five numeric inputs. One-hot encoding produces eight categorical columns, giving **22 transformed features** with the remaining 14 inputs. Predictive importance describes associations in this dataset; it does not establish causal security effects.

## Data flow and models

```text
Training/evaluation:
CSV -> remove duplicates -> select 16 features + Attack_Success
    -> stratified 80/20 split (seed 42)
    -> training-only preprocessing for Logistic Regression
    -> evaluate both saved pipelines on the documented holdout
    -> model files + evaluation.json

Live assessment:
UI inputs / scenario -> ordered one-row DataFrame
    -> each saved pipeline's preprocessing -> predict_proba(class 1)
    -> two probabilities, risk bands and comparison
```

**Random Forest** uses 150 trees, maximum depth 12 and a minimum of five samples per leaf, with `class_weight=None`. Its probability is the average class-1 leaf probability across the trees. The existing forest artifact is retained; it is not retrained when the app starts.

**Logistic Regression** uses training-only median imputation, numeric/binary standardization and categorical one-hot encoding, followed by an L2-regularized classifier (`C=1`, `lbfgs`, `max_iter=1000`). It computes a learned linear score and converts it to a probability with `p = 1 / (1 + exp(-z))`.

The app checks both model files against the SHA-256 hashes in [model/evaluation.json](model/evaluation.json) and validates the input schema. Update the artifacts together after retraining.

## Measured comparison

The dataset has **100,500 rows and 50 columns**. Removing 500 duplicates leaves 100,000 unique rows, split into 80,000 training and 20,000 test rows. The breach prevalence is 29.406%; majority-class test accuracy is 70.595%. There are 3,147 missing detection times among the unique rows; the training median is 103 minutes.

Both models are evaluated on the same **reconstructed documented holdout**, with a binary decision threshold of 0.5.

| Metric | Random Forest | Logistic Regression |
|---|---:|---:|
| Accuracy | 86.22% | 86.01% |
| Breach precision | 76.06% | 76.87% |
| Breach recall | 77.54% | 74.95% |
| Breach F1 | 0.7679 | 0.7590 |
| ROC-AUC | 0.9358 | 0.9334 |
| Average precision | 0.8460 | 0.8419 |
| Brier score (lower is better) | 0.0943 | 0.0963 |

The forest identifies 152 more true breaches and produces 109 more false positives on this split. Its accuracy advantage is **0.215 percentage points**; statistical significance has not been established. Logistic Regression has slightly higher precision. A higher probability for an unlabelled demonstration scenario does not prove a more accurate prediction.

The exact results, confusion matrices, environment versions, artifact hashes and all scenario outputs are stored in `model/evaluation.json`.

## Demonstration scenarios

The 14 scenarios retain the original A–D cases and add a reference incident, credentials/escalation, lateral movement, all defenses off, EDR off, MFA off, phishing, fast/slow response and persistence without observed encryption or exfiltration.

Display bands are **LOW <25%**, **MEDIUM 25–<50%**, **HIGH 50–<75%**, and **CRITICAL ≥75%**. These are illustrative UI bands, not validated operating thresholds.

Disabling EDR lowers the learned score in one reference scenario. This counterintuitive association is a model/data limitation, not evidence that disabling a control improves safety. Toggle comparisons are sensitivity examples rather than causal experiments.

## Repository contents

```text
Cyber-Breach-Prediction/
├── app.py                          # Streamlit entry point
├── project_config.py               # Schema, defaults, scenarios and explanations
├── requirements.txt                # App/training dependencies
├── requirements-notebook.txt       # Additional analysis libraries
├── retrain_model.py                # Training and measured evaluation
├── build_guide.py                  # Standalone guide generator
├── .streamlit/config.toml          # Visual theme
├── model/
│   ├── cyber_model.joblib           # Saved Random Forest pipeline
│   ├── logistic_model.joblib        # Saved Logistic Regression pipeline
│   └── evaluation.json             # Metrics, provenance, hashes and case outputs
├── data/
│   └── Enterprise_Cyber_Kill_Chain_Dataset.csv
├── notebooks/
│   ├── cyber-attacks-analysis.ipynb
│   └── cyber-attacks-analysis-16f.ipynb
├── docs/project-guide.html
├── .gitignore
└── README.md
```

Inference requires the app/configuration, runtime dependencies and the three files in `model/`. The dataset, notebooks and training script support academic analysis and reproduction. Local backups, draft reports, video production files, review evidence and editor/context notes are excluded from this release.

## Reproduce or update the evaluation

These commands **write new artifacts**; use a separate checkout if you want to preserve the bundled release.

```bash
python retrain_model.py
python build_guide.py
```

The default training command fits and saves Logistic Regression, evaluates both models and updates `evaluation.json`, while retaining the existing forest. Rebuild the guide after any model/evaluation update.

Optional training modes:

```bash
python retrain_model.py --cv 5          # Stratified training-fold ROC-AUC comparison
python retrain_model.py --retrain-forest
python build_guide.py
```

`--retrain-forest` preserves a copy of the previous forest in the ignored local archive before replacement and fits preprocessing on training rows only. The bundled evaluation has no recorded cross-validation run; the option is available for future analysis. Keep final test data out of hyperparameter, calibration and threshold selection.

For notebook analysis:

```bash
python -m pip install -r requirements-notebook.txt
```

Both notebooks are preserved as historical references. `cyber-attacks-analysis.ipynb` contains executed EDA and an older 40-feature experiment, which is not the current app model; its data-loading/export paths need local adjustment. `cyber-attacks-analysis-16f.ipynb` records the reduced design but has no saved execution outputs and uses historical pre-split imputation. The current training/evaluation implementation is `retrain_model.py`. Avoid exporting a historical notebook over the active model files.

## Limitations and next steps

- Define a prediction-time cutoff before making an early-warning claim. Encryption, exfiltration and final response timing may be unavailable at that point.
- The original forest's fit-time lineage is not embedded in its artifact. Its documented split is reconstructed for comparison. Historical imputation used the full dataset; the full-data and training medians happen to be equal here.
- Verify dataset/label provenance and investigate inconsistent stage/control associations before claiming representative enterprise performance.
- Add training-fold cross-validation, confidence intervals, probability calibration, feature ablations and cost-based threshold selection before operational use.

The project is intended for academic demonstration and education. It has not been validated for real-world security decisions.
