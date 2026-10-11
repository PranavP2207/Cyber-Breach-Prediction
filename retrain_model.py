"""Reproduce the 16-feature comparison without replacing the existing forest by default.

python retrain_model.py                  # fit LR, evaluate both, retain current RF
python retrain_model.py --retrain-forest # fit both with train-only preprocessing
python retrain_model.py --cv 5           # optional stratified CV on training rows
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from project_config import (FEATURE_COLUMNS, CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    PRESETS, CASE_NOTES, KILL_CHAIN_STAGES)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data/Enterprise_Cyber_Kill_Chain_Dataset.csv'
MODEL_PATH = ROOT / 'model/cyber_model.joblib'
LR_PATH = ROOT / 'model/logistic_model.joblib'
EVAL_PATH = ROOT / 'model/evaluation.json'
LEAKAGE_COLS = ['Recovery_Cost_USD', 'Records_Compromised', 'Downtime_Hours',
    'Financial_Loss_USD', 'Cyber_Risk_Score', 'Risk_Level', 'Incident_Severity']


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def preprocessor(scale=False):
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale:
        steps.append(('scaler', StandardScaler()))
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), CATEGORICAL_FEATURES),
        ('num', Pipeline(steps), NUMERICAL_FEATURES),
    ])


def scores(model, X, y):
    probability = model.predict_proba(X)[:, list(model.classes_).index(1)]
    predicted = (probability >= .5).astype(int)
    return {
        'accuracy': float(accuracy_score(y, predicted)),
        'precision': float(precision_score(y, predicted, zero_division=0)),
        'recall': float(recall_score(y, predicted, zero_division=0)),
        'f1': float(f1_score(y, predicted, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, probability)),
        'average_precision': float(average_precision_score(y, probability)),
        'brier': float(brier_score_loss(y, probability)),
        'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def tier(p):
    return 'LOW' if p < .25 else 'MEDIUM' if p < .5 else 'HIGH' if p < .75 else 'CRITICAL'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retrain-forest', action='store_true')
    parser.add_argument('--cv', type=int, default=0, choices=[0, 3, 5, 10])
    args = parser.parse_args()
    raw = pd.read_csv(DATA_PATH)
    df = raw.drop_duplicates().copy()
    X, y = df[FEATURE_COLUMNS], df['Attack_Success']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    medians = X_train[NUMERICAL_FEATURES].median()
    print(f'Rows: {len(raw):,} raw -> {len(df):,} unique; {len(X_train):,} train / {len(X_test):,} test', flush=True)
    lr = Pipeline([
        ('preprocessor', preprocessor(scale=True)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42)),
    ]).fit(X_train, y_train)
    if args.retrain_forest:
        rf = Pipeline([
            ('preprocessor', preprocessor()),
            ('classifier', RandomForestClassifier(n_estimators=150, max_depth=12,
                min_samples_leaf=5, random_state=42, n_jobs=-1, class_weight=None)),
        ]).fit(X_train, y_train)
        # Retain the previous artifact before any explicitly requested retraining.
        backup = ROOT / 'archive/backups/cyber_model.before-retrain.joblib'
        if MODEL_PATH.exists() and not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(MODEL_PATH.read_bytes())
        joblib.dump(rf, MODEL_PATH, compress=3)
    else:
        rf = joblib.load(MODEL_PATH)
    if list(rf.feature_names_in_) != FEATURE_COLUMNS:
        raise ValueError('Saved forest does not accept the exact 16-feature schema.')
    # Legacy RF has no imputer. Supply training-only medians for its evaluation;
    # complete UI rows require no imputation. New pipelines impute internally.
    rf_test = X_test.copy()
    rf_test[NUMERICAL_FEATURES] = rf_test[NUMERICAL_FEATURES].fillna(medians)
    models = {'Random Forest': rf, 'Logistic Regression': lr}
    evaluation = {name: scores(model, rf_test if name == 'Random Forest' else X_test, y_test)
                  for name, model in models.items()}
    ohe = rf.named_steps['preprocessor'].named_transformers_['cat']
    weights = rf.named_steps['classifier'].feature_importances_
    importance, offset = {}, 0
    for feature, categories in zip(CATEGORICAL_FEATURES, ohe.categories_):
        importance[feature] = float(weights[offset:offset + len(categories)].sum())
        offset += len(categories)
    importance.update({feature: float(weight) for feature, weight in zip(NUMERICAL_FEATURES, weights[offset:])})
    cases = []
    for name, row in PRESETS.items():
        if row is None:
            continue
        frame = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        output = {}
        for model_name, model in models.items():
            p = float(model.predict_proba(frame)[0, list(model.classes_).index(1)])
            output[model_name] = {'probability': p, 'tier': tier(p), 'class_at_0_5': int(p >= .5)}
        cases.append({'name': name, 'note': CASE_NOTES[name[0]], 'features': row, 'outputs': output})
    cv = {}
    if args.cv:
        folds = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
        # Fresh pipelines fit imputation/encoding within each fold. Keep outer
        # parallelism at one; the forest already parallelizes its trees.
        fresh_rf = Pipeline([('preprocessor', preprocessor()),
            ('classifier', RandomForestClassifier(n_estimators=150, max_depth=12,
                min_samples_leaf=5, random_state=42, n_jobs=-1, class_weight=None))])
        for name, model in {'Random Forest': fresh_rf, 'Logistic Regression': lr}.items():
            values = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=folds, n_jobs=1)
            cv[name] = {'folds': args.cv, 'roc_auc_mean': float(values.mean()),
                'roc_auc_std': float(values.std()), 'fold_scores': values.tolist()}
    LR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr, LR_PATH, compress=3)
    metadata = {
        'protocol': 'Deduplicate before an 80/20 stratified split, random_state=42. LR preprocessing fits on training rows only. Both scored on the same reconstructed documented 20,000-row holdout at threshold 0.5.',
        'forest_provenance': ('Retrained by this script with train-only imputation.' if args.retrain_forest else
            'Existing 16-feature artifact retained byte-for-byte. Its documented split is reconstructed; original fit-time lineage is not embedded in the artifact. Original notebook/script imputed before splitting; full-data and training medians both equal 103 here.'),
        'raw_rows': len(raw), 'unique_rows': len(df), 'duplicates': len(raw)-len(df),
        'raw_columns': len(raw.columns), 'train_rows': len(X_train), 'test_rows': len(X_test),
        'test_positive_rows': int(y_test.sum()), 'positive_rate': float(y.mean()),
        'missing_detection_unique': int(X['Detection_Time_Min'].isna().sum()),
        'detection_train_median': float(medians['Detection_Time_Min']),
        'feature_order': FEATURE_COLUMNS, 'encoded_features': len(weights),
        'excluded_outcome_columns': LEAKAGE_COLS,
        'metrics': evaluation, 'grouped_importance': importance,
        'stage_rates': {stage: float(df.loc[df['Attack_Stage']==stage, 'Attack_Success'].mean()) for stage in KILL_CHAIN_STAGES},
        'majority_baseline_accuracy': float((y_test == 0).mean()),
        'cases': cases, 'cross_validation': cv,
        'forest_parameters': {key: rf.named_steps['classifier'].get_params()[key] for key in
            ['n_estimators','max_depth','min_samples_leaf','class_weight','random_state']},
        'logistic_parameters': {'max_iter': 1000, 'random_state': 42, 'iterations': int(lr.named_steps['classifier'].n_iter_[0]), 'scaling': 'StandardScaler on the 14 numeric/binary columns'},
        'artifacts': {p.name: sha256(p) for p in [MODEL_PATH, LR_PATH, DATA_PATH]},
        'versions': {'python': platform.python_version(), 'scikit_learn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__},
    }
    EVAL_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    for name, result in evaluation.items():
        print(name, json.dumps(result), flush=True)
    print('Saved LR and evaluation.json; existing RF retained.' if not args.retrain_forest else 'Saved both models and evaluation.json.', flush=True)


if __name__ == '__main__':
    main()
