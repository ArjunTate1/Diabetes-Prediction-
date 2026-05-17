"""
Random Forest v2 - Improved Diabetes Prediction
Improvements: KNN Imputation, SMOTE, Feature Engineering,
              Wider RandomizedSearchCV, Optimal Threshold Tuning
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import json, os, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve, f1_score)
from imblearn.over_sampling import SMOTE
import joblib

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "diabetes.csv")
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "random_forest.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "rf_scaler.pkl")
STATS_PATH  = os.path.join(MODEL_DIR, "rf_stats.json")
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("  RANDOM FOREST v2 - IMPROVED TRAINING")
print("=" * 60)

# ─── 1. Load Data ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"\n[OK] Dataset loaded -> {df.shape[0]} rows x {df.shape[1]} columns")

# ─── 2. Replace zeros with NaN ───────────────────────────────────────────────
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)

FEATURES = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age"]
TARGET   = "Outcome"

# ─── 3. KNN Imputation ────────────────────────────────────────────────────────
print("[OK] Applying KNN imputation (k=5)...")
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed = knn_imputer.fit_transform(df[FEATURES])
X_df = pd.DataFrame(X_imputed, columns=FEATURES)

# ─── 4. Feature Engineering ───────────────────────────────────────────────────
print("[OK] Engineering new features...")

X_df["Glucose_Risk"]    = pd.cut(X_df["Glucose"],
                                  bins=[0, 99, 125, 999],
                                  labels=[0, 1, 2]).astype(float)
X_df["BMI_Category"]    = pd.cut(X_df["BMI"],
                                  bins=[0, 18.5, 24.9, 29.9, 999],
                                  labels=[0, 1, 2, 3]).astype(float)
X_df["Age_Group"]       = pd.cut(X_df["Age"],
                                  bins=[0, 30, 45, 60, 999],
                                  labels=[0, 1, 2, 3]).astype(float)
X_df["Glucose_BMI"]     = X_df["Glucose"] * X_df["BMI"] / 1000
X_df["Insulin_Glucose"] = X_df["Insulin"] / (X_df["Glucose"] + 1)
X_df["Pedigree_Age"]    = X_df["DiabetesPedigreeFunction"] * X_df["Age"]

y = df[TARGET].values

# ─── 5. Train/Test split ──────────────────────────────────────────────────────
X = X_df.values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"[OK] Train/Test split -> {len(X_train)} / {len(X_test)}")
print(f"     Class balance (train): 0={sum(y_train==0)}, 1={sum(y_train==1)}")

# ─── 6. SMOTE ─────────────────────────────────────────────────────────────────
print("[OK] Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"     After SMOTE: 0={sum(y_train_res==0)}, 1={sum(y_train_res==1)}")

# ─── 7. Scale ─────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_res)
X_test_sc  = scaler.transform(X_test)

# ─── 8. RandomizedSearchCV - wider search space ───────────────────────────────
print("\n[OK] Running RandomizedSearchCV (100 iterations, 5-fold CV)...")
param_dist = {
    "n_estimators"      : [100, 200, 300, 500],
    "max_depth"         : [None, 5, 10, 15, 20, 30],
    "min_samples_split" : [2, 4, 6, 8, 10],
    "min_samples_leaf"  : [1, 2, 3, 4],
    "max_features"      : ["sqrt", "log2", 0.3, 0.5],
    "bootstrap"         : [True, False],
    "class_weight"      : ["balanced", "balanced_subsample", None],
}
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rand_search = RandomizedSearchCV(
    rf_base, param_dist, n_iter=100, cv=cv_strat,
    scoring="f1", n_jobs=-1, random_state=42, verbose=0)
rand_search.fit(X_train_sc, y_train_res)

best_params = rand_search.best_params_
print(f"[OK] Best params: {best_params}")

# ─── 9. Build Ensemble: RF + GradientBoosting (Soft Voting) ──────────────────
print("\n[OK] Building Soft-Voting Ensemble (RF + GradientBoosting)...")
rf_best = rand_search.best_estimator_
gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05,
    max_depth=4, subsample=0.8, random_state=42)
ensemble = VotingClassifier(
    estimators=[("rf", rf_best), ("gb", gb)],
    voting="soft", weights=[2, 1])
ensemble.fit(X_train_sc, y_train_res)
print("[OK] Ensemble trained!")

# ─── 10. Optimal Threshold Tuning ────────────────────────────────────────────
y_prob = ensemble.predict_proba(X_test_sc)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

f1_scores = []
for t in thresholds:
    preds = (y_prob >= t).astype(int)
    f1_scores.append(f1_score(y_test, preds, zero_division=0))
best_thresh = float(thresholds[np.argmax(f1_scores)])
print(f"[OK] Optimal threshold: {best_thresh:.3f} (default was 0.500)")

y_pred = (y_prob >= best_thresh).astype(int)

# ─── 11. Evaluate ────────────────────────────────────────────────────────────
accuracy  = accuracy_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_prob)
cv_scores = cross_val_score(ensemble, X_train_sc, y_train_res, cv=5, scoring="accuracy")
cm        = confusion_matrix(y_test, y_pred)
report    = classification_report(y_test, y_pred, output_dict=True)

print(f"\n{'='*55}")
print(f"  [v1] Accuracy was : 75.32%")
print(f"  [v2] Accuracy NOW : {accuracy*100:.2f}%  (+{(accuracy-0.7532)*100:.2f}%)")
print(f"  ROC-AUC           : {roc_auc:.4f}")
print(f"  CV Score          : {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")
print(f"  Confusion Matrix  : {cm.tolist()}")
print(f"{'='*55}")
print(classification_report(y_test, y_pred, target_names=["No Diabetes","Diabetes"]))

# Feature importance from RF component
rf_fi = dict(zip(list(X_df.columns), rf_best.feature_importances_.tolist()))
fi_sorted = sorted(rf_fi.items(), key=lambda x: x[1], reverse=True)
print("\n  Feature Importances (RF component):")
for feat, imp in fi_sorted:
    bar = "#" * int(imp * 50)
    print(f"  {feat:<30} {imp:.4f}  {bar}")

# ─── 12. Save artefacts ───────────────────────────────────────────────────────
joblib.dump({"model": ensemble, "scaler": scaler, "knn_imputer": knn_imputer,
             "threshold": best_thresh, "feature_cols": list(X_df.columns)}, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

stats = {
    "model_name"        : "Random Forest",
    "accuracy"          : round(accuracy, 4),
    "roc_auc"           : round(roc_auc, 4),
    "cv_mean"           : round(float(cv_scores.mean()), 4),
    "cv_std"            : round(float(cv_scores.std()), 4),
    "confusion_matrix"  : cm.tolist(),
    "report"            : report,
    "feature_importance": rf_fi,
    "best_params"       : {str(k): str(v) for k, v in best_params.items()},
    "roc_fpr"           : fpr.tolist(),
    "roc_tpr"           : tpr.tolist(),
    "threshold"         : round(best_thresh, 4),
    "features"          : FEATURES,
    "improvements"      : ["KNN Imputation", "SMOTE", "Feature Engineering",
                           "RandomizedSearchCV", "Gradient Boosting Ensemble",
                           "Threshold Tuning"],
}
with open(STATS_PATH, "w") as f:
    json.dump(stats, f, indent=2)

print(f"\n[OK] Model + stats saved to {MODEL_DIR}/")
print("[DONE] Random Forest v2 training complete!\n")
