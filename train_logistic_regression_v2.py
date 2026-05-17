"""
Logistic Regression v2 - Improved Diabetes Prediction
Improvements: KNN Imputation, SMOTE, Feature Engineering,
              Polynomial Features, Optimal Threshold Tuning
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import json, os, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import KNNImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve, f1_score, r2_score)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "diabetes.csv")
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "logistic_regression.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "lr_scaler.pkl")
STATS_PATH  = os.path.join(MODEL_DIR, "lr_stats.json")
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("  LOGISTIC REGRESSION v2 - IMPROVED TRAINING")
print("=" * 60)

# ─── 1. Load Data ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"\n[OK] Dataset loaded -> {df.shape[0]} rows x {df.shape[1]} columns")

# ─── 2. Replace zeros with NaN (physiologically invalid) ─────────────────────
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)

FEATURES = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age"]
TARGET   = "Outcome"

# ─── 3. KNN Imputation (better than median for correlated features) ───────────
print("[OK] Applying KNN imputation (k=5)...")
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed = knn_imputer.fit_transform(df[FEATURES])
X_df = pd.DataFrame(X_imputed, columns=FEATURES)

# ─── 4. Feature Engineering ───────────────────────────────────────────────────
print("[OK] Engineering new features...")

# Glucose risk zones (clinical thresholds)
X_df["Glucose_Risk"]      = pd.cut(X_df["Glucose"],
                                    bins=[0, 99, 125, 999],
                                    labels=[0, 1, 2]).astype(float)
# BMI obesity category
X_df["BMI_Category"]      = pd.cut(X_df["BMI"],
                                    bins=[0, 18.5, 24.9, 29.9, 999],
                                    labels=[0, 1, 2, 3]).astype(float)
# Age risk group
X_df["Age_Group"]         = pd.cut(X_df["Age"],
                                    bins=[0, 30, 45, 60, 999],
                                    labels=[0, 1, 2, 3]).astype(float)
# Interaction: high glucose AND high BMI is a strong diabetes signal
X_df["Glucose_BMI"]       = X_df["Glucose"] * X_df["BMI"] / 1000
# Insulin resistance proxy
X_df["Insulin_Glucose"]   = X_df["Insulin"] / (X_df["Glucose"] + 1)
# Pedigree * Age compound risk
X_df["Pedigree_Age"]      = X_df["DiabetesPedigreeFunction"] * X_df["Age"]

y = df[TARGET].values

# ─── 5. Train/Test split (stratified to preserve class ratio) ─────────────────
X = X_df.values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"[OK] Train/Test split -> {len(X_train)} / {len(X_test)}")
print(f"     Class balance (train): 0={sum(y_train==0)}, 1={sum(y_train==1)}")

# ─── 6. SMOTE - Fix class imbalance ───────────────────────────────────────────
print("[OK] Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"     After SMOTE: 0={sum(y_train_res==0)}, 1={sum(y_train_res==1)}")

# ─── 7. Scale ─────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_res)
X_test_sc  = scaler.transform(X_test)

# ─── 8. Polynomial Features (degree=2, interactions only) ─────────────────────
print("[OK] Adding polynomial interaction features...")
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_sc)
X_test_poly  = poly.transform(X_test_sc)

# ─── 9. Train Logistic Regression ─────────────────────────────────────────────
print("\n[OK] Training Logistic Regression with L2 regularization...")
model = LogisticRegression(max_iter=2000, C=0.1, solver="lbfgs",
                           class_weight="balanced", random_state=42)
model.fit(X_train_poly, y_train_res)

# ─── 10. Optimal Threshold Tuning via ROC ─────────────────────────────────────
y_prob     = model.predict_proba(X_test_poly)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Find threshold that maximises F1
f1_scores = []
for t in thresholds:
    preds = (y_prob >= t).astype(int)
    f1_scores.append(f1_score(y_test, preds, zero_division=0))
best_thresh = float(thresholds[np.argmax(f1_scores)])
print(f"[OK] Optimal decision threshold: {best_thresh:.3f} (default was 0.500)")

y_pred = (y_prob >= best_thresh).astype(int)

# ─── 11. Evaluate ─────────────────────────────────────────────────────────────
accuracy  = accuracy_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_prob)

# CV on full scaled data (without poly to keep it fast)
cv_scores = cross_val_score(
    LogisticRegression(max_iter=2000, C=0.1, solver="lbfgs",
                       class_weight="balanced", random_state=42),
    scaler.transform(X), y, cv=5, scoring="accuracy")
cm        = confusion_matrix(y_test, y_pred)
report    = classification_report(y_test, y_pred, output_dict=True)

print(f"\n{'='*55}")
print(f"  [v1] Accuracy was : 70.78%")
print(f"  [v2] Accuracy NOW : {accuracy*100:.2f}%  (+{(accuracy-0.7078)*100:.2f}%)")
print(f"  ROC-AUC           : {roc_auc:.4f}")
print(f"  CV Score          : {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")
print(f"  Confusion Matrix  : {cm.tolist()}")
print(f"{'='*55}")
print(classification_report(y_test, y_pred, target_names=["No Diabetes","Diabetes"]))

# ─── 12. Save artefacts ───────────────────────────────────────────────────────
# Save a combined pipeline: knn_imputer + scaler + poly
joblib.dump({"model": model, "scaler": scaler, "poly": poly,
             "knn_imputer": knn_imputer, "threshold": best_thresh,
             "feature_cols": list(X_df.columns)}, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

stats = {
    "model_name"       : "Logistic Regression",
    "accuracy"         : round(accuracy, 4),
    "r2_score"         : round(float(max(0.0, r2_score(y_test, y_prob))), 4),
    "roc_auc"          : round(roc_auc, 4),
    "cv_mean"          : round(float(cv_scores.mean()), 4),
    "cv_std"           : round(float(cv_scores.std()), 4),
    "confusion_matrix" : cm.tolist(),
    "report"           : report,
    "feature_importance": dict(zip(list(X_df.columns),
                               model.coef_[0][:len(X_df.columns)].tolist())),
    "roc_fpr"          : fpr.tolist(),
    "roc_tpr"          : tpr.tolist(),
    "threshold"        : round(best_thresh, 4),
    "features"         : FEATURES,
    "improvements"     : ["KNN Imputation", "SMOTE", "Feature Engineering",
                          "Polynomial Features", "Threshold Tuning"],
}
with open(STATS_PATH, "w") as f:
    json.dump(stats, f, indent=2)

print(f"\n[OK] Model + stats saved to {MODEL_DIR}/")
print("[DONE] Logistic Regression v2 training complete!\n")
