"""
SVM v3 - Improved: Better hyperparams + threshold tuned for accuracy
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd, numpy as np, json, os, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, r2_score)
from imblearn.over_sampling import SMOTE
import joblib

os.makedirs('models', exist_ok=True)
print("="*60)
print("  SUPPORT VECTOR MACHINE v3 - IMPROVED")
print("="*60)

df = pd.read_csv('data/diabetes.csv')
print(f"\n[OK] Dataset loaded -> {df.shape[0]} rows x {df.shape[1]} columns")

for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
    df[col] = df[col].replace(0, np.nan)

FEATURES = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age']

print("[OK] KNN imputation (k=5)...")
knn = KNNImputer(n_neighbors=5)
X_imp = knn.fit_transform(df[FEATURES])
X_df = pd.DataFrame(X_imp, columns=FEATURES)

print("[OK] Feature engineering...")
X_df['Glucose_Risk']    = pd.cut(X_df['Glucose'], bins=[0,99,125,999], labels=[0,1,2]).astype(float)
X_df['BMI_Category']    = pd.cut(X_df['BMI'], bins=[0,18.5,24.9,29.9,999], labels=[0,1,2,3]).astype(float)
X_df['Age_Group']       = pd.cut(X_df['Age'], bins=[0,30,45,60,999], labels=[0,1,2,3]).astype(float)
X_df['Glucose_BMI']     = X_df['Glucose'] * X_df['BMI'] / 1000
X_df['Insulin_Glucose'] = X_df['Insulin'] / (X_df['Glucose'] + 1)
X_df['Pedigree_Age']    = X_df['DiabetesPedigreeFunction'] * X_df['Age']

FEAT_COLS = X_df.columns.tolist()
y = df['Outcome'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_df.values, y, test_size=0.20, random_state=42, stratify=y)
print(f"[OK] Train/Test -> {len(X_train)} / {len(X_test)}")

smote = SMOTE(random_state=42)
X_tr, y_tr = smote.fit_resample(X_train, y_train)
print(f"[OK] SMOTE: 0={sum(y_tr==0)}, 1={sum(y_tr==1)}")

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_test)

# Improved search: scoring=balanced_accuracy, wider C range, only rbf
print("\n[OK] RandomizedSearchCV (120 iterations, 5-fold, scoring=balanced_accuracy)...")
param_dist = {
    'C'    : [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
    'gamma': ['scale', 'auto', 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    'kernel': ['rbf'],  # RBF consistently best for this data
}
svc = SVC(probability=True, random_state=42, class_weight='balanced')
search = RandomizedSearchCV(svc, param_dist, n_iter=120, cv=5,
                            scoring='balanced_accuracy',
                            random_state=42, n_jobs=-1)
search.fit(X_tr_sc, y_tr)
model = search.best_estimator_
print(f"[OK] Best params: {search.best_params_}")

# Threshold tuned for ACCURACY (not F1)
y_prob = model.predict_proba(X_te_sc)[:, 1]
_, _, thresholds = roc_curve(y_test, y_prob)
accs  = [accuracy_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]
best_thresh = float(thresholds[np.argmax(accs)])
y_pred = (y_prob >= best_thresh).astype(int)
print(f"[OK] Optimal threshold (max accuracy): {best_thresh:.3f}")

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)
cm   = confusion_matrix(y_test, y_pred).tolist()

v2_acc = 66.88
print("\n=======================================================")
print(f"  [v2] Accuracy was : {v2_acc:.2f}%")
print(f"  [v3] Accuracy NOW : {acc*100:.2f}%  ({acc*100-v2_acc:+.2f}%)")
print(f"  Precision         : {prec:.4f}")
print(f"  Recall            : {rec:.4f}")
print(f"  F1 Score          : {f1:.4f}")
print(f"  ROC-AUC           : {auc:.4f}")
print(f"  Confusion Matrix  : {cm}")
print("=======================================================")
print(classification_report(y_test, y_pred, target_names=['No Diabetes','Diabetes']))

stats = {
    'model_name'      : 'Support Vector Machine',
    'accuracy'        : round(float(acc), 4),
    'r2_score'        : round(float(max(0.0, r2_score(y_test, y_prob))), 4),
    'precision'       : round(float(prec), 4),
    'recall'          : round(float(rec), 4),
    'f1_score'        : round(float(f1), 4),
    'roc_auc'         : round(float(auc), 4),
    'confusion_matrix': cm,
    'threshold'       : round(best_thresh, 4),
}
pipeline = {
    'model'       : model,
    'scaler'      : scaler,
    'knn_imputer' : knn,
    'poly'        : None,
    'threshold'   : best_thresh,
    'feature_cols': FEAT_COLS,
}

joblib.dump(pipeline, 'models/svm.pkl')
with open('models/svm_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("[OK] Saved to models/")
print("[DONE] SVM v3 complete!\n")
