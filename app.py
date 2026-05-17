"""
Flask Web App - Diabetes Prediction: 4-Model Comparison Dashboard
Models: Logistic Regression | SVM | Random Forest | Gradient Boosting
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from flask import Flask, request, jsonify, render_template
import joblib, json, numpy as np, os, pandas as pd

app = Flask(__name__)
MODEL_DIR = "models"

BASE_FEATURES = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                 "Insulin","BMI","DiabetesPedigreeFunction","Age"]

def engineer_features(df):
    df = df.copy()
    df["Glucose_Risk"]    = pd.cut(df["Glucose"], bins=[0,99,125,999], labels=[0,1,2]).astype(float)
    df["BMI_Category"]    = pd.cut(df["BMI"], bins=[0,18.5,24.9,29.9,999], labels=[0,1,2,3]).astype(float)
    df["Age_Group"]       = pd.cut(df["Age"], bins=[0,30,45,60,999], labels=[0,1,2,3]).astype(float)
    df["Glucose_BMI"]     = df["Glucose"] * df["BMI"] / 1000
    df["Insulin_Glucose"] = df["Insulin"] / (df["Glucose"] + 1)
    df["Pedigree_Age"]    = df["DiabetesPedigreeFunction"] * df["Age"]
    return df

MODEL_CONFIG = [
    ("lr",  "Logistic Regression",    "logistic_regression.pkl", "lr_stats.json"),
    ("svm", "Support Vector Machine", "svm.pkl",                 "svm_stats.json"),
    ("rf",  "Random Forest",          "random_forest.pkl",       "rf_stats.json"),
    ("gb",  "Gradient Boosting",      "gradient_boosting.pkl",   "gb_stats.json"),
]

def load_models():
    pipelines, stats = {}, {}
    for key, name, pkl, stat_file in MODEL_CONFIG:
        try:
            raw = joblib.load(os.path.join(MODEL_DIR, pkl))
            if isinstance(raw, dict):
                pipelines[key] = raw
            else:
                scaler_file = f"{key}_scaler.pkl"
                scaler = joblib.load(os.path.join(MODEL_DIR, scaler_file))
                pipelines[key] = {"model": raw, "scaler": scaler,
                                  "knn_imputer": None, "poly": None,
                                  "threshold": 0.5, "feature_cols": BASE_FEATURES}
            with open(os.path.join(MODEL_DIR, stat_file)) as f:
                stats[key] = json.load(f)
            print(f"[OK] {name} loaded")
        except Exception as e:
            print(f"[ERR] {name}: {e}")
    return pipelines, stats

pipelines, stats = load_models()

@app.route("/")
def index():
    return render_template("index.html", stats=stats)

@app.route("/stats")
def get_stats():
    return jsonify(stats)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        row = {f: float(data.get(f, 0)) for f in BASE_FEATURES}
        df_in = pd.DataFrame([row])

        result = {}
        for key, label, _, _ in MODEL_CONFIG:
            if key not in pipelines:
                result[key] = {"error": f"{label} not loaded"}
                continue
            p = pipelines[key]

            # Preprocess
            df_proc = df_in.copy()
            for col in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
                df_proc[col] = df_proc[col].replace(0, np.nan)
            if p.get("knn_imputer"):
                df_proc[BASE_FEATURES] = p["knn_imputer"].transform(df_proc[BASE_FEATURES])
            df_proc = engineer_features(df_proc)

            feat_cols = p.get("feature_cols", BASE_FEATURES)
            X = df_proc[feat_cols].values
            X_sc = p["scaler"].transform(X)
            if p.get("poly"):
                X_sc = p["poly"].transform(X_sc)

            prob = float(p["model"].predict_proba(X_sc)[0][1])
            # Apply isotonic calibrator if available (improves probability quality)
            if p.get("calibrator") is not None:
                prob = float(np.clip(p["calibrator"].predict([prob])[0], 0.0, 1.0))
            thresh = p.get("threshold", 0.5)
            pred = int(prob >= thresh)

            result[key] = {
                "label"      : label,
                "prediction" : pred,
                "probability": round(prob * 100, 2),
                "diagnosis"  : "Diabetic" if pred == 1 else "Non-Diabetic",
            }

        return jsonify({"success": True, "results": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    print("\n[START] 4-Model Diabetes Prediction Dashboard")
    print("   http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
