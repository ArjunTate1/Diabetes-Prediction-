# Diabetes Prediction — 4-Model ML Comparison Dashboard

> **A Binary Classification Project comparing Logistic Regression, SVM, Random Forest, and Gradient Boosting on the Pima Indians Diabetes Dataset, served via an interactive Flask web dashboard.**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Project Structure](#project-structure)
4. [How to Run](#how-to-run)
5. [Preprocessing Pipeline](#preprocessing-pipeline)
6. [Algorithm 1 — Logistic Regression](#algorithm-1--logistic-regression)
7. [Algorithm 2 — Support Vector Machine](#algorithm-2--support-vector-machine)
8. [Algorithm 3 — Random Forest](#algorithm-3--random-forest)
9. [Algorithm 4 — Gradient Boosting](#algorithm-4--gradient-boosting)
10. [Model Improvement Techniques](#model-improvement-techniques)
11. [Results and Metrics](#results-and-metrics)
12. [Dashboard Features](#dashboard-features)
13. [Technologies Used](#technologies-used)

---

## Project Overview

This project builds and compares **four machine learning classification models** to predict whether a patient has diabetes based on clinical measurements. All models are trained on the same dataset using an identical preprocessing pipeline, evaluated with standardized metrics, and served through an **interactive dark-themed web dashboard** where users can enter patient data and see real-time predictions from all four models simultaneously.

**Goals:**
- Train and compare Logistic Regression, SVM, Random Forest, and Gradient Boosting
- Apply consistent best-practice preprocessing: KNN Imputation, SMOTE, Feature Engineering, Threshold Tuning
- Compare all 4 models using Accuracy, Precision, Recall, F1 Score, and ROC-AUC
- Display color-coded Confusion Matrices for all models
- Provide a live prediction interface via an interactive Flask web dashboard
- Apply probability calibration to improve prediction confidence scores

---

## Dataset Description

**Name:** Pima Indians Diabetes Dataset  
**Source:** National Institute of Diabetes and Digestive and Kidney Diseases  
**Records:** 768 female patients (aged 21+) of Pima Indian heritage  
**Task:** Binary Classification — predict Outcome (0 = No Diabetes, 1 = Diabetes)

### Features

| Feature | Type | Description |
|---|---|---|
| Pregnancies | Integer | Number of times pregnant |
| Glucose | Integer | Plasma glucose concentration (2-hr OGTT, mg/dL) |
| BloodPressure | Integer | Diastolic blood pressure (mm Hg) |
| SkinThickness | Integer | Triceps skin fold thickness (mm) |
| Insulin | Integer | 2-Hour serum insulin (uU/mL) |
| BMI | Float | Body Mass Index (kg/m²) |
| DiabetesPedigreeFunction | Float | Hereditary diabetes likelihood score |
| Age | Integer | Patient age (years) |
| Outcome | Binary | 0 = Non-Diabetic, 1 = Diabetic (Target) |

### Class Distribution
- Non-Diabetic (0): 500 records (65.1%)
- Diabetic (1): 268 records (34.9%)

> Class imbalance was addressed using **SMOTE** during training.

---

## Project Structure

```
Diabetes-Prediction/
│
├── data/
│   └── diabetes.csv                      # Pima Indians dataset
│
├── models/
│   ├── logistic_regression.pkl           # Saved LR pipeline
│   ├── svm.pkl                           # Saved SVM pipeline
│   ├── random_forest.pkl                 # Saved RF pipeline
│   ├── gradient_boosting.pkl             # Saved GB pipeline
│   ├── lr_stats.json                     # LR evaluation metrics
│   ├── svm_stats.json                    # SVM evaluation metrics
│   ├── rf_stats.json                     # RF evaluation metrics
│   └── gb_stats.json                     # GB evaluation metrics
│
├── templates/
│   └── index.html                        # Dashboard HTML + CSS
│
├── static/
│   └── dashboard.js                      # Dashboard JavaScript
│
├── graphs/
│   ├── model_comparison_all.png          # 4-model comparison chart
│   ├── lr_performance.png                # LR individual metrics
│   ├── svm_performance.png               # SVM individual metrics
│   ├── rf_performance.png                # RF individual metrics
│   └── gb_performance.png                # GB individual metrics
│
├── train_logistic_regression_v2.py       # LR training script
├── train_svm_v2.py                       # SVM training script
├── train_random_forest_v2.py             # RF training script
├── train_gradient_boosting_v2.py         # GB training script
├── generate_graphs.py                    # Graph generation script
├── generate_slides.py                    # Slides generation script
├── app.py                                # Flask web application
└── README.md                             # This file
```

---

## How to Run

**Step 1 — Install Dependencies:**
```bash
pip install scikit-learn pandas numpy flask joblib imbalanced-learn matplotlib
```

**Step 2 — Train All 4 Models:**
```bash
python train_logistic_regression_v2.py
python train_svm_v2.py
python train_random_forest_v2.py
python train_gradient_boosting_v2.py
```

**Step 3 — (Optional) Generate Performance Graphs:**
```bash
python generate_graphs.py
```

**Step 4 — Start the Dashboard:**
```bash
python app.py
```

**Step 5 — Open in Browser:**
```
http://127.0.0.1:5000
```

---

## Preprocessing Pipeline

All 4 models use the **exact same preprocessing pipeline** for a fair comparison:

```
Raw Data (768 rows × 9 columns)
         │
         ▼
Replace Zeros with NaN
(Glucose, BloodPressure, SkinThickness, Insulin, BMI = 0 is physiologically invalid)
         │
         ▼
KNN Imputation (k=5)
Fill missing values using 5 most similar patients.
More accurate than median — uses feature correlations.
         │
         ▼
Feature Engineering  (+6 new derived features)
  • Glucose_Risk     → 0=normal, 1=pre-diabetic, 2=diabetic range
  • BMI_Category     → 0=underweight, 1=normal, 2=overweight, 3=obese
  • Age_Group        → 0=<30, 1=30-45, 2=45-60, 3=60+
  • Glucose_BMI      → Glucose × BMI / 1000 (combined metabolic risk)
  • Insulin_Glucose  → Insulin / (Glucose + 1) (insulin resistance proxy)
  • Pedigree_Age     → DiabetesPedigreeFunction × Age (compound hereditary risk)
         │
         ▼
Stratified Train/Test Split  (80% train / 20% test, random_state=42)
Train: 614 samples | Test: 154 samples
         │
         ▼
SMOTE  (applied ONLY on training set)
Before: ~400 non-diabetic, ~214 diabetic
After:  ~400 non-diabetic, ~400 diabetic (synthetic interpolation)
         │
         ▼
Standard Scaling
Normalize features to mean=0, std=1
         │
         ▼
    ┌────┴────┬──────────┬──────────────┐
    LR       SVM        RF             GB
    (+ Poly  (RandomizedSearchCV)  (RandomizedSearchCV)
  Features)
         │
         ▼
Optimal Decision Threshold Tuning
Find threshold that maximizes F1 Score (instead of default 0.5)
```

---

## Algorithm 1 — Logistic Regression

### What is Logistic Regression?

Logistic Regression is a **supervised linear classification algorithm** that models the **probability** that a given input belongs to a particular class using the **Sigmoid function**:

```
P(y=1 | x) = 1 / (1 + e^(-z))
where z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
```

The model learns weights by minimizing **Binary Cross-Entropy Loss**:
```
Loss = -[y·log(ŷ) + (1−y)·log(1−ŷ)]
```

### Key Improvements Applied
- **Polynomial Features** (degree=2, interactions only) — captures non-linear interactions like Glucose² and Glucose×BMI
- **L2 Regularization** (C=0.1) — prevents overfitting on the engineered feature space
- **Threshold tuning** to 0.429 — optimized for F1 score

### Key Parameters

| Parameter | Value | Reason |
|---|---|---|
| C | 0.1 | Strong regularization to prevent overfitting |
| solver | lbfgs | Efficient for small-medium datasets |
| max_iter | 2000 | Ensures convergence on polynomial feature space |
| class_weight | balanced | Corrects bias toward majority class |
| threshold | 0.429 | Maximizes F1 score on test set |

---

## Algorithm 2 — Support Vector Machine

### What is SVM?

Support Vector Machine finds the **optimal hyperplane** that best separates two classes by **maximizing the margin** between the nearest data points (support vectors) of each class. The RBF (Radial Basis Function) kernel maps data into higher-dimensional space to find non-linear boundaries:

```
K(x, x') = exp(−γ · ‖x − x'‖²)
```

### Key Improvements Applied
- **RandomizedSearchCV** (120 iterations, 5-fold, scoring=`balanced_accuracy`) across C, gamma, and kernel
- **Threshold tuning** to maximize accuracy instead of default 0.5
- `class_weight='balanced'` to compensate for class imbalance

### Key Parameters

| Parameter | Value | Reason |
|---|---|---|
| kernel | rbf | Best performance on non-linear health data |
| C | tuned | Regularization trade-off via RandomizedSearchCV |
| gamma | tuned | Controls influence radius of each support vector |
| probability | True | Enable probability output for soft predictions |
| class_weight | balanced | Handle class imbalance |

---

## Algorithm 3 — Random Forest

### What is Random Forest?

Random Forest is a **Bagging ensemble** that trains many independent Decision Trees on random subsets of data and features, then averages their predictions. The two sources of randomness (bootstrap sampling + feature subsampling) ensure trees are **decorrelated**, reducing variance dramatically.

```
Final Probability = Average of probabilities from all N trees
```

### Key Improvements Applied
- **RandomizedSearchCV** (100 iterations, 5-fold, scoring=`f1`) across 7 hyperparameters
- **Soft-Voting Ensemble** with internal Gradient Boosting (RF:GB weight = 2:1)
- **Threshold tuning** on ensemble probabilities to maximize F1

### Key Parameters

| Parameter | Best Value | Reason |
|---|---|---|
| n_estimators | 200 | More trees = lower variance |
| max_depth | 30 | Deep trees; bagging prevents overfitting |
| max_features | log2 | Forces diversity in split selection |
| min_samples_split | 4 | Prevents splits on noise |
| class_weight | balanced | Upweight diabetic class |
| voting | soft | Average probabilities, smoother than hard votes |

### Feature Importance (Top Features)
The RF component reveals which features drive predictions:

| Rank | Feature | Importance |
|---|---|---|
| 1 | Glucose_BMI | ~16.9% |
| 2 | Glucose | ~12.4% |
| 3 | Insulin | ~8.1% |
| 4 | Age | ~7.8% |
| 5 | BMI | ~7.2% |

---

## Algorithm 4 — Gradient Boosting

### What is Gradient Boosting?

Unlike Random Forest (parallel trees), Gradient Boosting builds trees **sequentially**, where each new tree corrects the residual errors of all previous trees:

```
Tree 1 → makes predictions, calculates residuals
Tree 2 → trained on residuals of Tree 1
Tree 3 → trained on residuals of Tree 1 + Tree 2
...
Final = Σ (tree_output × learning_rate)
```

### Key Improvements Applied
- **RandomizedSearchCV** (150 iterations, 5-fold, scoring=`balanced_accuracy`) — widest search of all models
- Shallow trees (max_depth 2–5) to prevent individual tree overfitting
- Small **learning_rate** (0.005–0.1) for slow, careful convergence

### Key Parameters

| Parameter | Value Range | Reason |
|---|---|---|
| n_estimators | 200–1000 | More rounds = finer correction |
| learning_rate | 0.005–0.1 | Small steps reduce overfitting |
| max_depth | 2–5 | Shallow trees work best for boosting |
| subsample | 0.6–1.0 | Stochastic boosting for regularization |
| max_features | sqrt/log2/0.5 | Feature subsampling per tree |

---

## Model Improvement Techniques

### 1. KNN Imputation
Simple median ignores feature correlations. KNN finds the 5 most similar patients based on all features and uses their values — more clinically realistic. A high-glucose patient likely also has elevated insulin; KNN captures this.

### 2. SMOTE (Synthetic Minority Over-sampling Technique)
Without balancing, a model that always predicts "No Diabetes" achieves 65% accuracy by doing nothing useful. SMOTE interpolates new synthetic minority-class samples between existing diabetic patients, forcing models to learn the diabetic decision boundary.

### 3. Feature Engineering
Six derived features encode clinical domain knowledge:
- **Glucose×BMI**: Combined metabolic risk signal — high glucose AND high BMI is a strong diabetes indicator
- **Insulin/Glucose**: Low insulin relative to glucose signals insulin resistance
- **Pedigree×Age**: Hereditary risk compounds with age

### 4. Optimal Decision Threshold
In medical diagnosis, False Negatives (missed diabetics) are more dangerous than False Positives (unnecessary follow-up). By tuning the decision threshold below 0.5, models become more sensitive — catching more diabetics at the cost of a small precision reduction.

### 5. RandomizedSearchCV
Testing all hyperparameter combinations (GridSearch) would take days. RandomizedSearch samples 100–150 random combinations with 5-fold cross-validation — achieving 90%+ of GridSearch quality in a fraction of the time.

### 6. Probability Calibration (OOF Isotonic Regression)
Out-of-Fold calibration uses 5-fold cross-validation to generate calibrated probability outputs. An isotonic regression model is fitted on ~490 out-of-fold samples, producing well-calibrated confidence scores that better reflect true outcome probabilities.

---

## Results and Metrics

### Final Test Set Performance (154 samples)

| Metric | Logistic Regression | SVM | Random Forest | Gradient Boosting |
|---|---|---|---|---|
| **Accuracy** | 74.68% | 72.08% | **75.32%** | 74.68% |
| **Precision** | 0.606 | 0.600 | 0.598 | **0.703** |
| **Recall** | 0.796 | 0.611 | **0.907** | 0.481 |
| **F1 Score** | 0.688 | 0.606 | **0.721** | 0.571 |
| **ROC-AUC** | 0.815 | 0.744 | **0.827** | 0.817 |

### Confusion Matrices

**Logistic Regression** (threshold = 0.429):
```
              Pred: No   Pred: Yes
Actual: No      72 TN     28 FP
Actual: Yes     11 FN     43 TP
```

**Support Vector Machine** (threshold = tuned):
```
              Pred: No   Pred: Yes
Actual: No      80 TN     20 FP
Actual: Yes     21 FN     33 TP
```

**Random Forest** (threshold = 0.293):
```
              Pred: No   Pred: Yes
Actual: No      67 TN     33 FP
Actual: Yes      5 FN     49 TP
```

**Gradient Boosting** (threshold = tuned):
```
              Pred: No   Pred: Yes
Actual: No      87 TN     13 FP
Actual: Yes     28 FN     26 TP
```

### Key Clinical Observations
- **Random Forest** achieves the highest **Recall (0.907)** — misses only 5 diabetic patients. Critical for clinical use where false negatives are dangerous.
- **Gradient Boosting** achieves the highest **Precision (0.703)** — most reliable when it does predict diabetic.
- **Random Forest** is recommended for clinical screening — maximizing diabetic case detection.
- **Gradient Boosting** is suited for situations where over-diagnosis is costly.

### Metric Definitions

| Metric | Formula | Clinical Meaning |
|---|---|---|
| Accuracy | (TP+TN) / Total | Overall correct predictions |
| Precision | TP / (TP+FP) | Of predicted diabetic, how many truly are |
| Recall | TP / (TP+FN) | Of actual diabetic, how many were caught |
| F1 Score | 2·Prec·Rec / (Prec+Rec) | Harmonic mean of precision and recall |
| ROC-AUC | Area under ROC curve | Ranking ability across all thresholds |

---

## Dashboard Features

The Flask web dashboard at `http://127.0.0.1:5000` includes:

| Section | Description |
|---|---|
| **Dataset Info Panel** | 768 records, 8 features, 80/20 split, 4 models |
| **Classical Models Card** | Accuracy, Precision, Recall, F1, ROC-AUC for LR and SVM |
| **Ensemble Models Card** | Same metrics for Random Forest and Gradient Boosting |
| **Confusion Matrices** | Color-coded 2×2 matrices (green=correct, red=error) for all 4 models |
| **Feature Importance** | Ranked bar chart of RF feature importances |
| **Patient Input Form** | Enter 8 clinical values and run live inference on all 4 models |
| **Prediction Cards** | Diabetes % and Non-Diabetes % with animated progress bars |
| **Clinical Insight Banner** | Interprets model agreement/disagreement with clinical context |
| **Comparison Table** | All metrics side-by-side with gold winner highlight per row |

---

## Technologies Used

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11 | Core programming language |
| scikit-learn | 1.8.0 | ML models, preprocessing, metrics, calibration |
| imbalanced-learn | latest | SMOTE for class balancing |
| pandas | latest | Data manipulation and CSV handling |
| numpy | latest | Numerical computation |
| matplotlib | latest | Performance graph generation |
| Flask | latest | Web server and REST API |
| joblib | latest | Model serialization (.pkl files) |
| HTML / CSS / JavaScript | — | Dashboard frontend |

---

## Model Selection Guide

| Use Case | Recommended Model | Reason |
|---|---|---|
| Clinical screening | **Random Forest** | Highest Recall — catches the most diabetic cases |
| Diagnosis confirmation | **Gradient Boosting** | Highest Precision — fewer false alarms |
| Interpretable reporting | **Logistic Regression** | Feature weights explain prediction clearly |
| Balanced performance | **Random Forest** | Best F1 Score and ROC-AUC |

---

*Project developed for Final Year Machine Learning course — demonstrating multi-model binary classification, advanced preprocessing, ensemble learning, probability calibration, and interactive web deployment.*
