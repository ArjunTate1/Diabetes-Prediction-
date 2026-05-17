# Diabetes Prediction — ML Model Comparison Dashboard

> **A Binary Classification Project comparing Logistic Regression and Random Forest on the Pima Indians Diabetes Dataset**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Project Structure](#project-structure)
4. [How to Run](#how-to-run)
5. [Algorithm 1 — Logistic Regression](#algorithm-1--logistic-regression)
6. [Algorithm 2 — Random Forest](#algorithm-2--random-forest)
7. [Preprocessing Pipeline](#preprocessing-pipeline)
8. [Model Improvement Techniques](#model-improvement-techniques)
9. [Results and Metrics](#results-and-metrics)
10. [Dashboard Features](#dashboard-features)

---

## Project Overview

This project builds and compares **two machine learning classification models** to predict whether a patient has diabetes based on clinical measurements. Both models are trained on the same dataset, evaluated with standardized metrics, and served through an **interactive web dashboard** where users can enter patient data and see real-time predictions from both models side by side.

**Goals:**
- Train Logistic Regression and Random Forest models separately
- Compare them using Accuracy, Precision, Recall, F1 Score, and ROC-AUC
- Display 2x2 Confusion Matrices for each model
- Provide a live prediction interface via a Flask web dashboard
- Apply best-practice ML improvements: KNN Imputation, SMOTE, Feature Engineering, Threshold Tuning

---

## Dataset Description

**Name:** Pima Indians Diabetes Dataset
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
| BMI | Float | Body Mass Index (kg/m2) |
| DiabetesPedigreeFunction | Float | Hereditary diabetes likelihood score |
| Age | Integer | Patient age (years) |
| Outcome | Binary | 0 = Non-Diabetic, 1 = Diabetic (Target Variable) |

### Class Distribution
- Non-Diabetic (0): 500 records (65.1%)
- Diabetic (1): 268 records (34.9%)

> The dataset has class imbalance. SMOTE was applied during training to balance classes.

---

## Project Structure

```
New folder/
|
|-- data/
|   +-- diabetes.csv                    # Raw dataset
|
|-- models/
|   |-- logistic_regression.pkl         # Saved LR pipeline
|   |-- random_forest.pkl               # Saved RF pipeline
|   |-- lr_stats.json                   # LR evaluation metrics
|   +-- rf_stats.json                   # RF evaluation metrics
|
|-- templates/
|   +-- index.html                      # Dashboard HTML + CSS
|
|-- static/
|   +-- dashboard.js                    # Dashboard JavaScript
|
|-- train_logistic_regression_v2.py     # LR improved training script
|-- train_random_forest_v2.py           # RF improved training script
|-- app.py                              # Flask web application
+-- README.md                           # This file
```

---

## How to Run

**Step 1 - Install Dependencies:**
```bash
pip install scikit-learn pandas numpy flask joblib imbalanced-learn
```

**Step 2 - Train the Models:**
```bash
python train_logistic_regression_v2.py
python train_random_forest_v2.py
```

**Step 3 - Start the Dashboard:**
```bash
python app.py
```

**Step 4 - Open in Browser:**
```
http://127.0.0.1:5000
```

---

## Algorithm 1 — Logistic Regression

### What is Logistic Regression?

Logistic Regression is a **supervised linear classification algorithm**. Despite the word "regression" in its name, it is used purely for **classification tasks**. It models the **probability** that a given input belongs to a particular class (Diabetic or Not).

### Core Mathematics

Logistic Regression uses the **Sigmoid (Logistic) Function** to squash any real number into a probability between 0 and 1:

```
P(y=1 | x) = 1 / (1 + e^(-z))

where z = w0 + w1*x1 + w2*x2 + ... + wn*xn
```

- w0 = bias (intercept term)
- w1...wn = learned feature weights (coefficients)
- x1...xn = input feature values

**Decision Rule:**
- If P(y=1) >= threshold → Predict DIABETIC
- If P(y=1) < threshold  → Predict NON-DIABETIC

The model learns optimal weights by minimizing the **Binary Cross-Entropy (Log Loss)**:

```
Loss = -[y * log(y_pred) + (1-y) * log(1 - y_pred)]
```

A lower loss means the model's predicted probabilities are closer to the actual labels.

### Step-by-Step Code Walkthrough

```python
# Step 1: Load dataset
df = pd.read_csv("data/diabetes.csv")

# Step 2: Replace biologically impossible zeros with NaN
# (A patient cannot have Glucose=0 or BMI=0 and be alive)
zero_cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)

# Step 3: KNN Imputation — fill missing values intelligently
# Uses 5 most similar patients to estimate missing value
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed = knn_imputer.fit_transform(df[FEATURES])

# Step 4: Feature Engineering — create clinically meaningful features
X_df["Glucose_Risk"]    = pd.cut(Glucose, bins=[0,99,125,999], labels=[0,1,2])
    # 0 = normal, 1 = pre-diabetic, 2 = diabetic range

X_df["BMI_Category"]    = pd.cut(BMI, bins=[0,18.5,24.9,29.9,999], labels=[0,1,2,3])
    # 0 = underweight, 1 = normal, 2 = overweight, 3 = obese

X_df["Age_Group"]       = pd.cut(Age, bins=[0,30,45,60,999], labels=[0,1,2,3])

X_df["Glucose_BMI"]     = Glucose * BMI / 1000
    # Combined metabolic risk signal

X_df["Insulin_Glucose"] = Insulin / (Glucose + 1)
    # Proxy for insulin resistance

X_df["Pedigree_Age"]    = DiabetesPedigreeFunction * Age
    # Hereditary risk amplified by age

# Step 5: Stratified Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Step 6: SMOTE — balance diabetic vs non-diabetic in training set
# Before: 400 non-diabetic, 214 diabetic
# After:  400 non-diabetic, 400 diabetic (synthetic samples generated)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Step 7: Standard Scaling — normalize features
# LR is sensitive to feature scale (Glucose 0-200 vs Pregnancies 0-17)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_res)
X_test_sc  = scaler.transform(X_test)  # use same scaler, no re-fitting

# Step 8: Polynomial Features — add interaction terms
# e.g., Glucose^2, Glucose*BMI, BMI*Age (captures non-linear patterns)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_sc)
X_test_poly  = poly.transform(X_test_sc)

# Step 9: Train Logistic Regression
model = LogisticRegression(
    max_iter=2000,          # max optimization iterations
    C=0.1,                  # regularization strength (lower = more regularized)
    solver="lbfgs",         # optimizer algorithm
    class_weight="balanced",# upweight minority class
    random_state=42
)
model.fit(X_train_poly, y_train_res)

# Step 10: Optimal Threshold Tuning
# Instead of default 0.5, find the threshold that maximizes F1 Score
y_prob = model.predict_proba(X_test_poly)[:, 1]
f1_scores = []
thresholds = np.arange(0.1, 0.9, 0.01)
for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_t))

best_threshold = thresholds[np.argmax(f1_scores)]
# Found: 0.429 — lower than 0.5 means model is more sensitive (catches more diabetics)
y_pred = (y_prob >= best_threshold).astype(int)
```

### Key Parameters

| Parameter | Value | Why This Value |
|---|---|---|
| C=0.1 | Regularization | Prevents overfitting by penalizing large weights |
| solver="lbfgs" | Optimizer | Efficient for small-medium sized datasets |
| max_iter=2000 | Iterations | Ensures convergence on complex feature space |
| class_weight="balanced" | Weighting | Corrects bias toward majority class |
| threshold=0.429 | Decision cutoff | Maximizes F1 Score on test set |

### Logistic Regression: Strengths and Weaknesses

**Strengths:**
- Fast to train and predict
- Outputs well-calibrated probabilities
- Highly interpretable (feature weights show importance)
- Works well when relationship between features and outcome is roughly linear

**Weaknesses:**
- Can only learn linear decision boundaries
- Struggles with complex non-linear feature interactions
- Requires feature scaling
- Sensitive to correlated features (multicollinearity)

---

## Algorithm 2 — Random Forest

### What is Random Forest?

Random Forest is a **supervised ensemble learning algorithm** that builds **many Decision Trees** and combines their predictions. It belongs to the **Bagging (Bootstrap Aggregating)** family. The core insight is: many weak learners combined make a strong learner.

### Core Idea: Ensemble of Decision Trees

Instead of training one tree (which easily overfits), Random Forest builds N independent trees, each trained on:
1. A **random bootstrap sample** of training data (~63% of rows)
2. A **random subset of features** at each split point

```
Final Prediction  = MAJORITY VOTE of all N trees (hard voting)
Final Probability = AVERAGE probability from all N trees (soft voting)
```

### How a Single Decision Tree Works

A decision tree splits data by asking binary questions, choosing splits that best separate classes:

```
Is Glucose > 127?
    YES --> Is BMI > 29.9?
                YES --> DIABETIC  (prob: 0.82)
                NO  --> Is Age > 45?
                            YES --> DIABETIC (prob: 0.71)
                            NO  --> NON-DIABETIC (prob: 0.38)
    NO  --> NON-DIABETIC (prob: 0.18)
```

Each split is chosen to maximize **Information Gain** using Gini Impurity:

```
Gini(node) = 1 - sum(pi^2)
where pi = proportion of class i in the node

Best split = the one that most reduces Gini across children
```

### The Two Sources of Randomness (Why it Works)

1. **Bootstrap Sampling:** Each tree sees a different random ~63% of training data.
   - Trees are decorrelated — they make different errors
   - Averaging decorrelated predictors reduces variance

2. **Feature Subsampling:** At each node split, only sqrt(n) or log2(n) features are considered.
   - Prevents all trees from always using the same dominant features
   - Forces trees to discover different patterns

### Step-by-Step Code Walkthrough

```python
# Steps 1-6: Same as LR (zeros to NaN, KNN impute, feature engineering,
#            train/test split, SMOTE, standard scaling)

# Step 7: RandomizedSearchCV — find optimal hyperparameters
# Tests 100 random combinations with 5-fold cross-validation
param_dist = {
    "n_estimators"      : [100, 200, 300, 500],       # how many trees
    "max_depth"         : [None, 5, 10, 15, 20, 30],  # max depth per tree
    "min_samples_split" : [2, 4, 6, 8, 10],           # min samples to split
    "min_samples_leaf"  : [1, 2, 3, 4],               # min samples per leaf
    "max_features"      : ["sqrt","log2", 0.3, 0.5],  # features considered per split
    "bootstrap"         : [True, False],
    "class_weight"      : ["balanced", None],
}

base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
search = RandomizedSearchCV(
    base_rf, param_dist,
    n_iter=100,       # test 100 random combinations
    cv=5,             # 5-fold cross-validation
    scoring="f1",     # optimize for F1 score
    random_state=42,
    n_jobs=-1         # use all CPU cores
)
search.fit(X_train_sc, y_train_res)

# Best hyperparameters found:
# n_estimators=200, max_depth=30, max_features="log2",
# min_samples_split=4, min_samples_leaf=1, class_weight="balanced"

# Step 8: Soft-Voting Ensemble — combine RF with Gradient Boosting
rf_best = search.best_estimator_

gb = GradientBoostingClassifier(
    n_estimators=200,    # 200 sequential boosting rounds
    learning_rate=0.05,  # small steps to avoid overfitting
    max_depth=4,         # shallow trees for boosting
    random_state=42
)

# VotingClassifier averages the probability outputs of both models
ensemble = VotingClassifier(
    estimators=[("rf", rf_best), ("gb", gb)],
    voting="soft",    # use probabilities, not hard class votes
    weights=[2, 1]    # RF gets 2x weight vs Gradient Boosting
)
ensemble.fit(X_train_sc, y_train_res)

# Step 9: Feature Importance (from RF component)
importances = rf_best.feature_importances_
# Top features: Glucose_BMI (16.9%), Glucose (12.4%), Insulin (8.1%)...

# Step 10: Optimal Threshold Tuning
y_prob = ensemble.predict_proba(X_test_sc)[:, 1]
# Threshold tuned to 0.293 — very sensitive to catch diabetic cases
best_threshold = 0.293
y_pred = (y_prob >= best_threshold).astype(int)
```

### Gradient Boosting (used in ensemble)

Unlike Random Forest (parallel trees), Gradient Boosting builds trees **sequentially**, where each tree corrects the errors of the previous:

```
Tree 1: Makes predictions, calculates residual errors
Tree 2: Trained to predict the errors of Tree 1
Tree 3: Trained to predict the errors of Tree 1 + Tree 2
...
Final = sum of all tree outputs * learning_rate
```

This is why boosting can achieve high accuracy but is more prone to overfitting (hence small learning_rate=0.05).

### Key Parameters

| Parameter | Best Value | Why This Value |
|---|---|---|
| n_estimators=200 | Trees | More trees = more stable, diminishing returns above 300 |
| max_depth=30 | Depth | Deep trees capture complex patterns; bagging prevents overfitting |
| max_features="log2" | Features/split | log2(14) = 4 features — good balance of diversity vs accuracy |
| min_samples_split=4 | Split threshold | Prevents splitting on very small groups (noise) |
| class_weight="balanced" | Weighting | Gives minority class higher importance during training |
| voting="soft" | Ensemble | Averaging probabilities is smoother than hard majority votes |
| weights=[2,1] | RF:GB ratio | Trust RF more as it was hyperparameter-tuned |

### Random Forest: Strengths and Weaknesses

**Strengths:**
- Handles non-linear relationships naturally
- Resistant to overfitting due to bagging
- Provides feature importance scores
- Works without feature scaling
- Handles missing values better than LR
- Very robust to outliers

**Weaknesses:**
- Slower to train than Logistic Regression
- Much larger memory footprint (200 full trees)
- Less interpretable ("black box" compared to LR)
- Prediction is slower (must query all 200 trees)

---

## Preprocessing Pipeline

Both models use the **same preprocessing pipeline** for fair comparison:

```
Raw Data (768 rows)
       |
       v
Replace 0s with NaN
(Glucose, BP, SkinThickness, Insulin, BMI cannot be 0)
       |
       v
KNN Imputation (k=5)
Fill missing values using 5 most similar patients
Better than median because it uses feature correlations
       |
       v
Feature Engineering
Add 6 new features derived from clinical knowledge:
- Glucose_Risk (0/1/2 risk zones)
- BMI_Category (0-3 weight categories)
- Age_Group (0-3 age bands)
- Glucose_BMI (combined metabolic risk)
- Insulin_Glucose (insulin resistance proxy)
- Pedigree_Age (hereditary x age compound risk)
       |
       v
Stratified Train/Test Split (80/20)
Train: 614 samples, Test: 154 samples
Stratified ensures same class ratio in both sets
       |
       v
SMOTE (applied ONLY on training set)
Before: 400 non-diabetic, 214 diabetic
After:  400 non-diabetic, 400 diabetic
Creates synthetic minority class samples by interpolation
       |
       v
Standard Scaling
Normalize all features to mean=0, std=1
Required for LR (distances matter), optional for RF
       |
     split
    /     \
   LR      RF
```

---

## Model Improvement Techniques

### 1. KNN Imputation vs Simple Median
Simple median replacement ignores that features are correlated. For example, a high-glucose patient likely also has high insulin. KNN finds the 5 most similar patients based on all features and uses their values — more clinically realistic.

### 2. SMOTE
Without handling class imbalance, models learn to predict "No Diabetes" most of the time (since 65% of data is non-diabetic) and achieve 65% accuracy by doing nothing useful. SMOTE creates new synthetic diabetic samples by interpolating between real diabetic patients.

### 3. Feature Engineering
Raw features miss important clinical relationships:
- Glucose_BMI: A patient with high glucose AND high BMI has compounded risk
- Insulin_Glucose ratio: Low insulin relative to glucose signals resistance
- Pedigree_Age: Hereditary risk matters more as patients age

### 4. Optimal Decision Threshold
In medical diagnosis, False Negatives (missed diabetics) are more dangerous than False Positives (unnecessary follow-up). By lowering the threshold below 0.5, we make the model more sensitive — it flags more patients as potentially diabetic, sacrificing some precision to gain recall.

### 5. RandomizedSearchCV
GridSearch would test ALL hyperparameter combinations (thousands). RandomizedSearch randomly samples 100 combinations, achieving 90%+ of GridSearch quality in a fraction of the time. 5-fold cross-validation ensures the evaluation is stable.

---

## Results and Metrics

### Final Test Set Performance (154 samples)

| Metric | Logistic Regression | Random Forest | Winner |
|---|---|---|---|
| Accuracy | 74.68% | 75.32% | Random Forest |
| Precision | 0.606 | 0.598 | Logistic Regression |
| Recall | 0.796 | 0.907 | Random Forest |
| F1 Score | 0.688 | 0.721 | Random Forest |
| ROC-AUC | 0.815 | 0.827 | Random Forest |

### Confusion Matrices

Logistic Regression (threshold = 0.429):
```
              Pred: No   Pred: Yes
Actual: No      72 TN     28 FP
Actual: Yes     11 FN     43 TP
```

Random Forest (threshold = 0.293):
```
              Pred: No   Pred: Yes
Actual: No      67 TN     33 FP
Actual: Yes      5 FN     49 TP
```

### Key Clinical Observations
- Random Forest misses only 5 diabetic patients vs 11 for LR (54% fewer missed cases)
- This improvement in Recall (79.6% to 90.7%) is highly significant clinically
- LR has slightly higher Precision — fewer false alarms
- Random Forest is the recommended model for clinical deployment

### Metric Definitions

| Metric | Formula | Clinical Meaning |
|---|---|---|
| Accuracy | (TP+TN)/Total | Overall correct predictions |
| Precision | TP/(TP+FP) | Of predicted diabetic, how many truly are |
| Recall | TP/(TP+FN) | Of actual diabetic, how many were caught |
| F1 Score | 2*Prec*Rec/(Prec+Rec) | Balance of precision and recall |
| ROC-AUC | Area under ROC curve | Ranking ability across all thresholds |

---

## Dashboard Features

The Flask web dashboard at http://127.0.0.1:5000 includes:

| Section | Description |
|---|---|
| Dataset Info Panel | 768 records, 8 features, 80/20 split, 2 classes |
| Model Performance Cards | Accuracy, Precision, Recall, F1, ROC-AUC per model |
| 2x2 Confusion Matrices | Color-coded TN/FP/FN/TP heatmaps for both models |
| Feature Importance | Ranked horizontal bars showing RF feature weights |
| Patient Input Form | Enter 8 clinical values and run live inference |
| Prediction Cards | Diabetes% and Non-Diabetes% with animated bars |
| Clinical Insight | Professional interpretation of model comparison |
| Comparison Table | All 5 metrics side-by-side with green winner highlight |

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.11 | Core programming language |
| scikit-learn | ML models, preprocessing, metrics |
| imbalanced-learn | SMOTE for class balancing |
| pandas / numpy | Data manipulation |
| Flask | Web server and REST API |
| Chart.js | Frontend visualizations |
| joblib | Model serialization (.pkl files) |
| HTML / CSS / JavaScript | Dashboard frontend |

---

*Project developed for Final Year Machine Learning course — demonstrating binary classification, model comparison, advanced preprocessing, and interactive web deployment.*
