"""
train_engagement_model.py
=========================
Complete Production Engine for Student Engagement Scoring & Dropout Prediction.

Outputs:
1. Overall Engagement Score (0-100) & Engagement Level (Low Risk / Moderate Risk / High Risk)
2. Five Sub-Dimension Scores (0-100):
   - Activity Score
   - Consistency Score
   - Exploration Score
   - Persistence Score
   - Recency Score
3. Trend Indicator (Improving ↑ / Stable → / Declining ↓)
4. Primary Warning (Dynamic Natural Language Cause)
5. Advisor Recommended Action
6. ML Dropout Risk Probability (%)
"""

import os
import json
import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    classification_report
)
from sklearn.inspection import permutation_importance
import joblib

# Optional Gradient Boosting Imports
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# -----------------------------------------------------------------------------
import xgboost as xgb
import lightgbm as lgb
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "oulad_production_engagement_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "engagement_model_outputs")

# Columns strictly EXCLUDED from ML model inputs to prevent leakage
EXCLUDE_FROM_X = [
    'student_course_id', 'id_student', 'code_module', 'code_presentation',
    'is_dropout', 'final_result',
    'engagement_score', 'persistence_score', 'trend_direction', 'main_concern',
    'risk_level', 'dropout_risk_baseline'
]

TARGET_COL = 'is_dropout'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. LOAD DATASET
# -----------------------------------------------------------------------------
print("=" * 80)
print("1. LOADING OULAD DATASET")
print("=" * 80)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset '{DATASET_PATH}' not found. Run Cell 5 first.")

df = pd.read_csv(DATASET_PATH)
print(f"Loaded {len(df):,} student records.")

# Separate Features (X) and Target (y)
feature_cols = [col for col in df.columns if col not in EXCLUDE_FROM_X]
X = df[feature_cols].copy()
y = df[TARGET_COL].astype(int).copy()

numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()


# -----------------------------------------------------------------------------
# 2. GROUPED TRAIN / VAL / TEST SPLIT (NO STUDENT LEAKAGE)
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("2. LEAKAGE-FREE GROUPED DATA SPLITTING")
print("=" * 80)

if "id_student" in df.columns:
    groups = df.loc[X.index, "id_student"].astype(str)
    
    splitter1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(splitter1.split(X, y, groups=groups))
    
    X_train, X_temp = X.iloc[train_idx], X.iloc[temp_idx]
    y_train, y_temp = y.iloc[train_idx], y.iloc[temp_idx]
    
    temp_groups = groups.iloc[temp_idx]
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE + 1)
    val_idx, test_idx = next(splitter2.split(X_temp, y_temp, groups=temp_groups))
    
    X_val, X_test = X_temp.iloc[val_idx], X_temp.iloc[test_idx]
    y_val, y_test = y_temp.iloc[val_idx], y_temp.iloc[test_idx]
else:
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp)

print(f"Train size: {len(X_train):,} | Val size: {len(X_val):,} | Test size: {len(X_test):,}")


# -----------------------------------------------------------------------------
# 3. PREPROCESSING & MODEL TRAINING
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("3. TRAINING DROPOUT RISK PREDICTION MODELS")
print("=" * 80)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ]
)

pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)

models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest (Tuned)": RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "Extra Trees (Tuned)": ExtraTreesClassifier(
        n_estimators=150,
        max_depth=12,               # Limit tree depth to prevent memorization
        min_samples_split=10,        # Require at least 10 samples to split a node
        min_samples_leaf=5,          # Require at least 5 samples per leaf
        max_features="sqrt",         # Randomly sample features per split
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=100, scale_pos_weight=pos_weight, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=100, scale_pos_weight=pos_weight, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
}

if HAS_XGB:
    models["XGBoost"] = xgb.XGBClassifier(
        n_estimators=100, scale_pos_weight=pos_weight, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
    )

if HAS_LGB:
    models["LightGBM"] = lgb.LGBMClassifier(
        n_estimators=100, scale_pos_weight=pos_weight, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )

def calculate_pr_auc(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    return auc(recall, precision)

# -----------------------------------------------------------------------------
# 3. TRAINING DROPOUT RISK PREDICTION MODELS (with overfitting check)
# -----------------------------------------------------------------------------

trained_pipelines = {}
model_results = []

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    pipeline.fit(X_train, y_train)
    
    # Predictions
    train_preds = pipeline.predict(X_train)
    val_preds = pipeline.predict(X_val)
    test_preds = pipeline.predict(X_test)
    
    # Accuracy
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    # Overfitting / Underfitting check
    diff_train_val = train_acc - val_acc
    diff_val_test = val_acc - test_acc
    
    print(f"[{name:<20}] "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} "
          f"| Train-Val Δ: {diff_train_val:.4f} | Val-Test Δ: {diff_val_test:.4f}")
    
    model_results.append({
        "Model": name,
        "Train_Accuracy": train_acc,
        "Val_Accuracy": val_acc,
        "Test_Accuracy": test_acc,
        "Train_Val_Diff": diff_train_val,
        "Val_Test_Diff": diff_val_test
    })
    trained_pipelines[name] = pipeline

# Select best model based on Validation Accuracy
results_df = pd.DataFrame(model_results).sort_values(by="Val_Accuracy", ascending=False)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_accuracy.csv"), index=False)

best_model_name = results_df.iloc[0]["Model"]
best_pipeline = trained_pipelines[best_model_name]
print(f"\n🏆 Best Model Selected (Accuracy-based): {best_model_name}")


# -----------------------------------------------------------------------------
# 4. FIVE-DIMENSION ENGAGEMENT SCORING & DASHBOARD GENERATOR
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("4. GENERATING 5-DIMENSION SCORES & ADVISOR DASHBOARDS")
print("=" * 80)

def percentile_score(series):
    """Converts raw metric into 0-100 percentile score."""
    return (series.rank(pct=True, method='min') * 100).round(1)

# Copy full dataframe for processing
dashboard_df = df.copy()

# A. Activity Score (0-100): Based on recent interaction clicks & velocity
dashboard_df['activity_score'] = percentile_score(dashboard_df['activity_velocity_recent'])

# B. Consistency Score (0-100): Based on active days in the current window & active days stability
dashboard_df['consistency_score'] = (
    0.60 * percentile_score(dashboard_df['active_days_recent']) +
    0.40 * (100 - np.minimum(np.abs(dashboard_df['active_days_change_pct']) * 50, 100))
).round(1)

# C. Exploration Score (0-100): Based on unique resources accessed
dashboard_df['exploration_score'] = percentile_score(dashboard_df['unique_resources_recent'])

# D. Persistence Score (0-100): Based on task completion rate & assessment scores
dashboard_df['persistence_score'] = (
    0.60 * (dashboard_df['completion_rate'] * 100) +
    0.40 * percentile_score(dashboard_df['avg_assessment_score'])
).round(1)

# E. Recency Score (0-100): Higher score = more recently active (inactivity penalty)
dashboard_df['recency_score'] = np.maximum(0, 100 - (dashboard_df['days_since_last_activity'] * 5)).round(1)

# OVERALL ENGAGEMENT SCORE (0-100)
dashboard_df['overall_engagement_score'] = (
    0.25 * dashboard_df['activity_score'] +
    0.20 * dashboard_df['consistency_score'] +
    0.15 * dashboard_df['exploration_score'] +
    0.25 * dashboard_df['persistence_score'] +
    0.15 * dashboard_df['recency_score']
).round(1)

# ENGAGEMENT LEVEL CATEGORIZATION
def get_engagement_level(score):
    if score >= 70:
        return "Low Risk (Healthy) 🟢"
    elif score >= 45:
        return "Moderate Risk 🟡"
    else:
        return "High Risk (At-Risk) 🔴"

dashboard_df['engagement_level'] = dashboard_df['overall_engagement_score'].apply(get_engagement_level)

# ML DROPOUT RISK PROBABILITY
ml_dropout_probs = best_pipeline.predict_proba(X)[:, 1]
dashboard_df['dropout_risk_pct'] = (ml_dropout_probs * 100).round(1)

# PRIMARY WARNING & ADVISOR ACTION ENGINE
def generate_advisor_guidance(row):
    warnings = []
    actions = []
    
    # Analyze dimensions to find primary concern
    if row['recency_score'] < 40:
        warnings.append("Student has been inactive on the platform for an extended period.")
        actions.append("Contact student via email/phone to check on platform access issues.")
        
    if row['persistence_score'] < 40:
        warnings.append("Task completion rate and assessment scores have dropped significantly.")
        actions.append("Review recent assignment submissions and schedule an academic support session.")
        
    if row['activity_score'] < 40 and row['consistency_score'] < 40:
        warnings.append("Interaction frequency and weekly active participation have declined.")
        actions.append("Review student's learning logs and send a gentle progress check-in.")
        
    if row['exploration_score'] < 30:
        warnings.append("Content exploration is narrow; student is missing key learning resources.")
        actions.append("Direct student toward core module readings and study materials.")

    # Default fallback if no severe flags exist
    if not warnings:
        if row['overall_engagement_score'] >= 70:
            primary_warning = "No immediate engagement concerns; participation is healthy."
            advisor_action = "Continue standard automated encouragement check-ins."
        else:
            primary_warning = "Moderate engagement drop observed across recent learning weeks."
            advisor_action = "Monitor upcoming assignment deadlines and weekly platform logins."
    else:
        primary_warning = " ".join(warnings[:2])
        advisor_action = actions[0]
        
    return pd.Series([primary_warning, advisor_action], index=['primary_warning', 'advisor_action'])

dashboard_df[['primary_warning', 'advisor_action']] = dashboard_df.apply(generate_advisor_guidance, axis=1)

# -----------------------------------------------------------------------------
# 5. EXPORT PREDICTIONS & DISPLAY EXAMPLE STUDENT PROFILE
# -----------------------------------------------------------------------------
output_cols = [
    'student_course_id', 'id_student', 'code_module', 'code_presentation',
    'overall_engagement_score', 'engagement_level',
    'activity_score', 'consistency_score', 'exploration_score', 
    'persistence_score', 'recency_score',
    'trend_direction', 'dropout_risk_pct',
    'primary_warning', 'advisor_action'
]

final_dashboard_df = dashboard_df[output_cols].copy()
export_path = os.path.join(OUTPUT_DIR, "student_engagement_predictions.csv")
final_dashboard_df.to_csv(export_path, index=False)

print(f"✓ Saved 5-Dimension Dashboard Dataset to: {export_path}")

# Print an example student output to verify format
sample_student = final_dashboard_df.iloc[0]
print("\n" + "=" * 80)
print("SAMPLE STUDENT ENGAGEMENT PROFILE OUTPUT")
print("=" * 80)
print(f"Student ID:                 {sample_student['student_course_id']}")
print(f"Overall Engagement Score:   {sample_student['overall_engagement_score']} / 100")
print(f"Engagement Level:           {sample_student['engagement_level']}")
print(f"ML Dropout Risk:            {sample_student['dropout_risk_pct']}%")
print(f"Trend:                      {sample_student['trend_direction']}")
print("-" * 40)
print(f"  • Activity Score:         {sample_student['activity_score']}")
print(f"  • Consistency Score:      {sample_student['consistency_score']}")
print(f"  • Exploration Score:      {sample_student['exploration_score']}")
print(f"  • Persistence Score:      {sample_student['persistence_score']}")
print(f"  • Recency Score:          {sample_student['recency_score']}")
print("-" * 40)
print(f"Primary Warning:\n  {sample_student['primary_warning']}")
print(f"\nAdvisor Action:\n  {sample_student['advisor_action']}")
print("=" * 80)

import pickle
import os

with open(os.path.join(OUTPUT_DIR, "best_dropout_model.pkl"), "wb") as f:
    pickle.dump(best_pipeline, f)

print("✓ Best model saved as best_dropout_model.pkl")


print("\n✓ PROCESS COMPLETE. All dashboard profiles and models saved to 'engagement_model_outputs/'")