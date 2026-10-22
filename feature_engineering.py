# -*- coding: utf-8 -*-
"""
Converted from IPYNB to PY
"""

# %% [code] Cell 1
import pandas as pd
from pathlib import Path

# Base directory where your dataset files are stored
DATA_DIR = Path("/home/austin/algorithms/gement_Scoring/dataset")

# Dictionary to hold all dataframes
dfs = {}

# File list
files = [
    "assessments.csv",
    "courses.csv",
    "studentAssessment.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentVle.csv",
    "vle.csv"
]

print("Loading OULAD dataset files...\n")

for file in files:
    file_path = DATA_DIR / file
    if file_path.exists():
        # Load CSV into dictionary using filename without extension as key
        key_name = file_path.stem
        dfs[key_name] = pd.read_csv(file_path)
        
        # Display row count, column count, and size in MB
        mem_usage = dfs[key_name].memory_usage(deep=True).sum() / (1024 ** 2)
        print(f"✓ Loaded {file:<25} | Shape: {str(dfs[key_name].shape):<15} | Memory: {mem_usage:.2f} MB")
    else:
        print(f"❌ File not found: {file_path}")

print("\n--- Data Loading Complete ---")


# %% [code] Cell 2
# Cell 2: Detailed Inspection of loaded OULAD tables
import pandas as pd

print("=" * 80)
print("OULAD DATASET HEALTH & STRUCTURAL INSPECTION")
print("=" * 80)

for name, df in dfs.items():
    print(f"\n>>> TABLE: {name}.csv")
    print(f"Shape: {df.shape[0]:,} rows | {df.shape[1]} columns")
    
    # 1. Inspect Data Types and Non-Null Counts
    info_df = pd.DataFrame({
        'Dtype': df.dtypes,
        'Null Count': df.isnull().sum(),
        'Null %': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique Values': df.nunique()
    })
    
    # 2. Check for string representation of missing values (e.g., '?')
    question_mark_counts = {}
    for col in df.select_dtypes(include=['object', 'string']).columns:
        q_count = (df[col].astype(str).str.strip() == '?').sum()
        if q_count > 0:
            question_mark_counts[col] = q_count
            
    print("\nColumn Summary:")
    print(info_df.to_string())
    
    if question_mark_counts:
        print("\n⚠️ Columns containing '?' string placeholders:")
        for col, q_cnt in question_mark_counts.items():
            print(f"  - {col}: {q_cnt:,} rows ({(q_cnt / len(df) * 100):.2f}%)")
            
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
    print("-" * 80)

# %% [code] Cell 3
# Cell 3: Data Cleaning, Type Conversion, and Primary Key Standardization
import pandas as pd
import numpy as np

print("Starting Data Cleaning and Standardization...\n")

# 1. Replace '?' string placeholders with np.nan across all tables
for name, df in dfs.items():
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str).str.strip().replace('?', np.nan)

# 2. Correct Specific Data Types
# studentAssessment: Convert score to float and impute missing scores with 0
dfs['studentAssessment']['score'] = dfs['studentAssessment']['score'].astype(float)
dfs['studentAssessment']['score'] = dfs['studentAssessment']['score'].fillna(0.0)

# assessments: Convert date to float
dfs['assessments']['date'] = dfs['assessments']['date'].astype(float)

# studentRegistration: Convert registration and unregistration dates to float
dfs['studentRegistration']['date_registration'] = dfs['studentRegistration']['date_registration'].astype(float)
dfs['studentRegistration']['date_unregistration'] = dfs['studentRegistration']['date_unregistration'].astype(float)

# 3. Create a unique composite key for student-course instances
# Unique ID format: "AAA_2013J_11391"
for table_name in ['studentInfo', 'studentRegistration', 'studentVle']:
    df = dfs[table_name]
    df['student_course_id'] = (
        df['code_module'].astype(str) + '_' + 
        df['code_presentation'].astype(str) + '_' + 
        df['id_student'].astype(str)
    )

# For studentAssessment, merge assessment metadata first to map code_module and code_presentation
dfs['studentAssessment'] = dfs['studentAssessment'].merge(
    dfs['assessments'][['id_assessment', 'code_module', 'code_presentation']], 
    on='id_assessment', 
    how='left'
)
dfs['studentAssessment']['student_course_id'] = (
    dfs['studentAssessment']['code_module'].astype(str) + '_' + 
    dfs['studentAssessment']['code_presentation'].astype(str) + '_' + 
    dfs['studentAssessment']['id_student'].astype(str)
)

print("✓ Replaced all '?' strings with np.nan.")
print("✓ Converted numerical columns (score, dates) to float dtypes.")
print("✓ Added composite key 'student_course_id' across all student tables.")
print(f"✓ Total Unique Student-Course Enrollments in studentInfo: {dfs['studentInfo']['student_course_id'].nunique():,}")

# %% [code] Cell 4
# Cell 4: Production Feature & Temporal Delta Extraction Pipeline
# Purpose: Extracts features strictly prior to cutoff Day T (Day 60). Computes explicit percentage 
# changes between the Current 14d Window and Previous 14d Window. Sets a clean target 
# `is_dropout` strictly based on actual course withdrawal (final_result == 'Withdrawn').

import pandas as pd
import numpy as np

def extract_production_student_features(dfs, target_day=60, window_days=14):
    """
    Extracts behavioral metrics and temporal deltas up to cutoff target_day.
    Target label: 1 if student unregistered/withdrew after target_day, 0 otherwise.
    """
    print(f"Extracting features at Cutoff Day T = {target_day} (Current Window = {window_days}d, Previous Window = {window_days}d)...")
    
    # 1. Active Cohort Selection: Students enrolled at or after cutoff Day T
    reg = dfs['studentRegistration'].copy()
    active_cohort = reg[(reg['date_unregistration'].isnull()) | (reg['date_unregistration'] > target_day)].copy()
    
    features = active_cohort[['student_course_id', 'code_module', 'code_presentation', 'id_student']].merge(
        dfs['studentInfo'][['student_course_id', 'final_result']], 
        on='student_course_id', 
        how='inner'
    )
    
    # Window Boundaries
    curr_start = target_day - window_days
    prev_start = target_day - (2 * window_days)
    
    # Filter VLE Logs
    vle_all = dfs['studentVle'][dfs['studentVle']['date'] <= target_day].copy()
    vle_curr = vle_all[(vle_all['date'] > curr_start) & (vle_all['date'] <= target_day)]
    vle_prev = vle_all[(vle_all['date'] > prev_start) & (vle_all['date'] <= curr_start)]
    
    # -------------------------------------------------------------------------
    # A. CURRENT WINDOW METRICS (T - 14 to T)
    # -------------------------------------------------------------------------
    eng_curr = vle_curr.groupby('student_course_id').agg(
        clicks_recent=('sum_click', 'sum'),
        active_days_recent=('date', 'nunique'),
        unique_resources_recent=('id_site', 'nunique')
    ).reset_index()
    eng_curr['activity_velocity_recent'] = eng_curr['clicks_recent'] / np.maximum(eng_curr['active_days_recent'], 1)
    
    # -------------------------------------------------------------------------
    # B. PREVIOUS WINDOW METRICS (T - 28 to T - 14)
    # -------------------------------------------------------------------------
    eng_prev = vle_prev.groupby('student_course_id').agg(
        clicks_prev=('sum_click', 'sum'),
        active_days_prev=('date', 'nunique'),
        unique_resources_prev=('id_site', 'nunique')
    ).reset_index()
    
    features = features.merge(eng_curr, on='student_course_id', how='left')
    features = features.merge(eng_prev, on='student_course_id', how='left')
    
    # Fill zero-activity records
    for col in ['clicks_recent', 'active_days_recent', 'unique_resources_recent', 'activity_velocity_recent',
                'clicks_prev', 'active_days_prev', 'unique_resources_prev']:
        features[col] = features[col].fillna(0)

    # -------------------------------------------------------------------------
    # C. EXPLICIT TEMPORAL CHANGE FEATURES (% Deltas)
    # -------------------------------------------------------------------------
    features['activity_change_pct'] = (
        (features['clicks_recent'] - features['clicks_prev']) / np.maximum(features['clicks_prev'], 1.0)
    ).clip(lower=-1.0, upper=5.0)  # Capped for numerical stability
    
    features['active_days_change_pct'] = (
        (features['active_days_recent'] - features['active_days_prev']) / np.maximum(features['active_days_prev'], 1.0)
    ).clip(lower=-1.0, upper=5.0)
    
    features['resource_exploration_change_pct'] = (
        (features['unique_resources_recent'] - features['unique_resources_prev']) / np.maximum(features['unique_resources_prev'], 1.0)
    ).clip(lower=-1.0, upper=5.0)

    # -------------------------------------------------------------------------
    # D. PERSISTENCE METRICS (Tasks Completed vs Scheduled)
    # -------------------------------------------------------------------------
    due_assessments = dfs['assessments'][
        (dfs['assessments']['date'] <= target_day) & 
        (dfs['assessments']['assessment_type'] != 'Exam')
    ]
    due_counts = due_assessments.groupby(['code_module', 'code_presentation']).agg(
        tasks_scheduled=('id_assessment', 'nunique')
    ).reset_index()
    
    features = features.merge(due_counts, on=['code_module', 'code_presentation'], how='left')
    features['tasks_scheduled'] = features['tasks_scheduled'].fillna(0)
    
    sub_assessments = dfs['studentAssessment'][dfs['studentAssessment']['date_submitted'] <= target_day].copy()
    pers_stats = sub_assessments.groupby('student_course_id').agg(
        tasks_completed=('id_assessment', 'nunique'),
        avg_assessment_score=('score', 'mean')
    ).reset_index()
    
    features = features.merge(pers_stats, on='student_course_id', how='left')
    features['tasks_completed'] = features['tasks_completed'].fillna(0)
    features['avg_assessment_score'] = features['avg_assessment_score'].fillna(0.0)
    
    features['completion_rate'] = np.where(
        features['tasks_scheduled'] > 0,
        features['tasks_completed'] / features['tasks_scheduled'],
        1.0
    )
    features['completion_rate'] = np.minimum(features['completion_rate'], 1.0)

    # -------------------------------------------------------------------------
    # E. RECENCY & CLEAN TARGET DEFINITION (Strict Withdrawal)
    # -------------------------------------------------------------------------
    last_act = vle_all.groupby('student_course_id').agg(last_active_date=('date', 'max')).reset_index()
    features = features.merge(last_act, on='student_course_id', how='left')
    
    features['days_since_last_activity'] = target_day - features['last_active_date']
    features['days_since_last_activity'] = features['days_since_last_activity'].fillna(target_day + 30)
    
    # Pure Target: 1 ONLY if final_result is 'Withdrawn' (Discontinued Course)
    features['is_dropout'] = np.where(features['final_result'] == 'Withdrawn', 1, 0)
    
    print(f"✓ Feature Extraction Complete! Active cohort size: {features.shape[0]:,} records")
    return features

df_features_production = extract_production_student_features(dfs, target_day=60, window_days=14)

# %% [code] Cell 5
# Cell 5: Production Scoring, Unified Explanation Logic, and Dataset Export
# Purpose: Calculates Engagement/Persistence scores, assigns harmonized trend directions and 
# explanations, and exports clean datasets for ML training and system execution.

import pandas as pd
import numpy as np

def build_unified_scoring_system(features_df, output_filename="oulad_production_engagement_dataset.csv"):
    df = features_df.copy()
    
    # Helper: Percentile scaling to 0–100 range
    def scale_to_100(series):
        return (series.rank(pct=True, method='min') * 100).round(2)

    # 1. ENGAGEMENT SCORE (0–100)
    df['engagement_score'] = (
        0.40 * scale_to_100(df['activity_velocity_recent']) + 
        0.35 * scale_to_100(df['active_days_recent']) + 
        0.25 * scale_to_100(df['unique_resources_recent'])
    ).round(2)

    # 2. PERSISTENCE SCORE (0–100)
    df['persistence_score'] = (
        0.60 * (df['completion_rate'] * 100) + 
        0.40 * scale_to_100(df['avg_assessment_score'])
    ).round(2)

    # -------------------------------------------------------------------------
    # 3. HARMONIZED TREND DIRECTION & EXPLANATION GENERATOR
    # -------------------------------------------------------------------------
    def evaluate_trend_and_explanation(row):
        act_change = row['activity_change_pct']
        days_change = row['active_days_change_pct']
        comp_rate = row['completion_rate']
        days_inactive = row['days_since_last_activity']
        
        # Composite change index
        composite_change = (0.6 * act_change) + (0.4 * days_change)
        
        # Scenario A: Severe Inactivity / Disengagement
        if days_inactive > 7:
            trend = "Declining ↓"
            concern = f"Extended inactivity period ({int(days_inactive)} days without platform interaction)."
        
        # Scenario B: Significant Activity Drop
        elif composite_change <= -0.25:
            trend = "Declining ↓"
            if comp_rate < 0.60:
                concern = "Recent activity frequency and task completion have decreased."
            else:
                concern = "Recent platform interaction frequency has dropped significantly."
                
        # Scenario C: Low Completion Drop
        elif comp_rate < 0.50 and composite_change < 0.0:
            trend = "Declining ↓"
            concern = "Task completion rate is below expected milestone levels."
            
        # Scenario D: Clear Positive Trajectory
        elif composite_change >= 0.20:
            trend = "Improving ↑"
            if row['engagement_score'] > 60 and comp_rate >= 0.80:
                concern = "No major concerns; platform engagement and task completion are improving."
            else:
                concern = "Activity trend is improving; keep maintaining steady participation."
                
        # Scenario E: Stable Participation
        else:
            trend = "Stable →"
            if comp_rate < 0.60:
                concern = "Engagement is stable, but upcoming task completion requires attention."
            else:
                concern = "Participation and activity frequency remain relatively stable."
                
        return pd.Series([trend, concern], index=['trend_direction', 'main_concern'])

    # Apply unified logic across every row
    df[['trend_direction', 'main_concern']] = df.apply(evaluate_trend_and_explanation, axis=1)

    # -------------------------------------------------------------------------
    # 4. COLUMN GROUPINGS FOR ML & SYSTEM EXPORT
    # -------------------------------------------------------------------------
    # Identifiers
    id_cols = ['student_course_id', 'id_student', 'code_module', 'code_presentation']
    
    # Pure Behavioral Features for ML Training Input (X)
    ml_features = [
        'clicks_recent', 'active_days_recent', 'activity_velocity_recent', 'unique_resources_recent',
        'activity_change_pct', 'active_days_change_pct', 'resource_exploration_change_pct',
        'tasks_completed', 'tasks_scheduled', 'completion_rate', 'days_since_last_activity'
    ]
    
    # Optional Academic Feature
    academic_feature = ['avg_assessment_score']
    
    # Target (y)
    target_col = ['is_dropout']
    
    # System Outputs / Dashboard Display (NOT for ML Input)
    dashboard_outputs = ['engagement_score', 'persistence_score', 'trend_direction', 'main_concern', 'final_result']
    
    # Final Full Dataset Export
    export_cols = id_cols + ml_features + academic_feature + target_col + dashboard_outputs
    final_df = df[export_cols].copy()
    
    final_df.to_csv(output_filename, index=False)
    
    print(f"✓ Production dataset exported to: {output_filename}")
    print(f"✓ Total records: {len(final_df):,} | Attributes: {len(export_cols)}")
    return final_df

ml_production_dataset = build_unified_scoring_system(df_features_production)

# Check class balance for pure dropout
print("\nTarget Class Balance (`is_dropout = 1` for Withdrawn):")
print(ml_production_dataset['is_dropout'].value_counts(normalize=True).round(4) * 100)

print("\nHarmonized Trend Direction Distribution:")
print(ml_production_dataset['trend_direction'].value_counts())

print("\nSample Output Profiles:")
sample_cols = ['student_course_id', 'engagement_score', 'persistence_score', 'trend_direction', 'main_concern', 'is_dropout']
print(ml_production_dataset[sample_cols].head(5).to_string())
