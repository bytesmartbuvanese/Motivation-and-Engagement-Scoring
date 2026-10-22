"""
main.py
=========================
FastAPI Production Endpoint for Student Engagement Scoring & Dropout Risk Prediction.

Endpoints:
1. GET /  --> System Health & Metadata
2. GET /students/{student_course_id} --> Retrieve full 5-Dimension Dashboard Profile for a student
3. POST /predict --> Predict Dropout Risk % and compute Engagement Profile for new/raw student features
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# APP INITIALIZATION & PATHS
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Student Engagement Scoring & Early-Warning API",
    description="Multi-dimensional engagement scoring engine and ML dropout risk predictor.",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "engagement_model_outputs", "best_dropout_model.pkl")
PREDICTIONS_CSV_PATH = os.path.join(BASE_DIR, "engagement_model_outputs", "student_engagement_predictions.csv")
RAW_DATASET_PATH = os.path.join(BASE_DIR, "engagement_model_outputs", "raw_student_features.csv")

# Global Cache Variables
model_pipeline = None
predictions_df = None
raw_df = None


# -----------------------------------------------------------------------------
# STARTUP EVENT: LOAD MODEL & DATASETS INTO MEMORY
# -----------------------------------------------------------------------------
@app.on_event("startup")
def load_artifacts():
    global model_pipeline, predictions_df, raw_df
    
    # 1. Load ML Model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model_pipeline = pickle.load(f)
            print(f"✓ Successfully loaded trained model from: {MODEL_PATH}")
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
    else:
        print(f"⚠️ Model file not found at: {MODEL_PATH}")

    # 2. Load Pre-computed Student Dashboard Profiles
    if os.path.exists(PREDICTIONS_CSV_PATH):
        predictions_df = pd.read_csv(PREDICTIONS_CSV_PATH)
        predictions_df['student_course_id'] = predictions_df['student_course_id'].astype(str)
        print(f"✓ Loaded {len(predictions_df):,} student dashboard profiles.")
    else:
        print(f"⚠️ Predictions dataset not found at: {PREDICTIONS_CSV_PATH}")

    # 3. Load Raw Production Feature Dataset
    if os.path.exists(RAW_DATASET_PATH):
        raw_df = pd.read_csv(RAW_DATASET_PATH)
        raw_df['student_course_id'] = raw_df['student_course_id'].astype(str)
        print(f"✓ Loaded {len(raw_df):,} raw feature records.")


# -----------------------------------------------------------------------------
# PYDANTIC SCHEMAS FOR API REQUESTS & RESPONSES
# -----------------------------------------------------------------------------
class StudentProfileResponse(BaseModel):
    student_course_id: str
    id_student: int
    code_module: str
    code_presentation: str
    overall_engagement_score: float
    engagement_level: str
    dropout_risk_pct: float
    trend_direction: str
    sub_dimension_scores: dict
    primary_warning: str
    advisor_action: str


class FeaturePredictionInput(BaseModel):
    clicks_recent: float = Field(..., example=42.0, description="Total clicks in current 14d window")
    active_days_recent: float = Field(..., example=5.0, description="Active days in current 14d window")
    activity_velocity_recent: float = Field(..., example=8.4, description="Clicks per active day")
    unique_resources_recent: float = Field(..., example=4.0, description="Unique resources accessed")
    activity_change_pct: float = Field(..., example=-0.30, description="% change in activity clicks vs past window")
    active_days_change_pct: float = Field(..., example=-0.20, description="% change in active days vs past window")
    resource_exploration_change_pct: float = Field(..., example=-0.10, description="% change in unique resources")
    tasks_completed: float = Field(..., example=2.0, description="Tasks completed up to cutoff day")
    tasks_scheduled: float = Field(..., example=2.0, description="Tasks scheduled up to cutoff day")
    completion_rate: float = Field(..., example=1.0, description="Completion rate (0.0 to 1.0)")
    days_since_last_activity: float = Field(..., example=6.0, description="Days elapsed since last VLE click")
    avg_assessment_score: Optional[float] = Field(0.0, example=75.0, description="Average assessment score")


class LivePredictionResponse(BaseModel):
    dropout_risk_pct: float
    engagement_score: float
    engagement_level: str
    primary_warning: str
    advisor_action: str


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS FOR SCORING & GUIDANCE
# -----------------------------------------------------------------------------
def compute_live_scores_and_guidance(features: dict, dropout_risk_pct: float):
    # Sub-dimension scores
    act_score = min(100.0, max(0.0, features['activity_velocity_recent'] * 8.0))
    cons_score = min(100.0, max(0.0, (features['active_days_recent'] / 7.0) * 100.0))
    exp_score = min(100.0, max(0.0, features['unique_resources_recent'] * 10.0))
    pers_score = min(100.0, max(0.0, (features['completion_rate'] * 60.0) + (features['avg_assessment_score'] * 0.4)))
    rec_score = max(0.0, 100.0 - (features['days_since_last_activity'] * 5.0))
    
    overall_score = round(
        (0.25 * act_score) + (0.20 * cons_score) + (0.15 * exp_score) + 
        (0.25 * pers_score) + (0.15 * rec_score), 1
    )
    
    # Engagement level
    if overall_score >= 70:
        level = "Low Risk (Healthy) 🟢"
    elif overall_score >= 45:
        level = "Moderate Risk 🟡"
    else:
        level = "High Risk (At-Risk) 🔴"
        
    # Warnings & Actions
    warnings = []
    actions = []
    if rec_score < 40:
        warnings.append("Student has been inactive on the platform for an extended period.")
        actions.append("Contact student via email/phone to check on platform access issues.")
    if pers_score < 40:
        warnings.append("Task completion rate and assessment scores have dropped significantly.")
        actions.append("Review recent assignment submissions and schedule an academic support session.")
    if act_score < 40 and cons_score < 40:
        warnings.append("Interaction frequency and weekly active participation have declined.")
        actions.append("Review student's learning logs and send a gentle progress check-in.")
        
    if not warnings:
        if overall_score >= 70:
            warning = "No immediate engagement concerns; participation is healthy."
            action = "Continue standard automated encouragement check-ins."
        else:
            warning = "Moderate engagement drop observed across recent learning weeks."
            action = "Monitor upcoming assignment deadlines and weekly platform logins."
    else:
        warning = " ".join(warnings[:2])
        action = actions[0]
        
    return overall_score, level, warning, action


# -----------------------------------------------------------------------------
# API ROUTE DEFINITIONS
# -----------------------------------------------------------------------------
@app.get("/")
def health_check():
    """System Health Check and Engine Metadata"""
    return {
        "status": "online",
        "service": "Student Engagement Scoring & Early-Warning API",
        "model_loaded": model_pipeline is not None,
        "precomputed_students_count": len(predictions_df) if predictions_df is not None else 0
    }


@app.get("/students/{student_course_id}", response_model=StudentProfileResponse)
def get_student_profile(
    student_course_id: str = Path(..., description="Unique ID format e.g. AAA_2013J_11391")
):
    """
    Fetch complete 5-Dimension Engagement Profile, Dropout Risk %, and Advisor Guidance
    for a specific student enrollment.
    """
    if predictions_df is None:
        raise HTTPException(status_code=500, detail="Student predictions dataset not loaded on server.")
        
    student_row = predictions_df[predictions_df['student_course_id'] == student_course_id]
    
    if student_row.empty:
        raise HTTPException(
            status_code=404, 
            detail=f"Student ID '{student_course_id}' not found. Verify course/student identifier."
        )
        
    s = student_row.iloc[0].to_dict()
    
    return StudentProfileResponse(
        student_course_id=str(s['student_course_id']),
        id_student=int(s['id_student']),
        code_module=str(s['code_module']),
        code_presentation=str(s['code_presentation']),
        overall_engagement_score=float(s['overall_engagement_score']),
        engagement_level=str(s['engagement_level']),
        dropout_risk_pct=float(s['dropout_risk_pct']),
        trend_direction=str(s['trend_direction']),
        sub_dimension_scores={
            "activity_score": float(s['activity_score']),
            "consistency_score": float(s['consistency_score']),
            "exploration_score": float(s['exploration_score']),
            "persistence_score": float(s['persistence_score']),
            "recency_score": float(s['recency_score'])
        },
        primary_warning=str(s['primary_warning']),
        advisor_action=str(s['advisor_action'])
    )


@app.post("/predict", response_model=LivePredictionResponse)
def predict_student_risk(input_data: FeaturePredictionInput):
    """
    Real-time prediction endpoint: Accepts raw behavioral features for a student,
    runs the ML pipeline to predict Dropout Risk %, and calculates dynamic engagement guidance.
    """
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="ML Model pipeline is not loaded on server.")
        
    input_dict = input_data.dict()
    input_df = pd.DataFrame([input_dict])
    
    try:
        # Predict dropout risk probability
        dropout_prob = model_pipeline.predict_proba(input_df)[0][1]
        dropout_risk_pct = round(float(dropout_prob * 100), 1)
        
        # Calculate dynamic engagement score & warnings
        overall_score, level, warning, action = compute_live_scores_and_guidance(input_dict, dropout_risk_pct)
        
        return LivePredictionResponse(
            dropout_risk_pct=dropout_risk_pct,
            engagement_score=overall_score,
            engagement_level=level,
            primary_warning=warning,
            advisor_action=action
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")