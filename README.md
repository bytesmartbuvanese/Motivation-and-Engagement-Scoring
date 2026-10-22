# Student Motivation & Engagement Scoring Engine with ML Dropout Early Warning System

An end-to-end learning analytics intelligence platform that quantifies multidimensional student engagement, detects disengagement trajectories, predicts course dropout risk early in the academic term (Cutoff Day 60), and delivers real-time prescriptive advisor interventions via a production-grade **FastAPI** microservice.

Built on the **Open University Learning Analytics Dataset (OULAD)** with over 10.6 million digital learning interaction logs across 32,500+ enrollments, this platform transforms passive Virtual Learning Environment (VLE) telemetry into actionable academic advising guidance.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
  - [The Higher Education Retention Problem](#the-higher-education-retention-problem)
  - [Core System Objectives](#core-system-objectives)
  - [Temporal Cutoff & Leakage Prevention](#temporal-cutoff--leakage-prevention)
- [System Architecture](#-system-architecture)
- [Dataset Architecture & Feature Engineering](#-dataset-architecture--feature-engineering)
  - [OULAD Relational Structure](#oulad-relational-structure)
  - [Data Cleansing & Standardization](#data-cleansing--standardization)
  - [Temporal Windowing Methodology](#temporal-windowing-methodology)
  - [Feature Dictionary](#feature-dictionary)
  - [Target Definition & Cohort Distribution](#target-definition--cohort-distribution)
- [5-Dimensional Engagement Scoring Framework](#-5-dimensional-engagement-scoring-framework)
  - [Mathematical Formulation of Sub-Dimensions](#mathematical-formulation-of-sub-dimensions)
  - [Composite Overall Engagement Score](#composite-overall-engagement-score)
  - [Engagement Risk Tiers](#engagement-risk-tiers)
  - [Harmonized Trend Trajectory Engine](#harmonized-trend-trajectory-engine)
- [Dynamic Early-Warning & Advisor Guidance Matrix](#-dynamic-early-warning--advisor-guidance-matrix)
- [Machine Learning Pipeline & Comparative Model Evaluation](#-machine-learning-pipeline--comparative-model-evaluation)
  - [Leakage-Free Grouped Splitting](#leakage-free-grouped-splitting)
  - [Preprocessing Pipeline Architecture](#preprocessing-pipeline-architecture)
  - [Comparative Model Benchmark Results](#comparative-model-benchmark-results)
  - [Model Selection & Overfitting Analysis](#model-selection--overfitting-analysis)
- [Production REST API Microservice (`main.py`)](#-production-rest-api-microservice-mainpy)
  - [API Endpoints Overview](#api-endpoints-overview)
  - [Endpoint 1: System Health (`GET /`)](#endpoint-1-system-health-get-)
  - [Endpoint 2: Student Profile Lookup (`GET /students/{id}`)](#endpoint-2-student-profile-lookup-get-studentsid)
  - [Endpoint 3: Live Behavioral Inference (`POST /predict`)](#endpoint-3-live-behavioral-inference-post-predict)
- [Repository Directory Structure](#-repository-directory-structure)
- [Installation & Execution Guide](#-installation--execution-guide)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Feature Extraction & Engineering](#2-feature-extraction--engineering)
  - [3. Model Training & Pipeline Serialization](#3-model-training--pipeline-serialization)
  - [4. Launching the FastAPI Production Service](#4-launching-the-fastapi-production-service)
- [Technologies & Dependencies](#-technologies--dependencies)

---

## 📌 Project Overview

### The Higher Education Retention Problem

In online, distance, and hybrid higher education environments, student attrition is heavily front-loaded in the first third of a course presentation. Traditional academic monitoring relies on **lagging indicators**—such as missed midterm examinations or failed grade postings—which often trigger alerts when it is already too late for effective advising intervention.

Conversely, interaction data from Virtual Learning Environments (VLE / LMS) represents **leading behavioral indicators**. Changes in access velocity, weekly login consistency, exploration across core learning materials, and submission timeliness reveal waning student motivation weeks before a student formally withdraws.

### Core System Objectives

1. **Quantify Engagement Across Five Meaningful Dimensions**: Measure student participation across **Activity**, **Consistency**, **Exploration**, **Persistence**, and **Recency** on a unified 0–100 scale.
2. **Predict Course Dropout Early (Cutoff Day 60)**: Predict whether an enrolled student will withdraw (`is_dropout = 1`) using only behavioral telemetry captured prior to Day 60 of the course.
3. **Harmonize Temporal Trajectories**: Detect whether student participation is **Improving (↑)**, **Stable (→)**, or **Declining (↓)** relative to their prior baseline.
4. **Prescriptive Advisor Action Triggers**: Automatically map diagnostic root causes to tailored advising actions (e.g., proactive phone check-ins, LMS access reviews, assignment scheduling support).
5. **Production-Ready REST Microservice**: Serve precomputed student profiles and real-time inference via an enterprise **FastAPI** service with strict Pydantic validation.

### Temporal Cutoff & Leakage Prevention

A common failure in academic early-warning models is **hindsight leakage**—using features from late in the semester to predict early-term behavior, or using global summary statistics that overlap the event of interest. 

This engine enforces a strict temporal boundary at **Cutoff Day $T = 60$**:
- **Active Cohort Only**: Only students still formally registered at Day 60 are evaluated ($N = 26,361$).
- **Dual-Window Comparison**: Behavioral change is measured between the **Current Window** (Days 46–60, 14 days) and the **Previous Window** (Days 32–46, 14 days).
- **Forward-Looking Target**: The target label identifies students who withdraw *after* Day 60.

---

## 🏗️ System Architecture

The project integrates data warehousing, temporal feature engineering, model training, and web service delivery into a unified analytics lifecycle:

```text
+----------------------------------------------------------------------------------------------------+
|                                    OULAD RELATIONAL DATASET                                         |
|  courses.csv  |  assessments.csv  |  studentInfo.csv  |  studentRegistration.csv  |  studentVle.csv  |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                             FEATURE ENGINEERING PIPELINE (Day T = 60)                              |
|  • Filter active cohort at Day 60 (26,361 records)                                                  |
|  • Extract Current 14d Window (Days 46-60) vs Previous 14d Window (Days 32-46)                     |
|  • Compute % Deltas: Activity Change %, Active Days Change %, Resource Exploration Change %         |
|  • Measure Persistence (Tasks Completed vs Scheduled, Grade Avg) and Recency (Days Inactive)       |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                         5-DIMENSIONAL ENGAGEMENT & EARLY-WARNING ENGINE                            |
|  • Percentile Scoring: Activity (0-100), Consistency (0-100), Exploration (0-100)                  |
|  • Persistence (0-100) & Recency Inactivity Penalty (0-100)                                        |
|  • Overall Composite Score = 0.25*Act + 0.20*Cons + 0.15*Exp + 0.25*Pers + 0.15*Rec                |
|  • Trajectory Classifier: Improving (↑), Stable (→), Declining (↓)                                 |
|  • Automated Root Cause Diagnostic & Advisor Action Generator                                      |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                        LEAKAGE-FREE ML PIPELINE (GroupShuffleSplit on Student)                     |
|  • Grouped 70/15/15 Split: Zero student overlap between Train, Validation, and Test                |
|  • Preprocessor: ColumnTransformer (Median Imputer + StandardScaler)                               |
|  • Multi-Model Benchmark: Tuned Random Forest (🏆 74.12% Val Acc), XGBoost, LightGBM, Extra Trees  |
|  • Serialized Artifact: engagement_model_outputs/best_dropout_model.pkl                            |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                    FASTAPI PRODUCTION SERVICE                                      |
|                                         (main.py on :8000)                                         |
|                                                                                                    |
|    GET /students/{student_course_id}                     POST /predict                             |
|    ──────────────────────────────────                    ─────────────                             |
|    Instantly returns precomputed 5D                      Accepts live raw features, predicts       |
|    profile, risk %, trend & advisor guidance             dropout probability %, and computes 5D    |
|    (from student_engagement_predictions.csv)             scores + warnings dynamically             |
+----------------------------------------------------------------------------------------------------+
```

---

## 📊 Dataset Architecture & Feature Engineering

### OULAD Relational Structure

The system processes seven core relational tables from the Open University Learning Analytics Dataset:

| File | Shape | Primary Key / Keys | Description |
| :--- | :--- | :--- | :--- |
| `courses.csv` | 22 rows × 3 cols | `code_module`, `code_presentation` | Module names, presentation terms, and module length in days. |
| `assessments.csv` | 206 rows × 6 cols | `id_assessment` | Assessment types (`TMA`, `CMA`, `Exam`), scheduled cutoff days, and weights. |
| `vle.csv` | 6,364 rows × 6 cols | `id_site` | Virtual learning environment resources, material activity types, and associated module weeks. |
| `studentInfo.csv` | 32,593 rows × 12 cols | `id_student`, `code_module`, `code_presentation` | Demographic information, studied credits, disability status, and `final_result`. |
| `studentRegistration.csv` | 32,593 rows × 5 cols | `id_student`, `code_module`, `code_presentation` | Enrollment dates (`date_registration`) and formal withdrawal dates (`date_unregistration`). |
| `studentAssessment.csv` | 173,912 rows × 5 cols | `id_assessment`, `id_student` | Student assessment submission dates and numeric grades (`0–100`). |
| `studentVle.csv` | 10,655,280 rows × 6 cols | Compound (`id_student`, `id_site`, `date`) | Granular interaction log tracking daily click counts per student per VLE resource. |

### Data Cleansing & Standardization

1. **Placeholder Harmonization**: String placeholders (`"?"`) found in `imd_band`, `date`, `date_registration`, and `date_unregistration` are standardized to `np.nan`.
2. **Type Casting**: Assessment scores and submission timeline dates are cast to floating-point numerical types with missing scores imputed as 0.0 (non-submission).
3. **Unified Composite Key**: A unique enrollment primary key is constructed across all student tables:
   $$\text{student\_course\_id} = \text{code\_module} + \text{"\_"} + \text{code\_presentation} + \text{"\_"} + \text{id\_student}$$
   *(Example: `AAA_2013J_11391`)*

### Temporal Windowing Methodology

To prevent hindsight bias, features are generated strictly up to **Cutoff Day $T = 60$**:
- **Recent Window ($W_1$)**: Days $47 \le t \le 60$ (14 days preceding the cutoff).
- **Previous Window ($W_2$)**: Days $33 \le t \le 46$ (14-day baseline prior to the recent window).
- **Temporal Percentage Deltas**:
  $$\Delta_{\text{clicks}} = \text{clip}\left(\frac{\text{clicks}_{W_1} - \text{clicks}_{W_2}}{\max(\text{clicks}_{W_2}, 1.0)}, -1.0, +5.0\right)$$
  $$\Delta_{\text{days}} = \text{clip}\left(\frac{\text{active\_days}_{W_1} - \text{active\_days}_{W_2}}{\max(\text{active\_days}_{W_2}, 1.0)}, -1.0, +5.0\right)$$
  $$\Delta_{\text{resources}} = \text{clip}\left(\frac{\text{resources}_{W_1} - \text{resources}_{W_2}}{\max(\text{resources}_{W_2}, 1.0)}, -1.0, +5.0\right)$$

### Feature Dictionary

The engineered production dataset (`oulad_production_engagement_dataset.csv`) contains **26,361 records** across **22 columns**:

| Feature Name | Type | Role | Value Range | Pedagogical Definition |
| :--- | :--- | :--- | :--- | :--- |
| `student_course_id` | String | Identifier | Composite string | Unique student-course presentation instance (e.g. `BBB_2013J_54210`). |
| `id_student` | Integer | Identifier | Positive integer | Unique Open University student identification number. |
| `code_module` | String | Metadata | `AAA` – `GGG` | Code of the course module (7 distinct academic subjects). |
| `code_presentation` | String | Metadata | `2013B` – `2014J` | Academic term presentation code (`B` = Feb, `J` = Oct). |
| `clicks_recent` | Float | Predictor | $0.0 - 2,500.0+$ | Total LMS interaction clicks during the current 14-day window. |
| `active_days_recent` | Float | Predictor | $0.0 - 14.0$ | Number of unique days the student accessed the VLE in the current window. |
| `activity_velocity_recent` | Float | Predictor | $0.0 - 250.0+$ | Clicks per active day ($\text{clicks\_recent} / \max(\text{active\_days\_recent}, 1)$). |
| `unique_resources_recent` | Float | Predictor | $0.0 - 100.0+$ | Unique learning materials/sites explored in the current window. |
| `activity_change_pct` | Float | Predictor | $-1.0 - +5.0$ | Fractional change in total clicks between current and previous windows. |
| `active_days_change_pct` | Float | Predictor | $-1.0 - +5.0$ | Fractional change in active access days between windows. |
| `resource_exploration_change_pct` | Float | Predictor | $-1.0 - +5.0$ | Fractional change in breadth of learning materials accessed. |
| `tasks_completed` | Float | Predictor | $0.0 - 8.0+$ | Non-exam assessments submitted on or before Day 60. |
| `tasks_scheduled` | Float | Predictor | $0.0 - 8.0+$ | Total non-exam assessments scheduled to be due on or before Day 60. |
| `completion_rate` | Float | Predictor | $0.0 - 1.0$ | Task progress ratio ($\min(\text{tasks\_completed} / \text{tasks\_scheduled}, 1.0)$). |
| `days_since_last_activity` | Float | Predictor | $0.0 - 90.0+$ | Days elapsed between the cutoff (Day 60) and student's most recent VLE click. |
| `avg_assessment_score` | Float | Predictor | $0.0 - 100.0$ | Mean grade across all assessments submitted up to Day 60. |
| `is_dropout` | Integer | **Target** | `0` or `1` | Binary dropout indicator ($1$ if student withdrew after Day 60, $0$ otherwise). |
| `engagement_score` | Float | Output | $0.0 - 100.0$ | Overall engagement index calculated from behavioral telemetry. |
| `persistence_score` | Float | Output | $0.0 - 100.0$ | Milestone persistence index calculated from assessment progress and grades. |
| `trend_direction` | String | Output | Categorical | Engagement trajectory (`Improving ↑`, `Stable →`, `Declining ↓`). |
| `main_concern` | String | Output | Text | Dynamic natural language diagnosis of engagement vulnerability. |
| `final_result` | String | Metadata | Categorical | Ground truth end-of-course status (`Pass`, `Distinction`, `Fail`, `Withdrawn`). |

### Target Definition & Cohort Distribution

The target `is_dropout` is strictly grounded in formal course withdrawal:
- **`is_dropout = 1`**: Student formally withdrew from the course after Cutoff Day 60.
- **`is_dropout = 0`**: Student persisted through course completion (earning `Pass`, `Distinction`, or `Fail`).

```text
Class Distribution across 26,361 Active Students at Day 60:
├── Retained (is_dropout = 0):  22,429 students  (85.08%)
└── Dropped Out (is_dropout = 1): 3,932 students  (14.92%)
```

---

## 🧭 5-Dimensional Engagement Scoring Framework

Rather than treating student activity as an opaque black box, the platform evaluates engagement across five interpretable sub-dimensions on a standardized 0–100 scale:

```text
                               +------------------------------------+
                               |     Overall Engagement Score       |
                               |             (0 - 100)              |
                               +------------------------------------+
                                 /      |        |        |       \
                             25%/    20%|     15%|     25%|    15% \
                               /        |        |        |         \
                              v         v        v        v          v
                       +----------+ +--------+ +----+ +-----------+ +-------+
                       | Activity | | Consist| |Expl| |Persistence| |Recency|
                       |  (0-100) | | (0-100)| |(100| |  (0-100)  | |(0-100)|
                       +----------+ +--------+ +----+ +-----------+ +-------+
```

### Mathematical Formulation of Sub-Dimensions

Let $P(x)$ denote the empirical percentile rank function mapping metric $x$ into $[0, 100]$:

1. **Activity Score ($S_{\text{act}}$)**: Measures interaction intensity and velocity:
   $$S_{\text{act}} = P(\text{activity\_velocity\_recent})$$
2. **Consistency Score ($S_{\text{cons}}$)**: Combines weekly access regularity with week-over-week stability:
   $$S_{\text{cons}} = 0.60 \times P(\text{active\_days\_recent}) + 0.40 \times \left(100 - \min(|\Delta_{\text{days}}| \times 50, 100)\right)$$
3. **Exploration Score ($S_{\text{exp}}$)**: Measures intellectual curiosity and syllabus breadth:
   $$S_{\text{exp}} = P(\text{unique\_resources\_recent})$$
4. **Persistence Score ($S_{\text{pers}}$)**: Evaluates follow-through on formal coursework milestones:
   $$S_{\text{pers}} = 0.60 \times (\text{completion\_rate} \times 100) + 0.40 \times P(\text{avg\_assessment\_score})$$
5. **Recency Score ($S_{\text{rec}}$)**: Applies a temporal decay penalty for consecutive days of dormancy:
   $$S_{\text{rec}} = \max(0, 100 - (\text{days\_since\_last\_activity} \times 5))$$

### Composite Overall Engagement Score

The overall score aggregates the sub-dimensions using pedagogical weighting:

$$\text{Overall Engagement Score} = 0.25 S_{\text{act}} + 0.20 S_{\text{cons}} + 0.15 S_{\text{exp}} + 0.25 S_{\text{pers}} + 0.15 S_{\text{rec}}$$

### Engagement Risk Tiers

| Tier | Score Range | Status Indicator | Pedagogical Interpretation |
| :--- | :---: | :---: | :--- |
| **Healthy** | $70.0 - 100.0$ | 🟢 Low Risk | Regular platform interaction, steady assignment completion, and consistent access. |
| **Moderate** | $45.0 - 69.9$ | 🟡 Moderate Risk | Uneven access patterns or subtle drops in velocity; early intervention candidate. |
| **At-Risk** | $0.0 - 44.9$ | 🔴 High Risk | Acute disengagement, extended inactivity, or missed milestone submissions. |

### Harmonized Trend Trajectory Engine

The platform assigns a trajectory vector by computing composite delta momentum $C = (0.6 \times \Delta_{\text{clicks}}) + (0.4 \times \Delta_{\text{days}})$:

- **Declining (↓)**:
  - $\text{days\_since\_last\_activity} > 7$ (dormant student), OR
  - $C \le -0.25$ (activity velocity declined by $>25\%$), OR
  - $\text{completion\_rate} < 0.50$ AND $C < 0.0$ (milestone deficiency).
- **Improving (↑)**:
  - $C \ge +0.20$ (interaction intensity grew by $\ge 20\%$).
- **Stable (→)**:
  - $-0.25 < C < +0.20$ (steady week-over-week participation).

---

## ⚠️ Dynamic Early-Warning & Advisor Guidance Matrix

The automated guidance engine identifies the primary underlying risk factor and generates immediate, prescriptive recommendations for academic advisors:

| Diagnostic Condition | Root Cause Category | Primary System Warning | Recommended Advisor Action |
| :--- | :--- | :--- | :--- |
| $S_{\text{rec}} < 40$ | **Severe Dormancy** | *"Student has been inactive on the platform for an extended period."* | **Contact student via email/phone to check on platform access issues.** |
| $S_{\text{pers}} < 40$ | **Milestone Failure** | *"Task completion rate and assessment scores have dropped significantly."* | **Review recent assignment submissions and schedule an academic support session.** |
| $S_{\text{act}} < 40 \land S_{\text{cons}} < 40$ | **Participation Decay** | *"Interaction frequency and weekly active participation have declined."* | **Review student's learning logs and send a gentle progress check-in.** |
| $S_{\text{exp}} < 30$ | **Narrow Exploration** | *"Content exploration is narrow; student is missing key learning resources."* | **Direct student toward core module readings and study materials.** |
| $S_{\text{overall}} < 70$ | **Moderate Slippage** | *"Moderate engagement drop observed across recent learning weeks."* | **Monitor upcoming assignment deadlines and weekly platform logins.** |
| $S_{\text{overall}} \ge 70$ | **Healthy Profile** | *"No immediate engagement concerns; participation is healthy."* | **Continue standard automated encouragement check-ins.** |

---

## 🤖 Machine Learning Pipeline & Comparative Model Evaluation

### Leakage-Free Grouped Splitting

In online education datasets, single students often enroll in multiple courses or repeat modules. Standard randomized train/test splits inadvertently cause **student identity leakage**, inflating validation accuracy.

To ensure true out-of-sample generalization, the training script uses `GroupShuffleSplit` on `id_student`:
- **Training Set (70%)**: 18,367 records
- **Validation Set (15%)**: 4,008 records
- **Test Set (15%)**: 3,986 records
- **Guarantee**: Zero student overlap between train, validation, and test sets.

### Preprocessing Pipeline Architecture

Data preprocessing is encapsulated inside a Scikit-Learn `ColumnTransformer`:
- **Numerical Features**: Imputed via `SimpleImputer(strategy='median')` and scaled via `StandardScaler()`.
- **Categorical Features**: Imputed via `SimpleImputer(strategy='most_frequent')` and encoded via `OneHotEncoder(handle_unknown='ignore')`.
- **Imbalance Handling**: Class weights are tuned via `class_weight='balanced_subsample'` (for tree ensembles) or `scale_pos_weight` (for gradient boosters) to address the 14.92% minority dropout rate.

### Comparative Model Benchmark Results

Five classification architectures were evaluated across training, validation, and holdout test sets:

| Model Architecture | Train Accuracy | Validation Accuracy | Test Accuracy | Overfit Δ (Train - Val) | Generalization Δ (Val - Test) | Selection Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Tuned)** | **81.07%** | **74.12%** | **74.01%** | **+0.0694** | **+0.0012** | 🏆 **Production Model** |
| **XGBoost Classifier** | 84.48% | 72.01% | 71.81% | +0.1247 | +0.0020 | Evaluated |
| **Extra Trees (Tuned)** | 71.67% | 68.94% | 69.14% | +0.0273 | -0.0020 | Evaluated |
| **LightGBM Classifier** | 73.14% | 67.34% | 66.56% | +0.0581 | +0.0077 | Evaluated |
| **Logistic Regression** | 65.01% | 65.40% | 64.09% | -0.0040 | +0.0131 | Baseline |

### Model Selection & Overfitting Analysis

- **Top Accuracy**: The **Tuned Random Forest Classifier** achieved the highest validation accuracy (**74.12%**) and holdout test accuracy (**74.01%**).
- **Near-Zero Test Variance**: The difference between validation and test accuracy is only **0.12%** ($\Delta = 0.0012$), proving exceptional out-of-sample stability on previously unseen students.
- **Controlled Complexity**: By constraining `max_depth=12`, `min_samples_split=10`, and `min_samples_leaf=5`, the Random Forest avoids the memorization observed in unregularized gradient boosted trees (XGBoost exhibited an overfit delta of 12.47%).
- **Artifact Export**: The winning pipeline is serialized to `engagement_model_outputs/best_dropout_model.pkl`.

---

## ⚡ Production REST API Microservice (`main.py`)

The production service is implemented in **FastAPI**, providing interactive OpenAPI Swagger documentation, strict input data validation using **Pydantic**, and sub-millisecond profile querying.

### API Endpoints Overview

| Method | Endpoint | Description | Query / Path Parameters |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | System health check, service status, and loaded record count. | None |
| `GET` | `/students/{student_course_id}` | Look up precomputed 5D engagement profile and advisor guidance. | `student_course_id` (e.g. `AAA_2013J_11391`) |
| `POST` | `/predict` | Predict dropout risk % and engagement profile for live student features. | Request Body (JSON) |
| `GET` | `/docs` | Interactive Swagger UI documentation and testing console. | None |

---

### Endpoint 1: System Health (`GET /`)

#### Request:
```bash
curl -X GET "http://127.0.0.1:8000/"
```

#### Response (`200 OK`):
```json
{
  "status": "online",
  "service": "Student Engagement Scoring & Early-Warning API",
  "model_loaded": true,
  "precomputed_students_count": 26361
}
```

---

### Endpoint 2: Student Profile Lookup (`GET /students/{id}`)

Retrieves the precomputed 5-dimensional breakdown, trend trajectory, dropout risk probability, and advisor action for any student enrolled at Day 60.

#### Request:
```bash
curl -X GET "http://127.0.0.1:8000/students/AAA_2013J_32885"
```

#### Response (`200 OK`):
```json
{
  "student_course_id": "AAA_2013J_32885",
  "id_student": 32885,
  "code_module": "AAA",
  "code_presentation": "2013J",
  "overall_engagement_score": 42.1,
  "engagement_level": "High Risk (At-Risk) 🔴",
  "dropout_risk_pct": 55.4,
  "trend_direction": "Stable →",
  "sub_dimension_scores": {
    "activity_score": 20.3,
    "consistency_score": 36.7,
    "exploration_score": 25.5,
    "persistence_score": 43.4,
    "recency_score": 100.0
  },
  "primary_warning": "Interaction frequency and weekly active participation have declined. Content exploration is narrow; student is missing key learning resources.",
  "advisor_action": "Review student's learning logs and send a gentle progress check-in."
}
```

---

### Endpoint 3: Live Behavioral Inference (`POST /predict`)

Accepts raw behavioral features for an enrolled student, feeds them through the serialized `best_dropout_model.pkl` pipeline, dynamically calculates engagement scores, and returns real-time advising recommendations.

#### Request:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "clicks_recent": 14.0,
       "active_days_recent": 2.0,
       "activity_velocity_recent": 7.0,
       "unique_resources_recent": 3.0,
       "activity_change_pct": -0.65,
       "active_days_change_pct": -0.50,
       "resource_exploration_change_pct": -0.30,
       "tasks_completed": 1.0,
       "tasks_scheduled": 2.0,
       "completion_rate": 0.50,
       "days_since_last_activity": 12.0,
       "avg_assessment_score": 52.0
     }'
```

#### Response (`200 OK`):
```json
{
  "dropout_risk_pct": 68.2,
  "engagement_score": 37.6,
  "engagement_level": "High Risk (At-Risk) 🔴",
  "primary_warning": "Student has been inactive on the platform for an extended period. Interaction frequency and weekly active participation have declined.",
  "advisor_action": "Contact student via email/phone to check on platform access issues."
}
```

---

## 📁 Repository Directory Structure

```text
Motivation and Engagement Scoring/
├── dataset/                                      # OULAD relational database files
│   ├── assessments.csv                           # Module assessment schedules & weights (206 rows)
│   ├── courses.csv                               # Course code metadata & term duration (22 rows)
│   ├── studentAssessment.csv                     # Assessment submissions & test scores (173,912 rows)
│   ├── studentInfo.csv                           # Demographics, credits, & final results (32,593 rows)
│   ├── studentRegistration.csv                   # Registration & withdrawal dates (32,593 rows)
│   ├── studentVle.csv                            # Granular daily click logs (10,655,280 rows)
│   └── vle.csv                                   # LMS learning material metadata (6,364 rows)
│
├── engagement_model_outputs/                     # Model artifacts & benchmark evaluations
│   ├── best_dropout_model.pkl                    # Serialized production Random Forest pipeline
│   ├── model_comparison_accuracy.csv             # Train/Val/Test benchmark comparison across 5 models
│   └── student_engagement_predictions.csv        # Precomputed 5D scores for all 26,361 students
│
├── feature_engineering.ipynb                     # Jupyter notebook: Data ingestion, cleaning, Day 60 windowing
├── train_engagement_model.py                     # Production Python script: Grouped split, training, scoring engine
├── main.py                                       # Production FastAPI REST microservice
├── oulad_production_engagement_dataset.csv       # Production feature dataset (26,361 rows × 22 cols)
└── README.md                                     # System documentation and operational guide
```

---

## 🚀 Installation & Execution Guide

### 1. Environment Setup

Clone the repository and activate the virtual environment:

```bash
cd "/home/austin/algorithms/Motivation and Engagement Scoring"

# Create and activate a Python 3.10+ virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost lightgbm fastapi uvicorn pydantic joblib
```

### 2. Feature Extraction & Engineering

To re-run the temporal feature engineering pipeline and generate the Day 60 production dataset:
- Launch Jupyter Notebook or Jupyter Lab:
  ```bash
  jupyter lab feature_engineering.ipynb
  ```
- Execute Cells 1 through 5 to inspect the raw OULAD relational tables, construct composite keys, compute 14-day sliding window deltas, and export `oulad_production_engagement_dataset.csv`.

### 3. Model Training & Pipeline Serialization

Execute the standalone training engine to train all benchmark algorithms, select the optimal model, and serialize the production artifacts:

```bash
python train_engagement_model.py
```

**Expected Console Output**:
```text
================================================================================
1. LOADING OULAD DATASET
================================================================================
Loaded 26,361 student records.

================================================================================
2. LEAKAGE-FREE GROUPED DATA SPLITTING
================================================================================
Train size: 18,367 | Val size: 4,008 | Test size: 3,986

================================================================================
3. TRAINING DROPOUT RISK PREDICTION MODELS
================================================================================
[Random Forest (Tuned)] Train Acc: 0.8107 | Val Acc: 0.7412 | Test Acc: 0.7401 | Train-Val Δ: 0.0694 | Val-Test Δ: 0.0012
[XGBoost             ] Train Acc: 0.8448 | Val Acc: 0.7201 | Test Acc: 0.7181 | Train-Val Δ: 0.1247 | Val-Test Δ: 0.0020
[Extra Trees (Tuned) ] Train Acc: 0.7167 | Val Acc: 0.6894 | Test Acc: 0.6914 | Train-Val Δ: 0.0273 | Val-Test Δ: -0.0020
[LightGBM            ] Train Acc: 0.7314 | Val Acc: 0.6734 | Test Acc: 0.6656 | Train-Val Δ: 0.0581 | Val-Test Δ: 0.0077
[Logistic Regression ] Train Acc: 0.6501 | Val Acc: 0.6540 | Test Acc: 0.6409 | Train-Val Δ: -0.0040 | Val-Test Δ: 0.0131

🏆 Best Model Selected (Accuracy-based): Random Forest (Tuned)
✓ Saved 5-Dimension Dashboard Dataset to: engagement_model_outputs/student_engagement_predictions.csv
✓ Best model saved as best_dropout_model.pkl
```

### 4. Launching the FastAPI Production Service

Start the ASGI web server with Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive Swagger UI**: Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser.
- **ReDoc Documentation**: Visit [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc).

---

## 🛠️ Technologies & Dependencies

| Category | Component / Library | Purpose in Platform |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core programming language for data engineering, modeling, and API services. |
| **Data Processing** | `pandas`, `numpy` | Vectorized aggregations, relational joins, sliding temporal window deltas. |
| **Machine Learning** | `scikit-learn` | `GroupShuffleSplit`, `ColumnTransformer`, `StandardScaler`, `RandomForestClassifier`. |
| **Gradient Boosting** | `xgboost`, `lightgbm` | High-performance gradient boosted decision trees for comparative model benchmarking. |
| **REST Microservice** | `FastAPI`, `uvicorn` | High-concurrency ASGI web framework with automatic OpenAPI documentation. |
| **Data Validation** | `pydantic` | Strict request and response payload schema modeling and type enforcement. |
| **Serialization** | `pickle`, `joblib` | Model pipeline serialization and fast loading for production inference. |
| **Data Source** | OULAD | Open University Learning Analytics Dataset (Open University, UK). |

---

*Developed as part of the Higher Education Learning Analytics and Student Retention Intelligence Suite.*
