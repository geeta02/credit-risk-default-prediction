# 🏦 Credit Risk Default Prediction

> An end-to-end machine learning system that predicts the probability of loan default for credit applicants — built with production-grade tools.

---

## 📌 Project Overview

When a bank receives a loan application, it needs to decide — **should we approve or reject this?** This project automates that decision using machine learning, and more importantly, **explains why** each decision was made.

Built on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset — 307,511 real loan applications with 122 features.

---

## 🎯 Problem Statement

**Binary Classification:** Given information about a loan applicant (income, age, employment, credit history), predict whether they will default on their loan.

```
TARGET = 1 → Client defaulted (didn't repay)
TARGET = 0 → Client repaid the loan
```

**Challenge:** Only 8.1% of applicants defaulted — severe class imbalance.

---

## 📊 Model Performance

| Metric | Score | Benchmark |
|--------|-------|-----------|
| ROC-AUC | **0.7642** | > 0.72 ✅ |
| PR-AUC | **0.2463** | > 3x baseline ✅ |
| KS Statistic | **0.3961** | > 0.30 ✅ |
| Default Catch Rate | **67.8%** | > 60% ✅ |

> KS Statistic of 0.3961 falls in the **"Good"** range (0.30–0.40) by banking industry standards.

---

## 🛠️ Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Data Processing | `pandas`, `numpy` | Cleaning, feature engineering |
| ML Model | `XGBoost` | Credit risk prediction |
| Hyperparameter Tuning | `Optuna` | 50-trial automated search |
| Explainability | `SHAP` | Global + local explanations |
| Experiment Tracking | `MLflow` | Parameters, metrics, model registry |
| API | `FastAPI` | REST endpoints for inference |
| Containerisation | `Docker` | Reproducible deployment |

---

## 📁 Project Structure

```
credit-risk-default-prediction/
│
├── app/
│   ├── main.py                     # FastAPI application — 4 endpoints
│   ├── model.py                    # Model loading + inference logic
│   ├── schemas.py                  # Pydantic request/response schemas
│   ├── feature_metadata.json       # Feature names + training medians
│   └── model_artifacts/
│       └── xgboost_model.json      # Trained XGBoost model
│
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb # Feature engineering pipeline
│   ├── 03_model_training.ipynb     # XGBoost training + evaluation
│   ├── 04_shap_analysis.ipynb      # SHAP global + local explanations
│   └── 05_mlflow_tracking.ipynb    # MLflow experiment logging
│
├── Dockerfile                      # Container build instructions
├── docker-compose.yml              # Service orchestration
├── requirements.txt                # Production dependencies
└── README.md
```

---

## 🔑 Key Design Decisions

### Feature Engineering
- **EXT_SOURCE aggregation** — instead of dropping the 56% missing `EXT_SOURCE_1`, computed `mean_ext_score` from all three bureau scores using partial means (skipna), preserving maximum signal
- **is_missing_ext flag** — flagged applicants with zero bureau records as a separate risk signal
- **DAYS_EMPLOYED fix** — 55,374 pensioners/unemployed had placeholder value `365243` (1000 years). Split into groups: pensioners/unemployed → filled with 0, working people with data errors → filled with median
- **Ratio features** — `INCOME_CREDIT_RATIO`, `ANNUITY_INCOME_RATIO`, `EMPLOYMENT_AGE_RATIO` capture financial relationships raw numbers cannot

### Model Training
- **XGBoost** chosen as industry standard for tabular financial data
- **scale_pos_weight = 11.39** to handle 8.1% class imbalance
- **Early stopping (50 rounds)** to prevent overfitting
- **Optuna (50 trials)** for automated hyperparameter search optimising PR-AUC directly

### Explainability
- **SHAP TreeExplainer** for fast, exact explanations on tree models
- **Global bar plot** — which features matter most overall
- **Beeswarm plot** — direction of each feature's impact
- **Waterfall plot** — why a specific applicant was approved/rejected

---

## 🚀 How to Run

### Option 1 — Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/geeta02/credit-risk-default-prediction.git
cd credit-risk-default-prediction

# Build and run
docker compose up --build
```

API will be available at `http://localhost:8000`

### Option 2 — Local

```bash
# Clone and setup
git clone https://github.com/geeta02/credit-risk-default-prediction.git
cd credit-risk-default-prediction

# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn app.main:app --reload --port 8000
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API status and version |
| `/health` | GET | Health check |
| `/predict` | POST | Default probability + decision |
| `/predict/explain` | POST | Prediction + SHAP explanation |

Interactive docs available at: `http://localhost:8000/docs`

---

## 📥 Example Request

```bash
curl -X POST http://localhost:8000/predict/explain \
  -H "Content-Type: application/json" \
  -d '{
    "AMT_INCOME_TOTAL": 202500,
    "AMT_CREDIT": 406597,
    "AMT_ANNUITY": 24700,
    "AMT_GOODS_PRICE": 351000,
    "AGE_YEARS": 25.5,
    "CODE_GENDER": "M",
    "EMPLOYMENT_YEARS": 1.2,
    "EMPLOYMENT_AGE_RATIO": 0.05,
    "NAME_CONTRACT_TYPE": 0,
    "CREDIT_TERM": 0.06,
    "INCOME_CREDIT_RATIO": 0.50,
    "ANNUITY_INCOME_RATIO": 0.12,
    "INCOME_PER_PERSON": 101250,
    "FLAG_OWN_CAR": 0,
    "FLAG_OWN_REALTY": 0,
    "EXT_SOURCE_1": 0.10,
    "EXT_SOURCE_2": 0.15,
    "EXT_SOURCE_3": 0.12
  }'
```

## 📤 Example Response

```json
{
  "default_probability": 0.8151,
  "risk_category": "HIGH",
  "decision": "REJECT",
  "confidence": "HIGH",
  "top_risk_factors": [
    {"feature": "mean_ext_score", "impact": 1.3079, "direction": "INCREASES_RISK"},
    {"feature": "AMT_GOODS_PRICE", "impact": 0.1562, "direction": "INCREASES_RISK"},
    {"feature": "CREDIT_TERM", "impact": 0.0670, "direction": "INCREASES_RISK"}
  ],
  "top_protective_factors": [
    {"feature": "AGE_YEARS", "impact": 0.2870, "direction": "DECREASES_RISK"},
    {"feature": "AMT_CREDIT", "impact": 0.0504, "direction": "DECREASES_RISK"}
  ],
  "explanation_summary": "Application REJECTED. Primary risk driver: mean_ext_score. Default probability: 81.5%."
}
```

---

## 📈 SHAP Analysis

### Global Feature Importance
The model relies most heavily on `mean_ext_score` (aggregated credit bureau score) — consistent with real-world credit scoring where bureau data is the single strongest predictor.

### Feature Impact Directions
| Feature | High Value | Low Value |
|---------|-----------|-----------|
| `mean_ext_score` | ✅ Lowers risk | ❌ Raises risk |
| `AGE_YEARS` | ✅ Lowers risk | ❌ Raises risk |
| `EMPLOYMENT_YEARS` | ✅ Lowers risk | ❌ Raises risk |
| `FLAG_OWN_CAR` | ✅ Lowers risk | ❌ Raises risk |
| `AMT_ANNUITY` | ❌ Raises risk | ✅ Lowers risk |

---

## 🧪 Risk Decision Thresholds

| Probability | Risk Category | Decision |
|-------------|--------------|----------|
| < 0.30 | LOW | APPROVE |
| 0.30 – 0.60 | MEDIUM | REVIEW |
| > 0.60 | HIGH | REJECT |

> Thresholds are business decisions — a conservative bank may use 0.20/0.40, an aggressive lender may use 0.40/0.70.

---

## 📦 Dataset

- **Source:** [Kaggle — Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
- **File used:** `application_train.csv`
- **Rows:** 307,511
- **Raw features:** 122
- **Engineered features:** 190
- **Default rate:** 8.1%

> Dataset not included in this repository due to size. Download from Kaggle and place in `data/` folder.

---

## 🏗️ MLflow Experiment Tracking

All experiments tracked locally using MLflow:
- Hyperparameters logged per run
- Metrics (ROC-AUC, PR-AUC, KS Stat) logged per run
- SHAP plots saved as artifacts
- Best model registered as `credit_risk_xgboost v1`

---

## 👩‍💻 Author

**Geetha Rajamanickam**
- GitHub: [@geeta02](https://github.com/geeta02)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
