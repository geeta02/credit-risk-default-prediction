from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.schemas import ApplicantInput, PredictionOutput, PredictionWithExplanation
from app.model import load_model, predict, predict_with_explanation
import time

# ── Lifespan ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Credit Risk API...")
    load_model()
    print("API ready ✅")
    yield
    print("Shutting down API...")


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Credit Risk Prediction API",
    description = "Predicts probability of loan default for an applicant",
    version     = "1.0.0",
    lifespan    = lifespan
)


# ── Endpoint 1: Root ─────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message" : "Credit Risk Prediction API",
        "version" : "1.0.0",
        "status"  : "running",
        "docs"    : "/docs"
    }


# ── Endpoint 2: Health ───────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status"     : "healthy",
        "model"      : "credit_risk_xgboost v1",
        "description": "XGBoost model for credit default prediction"
    }


# ── Endpoint 3: Predict ──────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionOutput)
def predict_default(data: ApplicantInput):
    """
    Predict default probability for a loan applicant.
    Returns probability, risk category and decision.
    """
    try:
        start   = time.time()
        result  = predict(data)
        elapsed = round((time.time() - start) * 1000, 2)
        print(f"Prediction in {elapsed}ms — "
              f"prob={result.default_probability} "
              f"decision={result.decision}")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ── Endpoint 4: Predict with Explanation ─────────────────────────────────────
@app.post("/predict/explain", response_model=PredictionWithExplanation)
def predict_explain(data: ApplicantInput):
    """
    Predict default probability AND explain the decision using SHAP.

    Returns:
    - default_probability: likelihood of default
    - risk_category: LOW / MEDIUM / HIGH
    - decision: APPROVE / REVIEW / REJECT
    - top_risk_factors: features increasing default risk
    - top_protective_factors: features decreasing default risk
    - explanation_summary: plain English explanation
    """
    try:
        start   = time.time()
        result  = predict_with_explanation(data)
        elapsed = round((time.time() - start) * 1000, 2)
        print(f"Explanation in {elapsed}ms — "
              f"prob={result.default_probability} "
              f"decision={result.decision}")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")