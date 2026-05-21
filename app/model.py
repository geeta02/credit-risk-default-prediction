import pandas as pd
import numpy as np
import xgboost as xgb
import json
import shap
from app.schemas import ApplicantInput, PredictionOutput, PredictionWithExplanation, FeatureFactor

# ── Global variables ────────────────────────────────────────────────────────
model           = None
explainer       = None
feature_names   = None
feature_medians = None

def load_model():
    """Load model and feature metadata at startup"""
    global model, explainer, feature_names, feature_medians

    # Load feature metadata
    with open('app/feature_metadata.json', 'r') as f:
        metadata = json.load(f)

    feature_names   = metadata['feature_names']
    feature_medians = metadata['feature_medians']

    # Load XGBoost model directly from file
    model = xgb.XGBClassifier()
    model.load_model('app/model_artifacts/xgboost_model.json')

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    print(f"Model loaded ✅")
    print(f"SHAP explainer ready ✅")
    print(f"Features expected : {len(feature_names)}")


def engineer_features(data: ApplicantInput) -> pd.DataFrame:
    """
    Build complete 190-feature row for model inference.

    Strategy:
      1. Start with training medians for ALL 190 features
      2. Override with actual user-provided values
      3. Handle EXT_SOURCE → mean_ext_score + is_missing_ext
      4. Handle CODE_GENDER → OHE columns
    """

    # ── Step 1: Start with all medians as defaults ──────────────────────────
    # This ensures we always have all 190 features
    row = {col: feature_medians.get(col, 0) for col in feature_names}

    # ── Step 2: Override with user-provided values ──────────────────────────
    row['AMT_INCOME_TOTAL']     = data.AMT_INCOME_TOTAL
    row['AMT_CREDIT']           = data.AMT_CREDIT
    row['AMT_ANNUITY']          = data.AMT_ANNUITY
    row['AMT_GOODS_PRICE']      = data.AMT_GOODS_PRICE
    row['AGE_YEARS']            = data.AGE_YEARS
    row['EMPLOYMENT_YEARS']     = data.EMPLOYMENT_YEARS
    row['EMPLOYMENT_AGE_RATIO'] = data.EMPLOYMENT_AGE_RATIO
    row['NAME_CONTRACT_TYPE']   = data.NAME_CONTRACT_TYPE
    row['CREDIT_TERM']          = data.CREDIT_TERM
    row['INCOME_CREDIT_RATIO']  = data.INCOME_CREDIT_RATIO
    row['ANNUITY_INCOME_RATIO'] = data.ANNUITY_INCOME_RATIO
    row['INCOME_PER_PERSON']    = data.INCOME_PER_PERSON
    row['FLAG_OWN_CAR']         = data.FLAG_OWN_CAR
    row['FLAG_OWN_REALTY']      = data.FLAG_OWN_REALTY

    # ── Step 3: EXT_SOURCE handling ─────────────────────────────────────────
    ext_values = {
        'EXT_SOURCE_1': data.EXT_SOURCE_1,
        'EXT_SOURCE_2': data.EXT_SOURCE_2,
        'EXT_SOURCE_3': data.EXT_SOURCE_3,
    }

    # Flag if all three missing
    all_missing    = all(v is None for v in ext_values.values())
    is_missing_ext = 1 if all_missing else 0

    # Mean of available scores
    valid_scores   = [v for v in ext_values.values() if v is not None]
    mean_ext_score = float(np.mean(valid_scores)) if valid_scores else 0.0

    row['mean_ext_score'] = mean_ext_score
    row['is_missing_ext'] = is_missing_ext

    # ── Step 4: CODE_GENDER one-hot encoding ────────────────────────────────
    # First reset all gender columns to 0
    for col in ['CODE_GENDER_F', 'CODE_GENDER_M', 'CODE_GENDER_UNKNOWN']:
        if col in row:
            row[col] = 0

    # Then set the correct one to 1
    gender_col = f'CODE_GENDER_{data.CODE_GENDER}'
    if gender_col in row:
        row[gender_col] = 1

    # ── Step 5: Build dataframe in correct column order ─────────────────────
    df = pd.DataFrame([row])[feature_names]

    return df


def predict(data: ApplicantInput) -> PredictionOutput:
    """Make prediction and return business decision"""

    X    = engineer_features(data)
    prob = float(model.predict_proba(X)[0][1])

    # Risk categorisation
    if prob < 0.30:
        risk_category = "LOW"
        decision      = "APPROVE"
        confidence    = "HIGH" if prob < 0.15 else "MEDIUM"
    elif prob < 0.60:
        risk_category = "MEDIUM"
        decision      = "REVIEW"
        confidence    = "MEDIUM"
    else:
        risk_category = "HIGH"
        decision      = "REJECT"
        confidence    = "HIGH" if prob > 0.75 else "MEDIUM"

    return PredictionOutput(
        default_probability = round(prob, 4),
        risk_category       = risk_category,
        decision            = decision,
        confidence          = confidence
    )
def predict_with_explanation(data: ApplicantInput) -> PredictionWithExplanation:
    """
    Make prediction AND explain it using SHAP values.
    Returns top risk factors and protective factors.
    """

    # ── Step 1: Engineer features ───────────────────────────────────────────
    X    = engineer_features(data)
    prob = float(model.predict_proba(X)[0][1])

    # ── Step 2: Compute SHAP values ─────────────────────────────────────────
    shap_values = explainer.shap_values(X)

    # Build feature → shap value mapping
    shap_series = pd.Series(
        shap_values[0],
        index = feature_names
    )

    # ── Step 3: Separate risk factors from protective factors ───────────────
    # Positive SHAP = increases default risk
    # Negative SHAP = decreases default risk (protective)

    risk_factors       = shap_series[shap_series > 0].sort_values(ascending=False)
    protective_factors = shap_series[shap_series < 0].sort_values(ascending=True)

    # ── Step 4: Build top factors list ──────────────────────────────────────
    top_risk = [
        FeatureFactor(
            feature   = feat,
            impact    = round(abs(val), 4),
            direction = "INCREASES_RISK"
        )
        for feat, val in risk_factors.head(5).items()
    ]

    top_protective = [
        FeatureFactor(
            feature   = feat,
            impact    = round(abs(val), 4),
            direction = "DECREASES_RISK"
        )
        for feat, val in protective_factors.head(5).items()
    ]

    # ── Step 5: Business decision ────────────────────────────────────────────
    if prob < 0.30:
        risk_category = "LOW"
        decision      = "APPROVE"
        confidence    = "HIGH" if prob < 0.15 else "MEDIUM"
    elif prob < 0.60:
        risk_category = "MEDIUM"
        decision      = "REVIEW"
        confidence    = "MEDIUM"
    else:
        risk_category = "HIGH"
        decision      = "REJECT"
        confidence    = "HIGH" if prob > 0.75 else "MEDIUM"

    # ── Step 6: Human readable summary ──────────────────────────────────────
    top_reason = risk_factors.index[0] if len(risk_factors) > 0 else "unknown"
    summary = (
        f"Application {decision}ED. "
        f"Primary risk driver: {top_reason}. "
        f"Default probability: {round(prob*100, 1)}%."
    )

    return PredictionWithExplanation(
        default_probability    = round(prob, 4),
        risk_category          = risk_category,
        decision               = decision,
        confidence             = confidence,
        top_risk_factors       = top_risk,
        top_protective_factors = top_protective,
        explanation_summary    = summary
    )