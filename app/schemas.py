from pydantic import BaseModel
from typing import List, Optional

class ApplicantInput(BaseModel):
    """Input features for credit risk prediction"""

    # Financial amounts
    AMT_INCOME_TOTAL  : float
    AMT_CREDIT        : float
    AMT_ANNUITY       : float
    AMT_GOODS_PRICE   : float

    # Personal info
    AGE_YEARS         : float
    CODE_GENDER       : str    # 'M', 'F', or 'UNKNOWN'

    # Employment
    EMPLOYMENT_YEARS      : float
    EMPLOYMENT_AGE_RATIO  : float

    # Loan info
    NAME_CONTRACT_TYPE : int   # 0=Cash, 1=Revolving
    CREDIT_TERM        : float
    INCOME_CREDIT_RATIO: float
    ANNUITY_INCOME_RATIO: float
    INCOME_PER_PERSON  : float

    # Assets
    FLAG_OWN_CAR    : int   # 0=No, 1=Yes
    FLAG_OWN_REALTY : int   # 0=No, 1=Yes

    # External credit scores
    EXT_SOURCE_1 : Optional[float] = None
    EXT_SOURCE_2 : Optional[float] = None
    EXT_SOURCE_3 : Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "AMT_INCOME_TOTAL"   : 202500,
                "AMT_CREDIT"         : 406597,
                "AMT_ANNUITY"        : 24700,
                "AMT_GOODS_PRICE"    : 351000,
                "AGE_YEARS"          : 35.5,
                "CODE_GENDER"        : "M",
                "EMPLOYMENT_YEARS"   : 5.2,
                "EMPLOYMENT_AGE_RATIO": 0.15,
                "NAME_CONTRACT_TYPE" : 0,
                "CREDIT_TERM"        : 0.06,
                "INCOME_CREDIT_RATIO": 0.50,
                "ANNUITY_INCOME_RATIO": 0.12,
                "INCOME_PER_PERSON"  : 101250,
                "FLAG_OWN_CAR"       : 0,
                "FLAG_OWN_REALTY"    : 1,
                "EXT_SOURCE_1"       : 0.50,
                "EXT_SOURCE_2"       : 0.60,
                "EXT_SOURCE_3"       : 0.55
            }
        }


class PredictionOutput(BaseModel):
    """Output of credit risk prediction"""
    default_probability : float
    risk_category       : str    # LOW, MEDIUM, HIGH
    decision            : str    # APPROVE, REVIEW, REJECT
    confidence          : str    # model confidence level


from typing import List

class FeatureFactor(BaseModel):
    """Single feature contribution to prediction"""
    feature   : str
    impact    : float
    direction : str   # INCREASES_RISK or DECREASES_RISK

class PredictionWithExplanation(BaseModel):
    """Prediction output with SHAP explanation"""
    default_probability     : float
    risk_category           : str
    decision                : str
    confidence              : str
    top_risk_factors        : List[FeatureFactor]
    top_protective_factors  : List[FeatureFactor]
    explanation_summary     : str