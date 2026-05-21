import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import mlflow
import sklearn

print("✓ pandas", pd.__version__)
print("✓ numpy", np.__version__)
print("✓ xgboost", xgb.__version__)
print("✓ sklearn", sklearn.__version__)
print("✓ mlflow", mlflow.__version__)
print("✓ shap", shap.__version__)

df = pd.read_csv("data/application_train.csv")
print(f"\n✓ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"✓ Default rate: {df['TARGET'].mean():.1%}")
