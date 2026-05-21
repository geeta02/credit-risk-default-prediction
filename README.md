# 🏦 Credit Risk Default Prediction

> An end-to-end ML system that predicts the probability of loan default for credit applicants — built with production-grade tools.

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


## 📦 Dataset

- **Source:** [Kaggle — Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
- **Rows:** 307,511
- **Raw features:** 122
- **Engineered features:** 190
- **Default rate:** 8.1%

