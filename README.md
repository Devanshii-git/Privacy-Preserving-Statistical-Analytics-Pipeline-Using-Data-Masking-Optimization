# Privacy-Preserving Statistical Analytics Pipeline

> *How do we share sensitive datasets without leaking private information or destroying the data's analytical value?*

Welcome to the **Privacy-Preserving Statistical Analytics Pipeline**! This is an advanced, AI-driven framework designed to systematically find the perfect balance between data privacy and machine learning utility. 

By leveraging a suite of privacy transformations and an intelligent optimization engine, this pipeline evaluates masked datasets against standard predictive models to ensure robust anonymity without sacrificing real-world generalization.

---

## The Research Goal

The core mission of this system is to optimize the classic **privacy-utility tradeoff**:

```text
maximize(objective) = utility_score - privacy_loss + balance_bonus
```

The pipeline automatically masks sensitive attributes, trains ML models on both the original and transformed data, measures analytical preservation, estimates privacy leakage, and selects the absolute best transformation configuration for your dataset.

---

## Pipeline Architecture

The workflow is fully automated from raw data to optimized, anonymized output:

```text
1. Original Dataset
2.   ↳ Data Cleaning & Preprocessing
3.   ↳ Sensitive Attribute Identification
4.   ↳ Privacy Transformation Layer
5.   ↳ Masked/Transformed Dataset
6.      ↳ AI/ML Model Training  &  Utility Evaluation
7.      ↳ Privacy Evaluation
8.   ↳ Optimization Engine
9.   ↳ Best Privacy-Utility Tradeoff Selection
```

---

## Implemented Privacy Transformations

We use a multi-faceted approach to protect data:
- **Laplace Noise** (Differential Privacy)
- **Gaussian Noise Addition**
- **Generalization** (e.g., converting exact ages into age ranges)
- **Data Swapping**
- **Feature Perturbation**
- **Hybrid Transformations** (A powerful combination of multiple methods!)

---

## Machine Learning & Generalization Controls

To ensure the masked data remains highly useful for downstream data science, the evaluator strictly enforces:
- **Robust Splitting**: 60/20/20 train-validation-test split.
- **Cross-Validation**: K-fold cross-validation for final original and best-masked model reports.
- **Model Diversity**: Regularized Logistic Regression, SVM, constrained Random Forest, and optional XGBoost.
- **Optimization**: Grid-search hyperparameter tuning & feature selection via mutual information.
- **Safety Checks**: Train-vs-validation gap checks, log-loss monitoring, and flags for overfitting, underfitting, and memorization risk.

---

## The Optimization Engine

The optimizer intelligently searches for the best masking parameters using a combination of:
- Compact Grid Search
- Randomized Search
- Lightweight Genetic Search
- **Optuna/TPE Bayesian Optimization** (used when `optuna` is installed)

Each candidate is rigorously scored using privacy metrics, utility metrics, statistical validation, and generalization diagnostics.

---

## Statistical Validation Metrics

We generate comprehensive validation metrics to ensure statistical fidelity:
- Correlation & Covariance Preservation
- Welch t-statistics & Approximate p-values
- KL Divergence & KS Statistics
- Wasserstein Distance
- Variance Preservation & Distribution Mean Shift

---

## Quick Start

Ready to run the pipeline?

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline
python main.py
```
*(Make sure to run these commands from the project root).*

### Dataset Handling
The pipeline looks for `data/adult.csv` by default. 
- If not found locally, it dynamically attempts to load the **Adult Income** dataset from OpenML. 
- If OpenML is unavailable, it automatically generates a highly realistic, deterministic **synthetic dataset** to ensure the pipeline always runs!

**Default Sensitive Attributes:** `Age`, `Gender`, `Income`

---

## Key Outputs

Once the pipeline completes, it generates a wealthy suite of artifacts in the `outputs/` directory:

**Data & Metrics (`outputs/`)**
- `best_masked_dataset.csv` - Your final, anonymized dataset!
- `optimization_results.csv`
- `original_ml_metrics.csv` & `best_masked_ml_metrics.csv`
- `statistical_validation.csv`
- Correlation and Covariance matrices (both original and masked).

**Visualizations (`outputs/figures/`)**
- `privacy_utility_tradeoff.png`
- `model_metric_comparison.png`
- `roc_auc_comparison.png`
- `generalization_diagnostics.png`
- `information_loss_by_method.png`
- `transformation_impact.png`

---
*Built for the future of privacy-preserving AI.*
