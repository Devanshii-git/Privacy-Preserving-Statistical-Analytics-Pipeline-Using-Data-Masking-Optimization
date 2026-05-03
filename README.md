# Privacy-Preserving Statistical Analytics Pipeline Using Data Masking Optimization

Advanced AI-driven framework for masking sensitive tabular attributes while preserving statistical utility, model performance, and generalization quality.

## Research Goal

The system optimizes the privacy-utility tradeoff:

```text
objective = utility_score - privacy_loss + balance_bonus
```

It masks sensitive attributes, trains ML models on original and transformed data, measures analytical preservation, estimates privacy leakage, and selects the best transformation configuration.

## Pipeline

```text
Original Dataset
  -> Data Cleaning & Preprocessing
  -> Sensitive Attribute Identification
  -> Privacy Transformation Layer
  -> Masked/Transformed Dataset
  -> AI/ML Model Training
  -> Utility Evaluation
  -> Privacy Evaluation
  -> Optimization Engine
  -> Best Privacy-Utility Tradeoff Selection
```

## Implemented Privacy Transformations

- Noise addition
- Differential privacy with Laplace noise
- Generalization, such as age ranges
- Data swapping
- Feature perturbation
- Hybrid transformation combining multiple methods

## ML and Generalization Controls

The evaluator uses:

- 60/20/20 train-validation-test split
- K-fold cross-validation for final original and best-masked model reports
- Regularized Logistic Regression and SVM
- Constrained Random Forest
- Optional XGBoost when installed
- Grid-search hyperparameter tuning
- Feature selection with mutual information
- Train-vs-validation gap checks
- Validation/test log-loss monitoring
- Overfitting, underfitting, and memorization-risk flags

## Optimization

The optimizer combines:

- Compact grid search
- Randomized search
- Lightweight genetic search
- Optuna/TPE Bayesian optimization when `optuna` is installed

Each candidate is scored with privacy metrics, utility metrics, statistical validation, and generalization diagnostics.

## Statistical Validation

Generated validation metrics include:

- Correlation preservation
- Covariance preservation
- Welch t-statistics
- Approximate p-values
- KL divergence
- KS statistics
- Wasserstein distance
- Variance preservation
- Distribution mean shift

## Run

```bash
pip install -r requirements.txt
python main.py
```

Run the commands from the project root.

## Dataset

The pipeline uses `data/adult.csv` when present. If no local file exists, it tries to load the Adult Income dataset from OpenML. If that is unavailable, it creates a deterministic Adult-style synthetic dataset.

Default sensitive attributes:

- Age
- Gender
- Income

## Key Outputs

- `outputs/best_masked_dataset.csv`
- `outputs/optimization_results.csv`
- `outputs/original_ml_metrics.csv`
- `outputs/best_masked_ml_metrics.csv`
- `outputs/statistical_validation.csv`
- `outputs/original_correlation.csv`
- `outputs/best_masked_correlation.csv`
- `outputs/original_covariance.csv`
- `outputs/best_masked_covariance.csv`
- `outputs/figures/privacy_utility_tradeoff.png`
- `outputs/figures/model_metric_comparison.png`
- `outputs/figures/roc_auc_comparison.png`
- `outputs/figures/generalization_diagnostics.png`
- `outputs/figures/information_loss_by_method.png`
- `outputs/figures/transformation_impact.png`
