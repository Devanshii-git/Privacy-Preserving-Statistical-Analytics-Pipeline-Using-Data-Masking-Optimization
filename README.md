# Privacy-Preserving Statistical Analytics Pipeline Using Data Masking Optimization

## Run

```bash
pip install -r requirements.txt
python main.py
```

Run the commands from the `project/` directory.

## Structure

```text
project/
|-- data/
|-- src/
|   |-- preprocessing.py
|   |-- masking.py
|   |-- optimization.py
|   |-- analysis.py
|   |-- evaluation.py
|   |-- visualization.py
|-- main.py
|-- requirements.txt
|-- README.md
```

## Dataset

The pipeline uses `data/adult.csv` when present. If no local file exists, it tries to load the Adult Income dataset from OpenML. If that is unavailable, it creates a deterministic Adult-style synthetic dataset with:

- Age
- Gender
- Income
- Education
- Occupation

Sensitive attributes:

- Age
- Gender
- Income

## Outputs

The run creates:

- `outputs/optimization_results.csv`
- `outputs/best_masked_dataset.csv`
- `outputs/original_ml_metrics.csv`
- `outputs/best_masked_ml_metrics.csv`
- `outputs/statistical_validation.csv`
- `outputs/figures/original_correlation_heatmap.png`
- `outputs/figures/best_masked_correlation_heatmap.png`
- `outputs/figures/accuracy_comparison.png`
- `outputs/figures/privacy_utility_tradeoff.png`
- `outputs/figures/top_config_scores.png`

## Example Log

```text
Privacy-Preserving Statistical Analytics Pipeline
==========================================================
Loaded dataset: 5000 rows x 5 columns
Sensitive attributes: Age, Gender, Income

Original ML metrics:
              model  accuracy   rmse  precision  recall
Logistic Regression    0.7424 0.5075     0.6890  0.4470
      Decision Tree    0.7440 0.5060     0.6815  0.4728

Running randomized masking parameter search...

Best masking configuration:
Method: generalization
Parameters: noise_std=10.0, bin=10, epsilon=1.0, hybrid_dp_budget_fraction=0.7
Privacy score: 0.5924
Utility score: 0.9418
Objective score: 0.7496
```
