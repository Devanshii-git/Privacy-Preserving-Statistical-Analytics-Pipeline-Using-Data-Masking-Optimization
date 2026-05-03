from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis import machine_learning_evaluation, statistical_analysis, statistical_validation
from src.masking import MaskingEngine
from src.optimization import OptimizationEngine
from src.preprocessing import DataIngestion, SensitiveAttributeIdentifier, extract_binary_target
from src.visualization import VisualizationModule


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Privacy-Preserving Statistical Analytics Pipeline")
    print("=" * 58)

    ingestion = DataIngestion(DATA_DIR, random_state=42)
    df = ingestion.load(sample_size=5000)
    df.to_csv(DATA_DIR / "working_dataset.csv", index=False)
    print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    sensitive_attributes = SensitiveAttributeIdentifier().identify(df)
    print(f"Sensitive attributes: {', '.join(sensitive_attributes)}")

    target = extract_binary_target(df)
    baseline_stats = statistical_analysis(df)
    baseline_stats["mean"].to_csv(OUTPUT_DIR / "original_mean.csv")
    baseline_stats["variance"].to_csv(OUTPUT_DIR / "original_variance.csv")
    baseline_stats["covariance"].to_csv(OUTPUT_DIR / "original_covariance.csv")
    baseline_stats["correlation"].to_csv(OUTPUT_DIR / "original_correlation.csv")

    baseline_ml = machine_learning_evaluation(df, target=target, random_state=42, tune_hyperparameters=True)
    baseline_ml.to_csv(OUTPUT_DIR / "original_ml_metrics.csv", index=False)
    print("\nOriginal ML metrics:")
    print(baseline_ml.round(4).to_string(index=False))

    masking_engine = MaskingEngine(sensitive_attributes=sensitive_attributes, random_state=42)
    optimizer = OptimizationEngine(
        masking_engine=masking_engine,
        quasi_identifiers=sensitive_attributes,
        privacy_weight=0.55,
        random_state=42,
    )

    print("\nRunning grid, randomized, genetic, and Optuna masking optimization...")
    results, best_masked, best_config = optimizer.run(df, baseline_ml=baseline_ml, target=target, n_trials=8)
    results.to_csv(OUTPUT_DIR / "optimization_results.csv", index=False)
    best_masked.to_csv(OUTPUT_DIR / "best_masked_dataset.csv", index=False)

    best_stats = statistical_analysis(best_masked)
    best_stats["mean"].to_csv(OUTPUT_DIR / "best_masked_mean.csv")
    best_stats["variance"].to_csv(OUTPUT_DIR / "best_masked_variance.csv")
    best_stats["covariance"].to_csv(OUTPUT_DIR / "best_masked_covariance.csv")
    best_stats["correlation"].to_csv(OUTPUT_DIR / "best_masked_correlation.csv")

    best_ml = machine_learning_evaluation(best_masked, target=target, random_state=42, tune_hyperparameters=True)
    best_ml.to_csv(OUTPUT_DIR / "best_masked_ml_metrics.csv", index=False)
    validation = statistical_validation(df, best_masked)
    validation.to_csv(OUTPUT_DIR / "statistical_validation.csv", index=False)

    visualizer = VisualizationModule(OUTPUT_DIR / "figures")
    visualizer.correlation_heatmap(df, "original")
    visualizer.correlation_heatmap(best_masked, "best_masked")
    visualizer.accuracy_comparison(baseline_ml, best_ml)
    visualizer.model_metric_comparison(baseline_ml, best_ml)
    visualizer.roc_auc_comparison(baseline_ml, best_ml)
    visualizer.privacy_utility_tradeoff(results)
    visualizer.metric_summary(results)
    visualizer.information_loss(results)
    visualizer.transformation_impact(validation)
    visualizer.generalization_diagnostics(baseline_ml, best_ml)

    best_row = results.iloc[0]
    print("\nBest masking configuration:")
    print(f"Method: {best_config.method}")
    print(
        "Parameters: "
        f"noise_std={best_config.noise_std}, "
        f"bin={best_config.generalization_bin}, "
        f"epsilon={best_config.epsilon}, "
        f"hybrid_dp_budget_fraction={best_config.hybrid_dp_budget_fraction}, "
        f"swap_fraction={best_config.swap_fraction}, "
        f"perturbation_strength={best_config.perturbation_strength}"
    )
    print(f"Optimizer: {best_row['optimizer']}")
    print(f"Privacy score: {best_row['privacy_score']:.4f}")
    print(f"Privacy loss: {best_row['privacy_loss']:.4f}")
    print(f"Utility score: {best_row['utility_score']:.4f}")
    print(f"Objective score (utility - privacy loss): {best_row['objective_score']:.4f}")
    print(f"k-anonymity approximation: {best_row['k_anonymity']:.2f}")
    print(f"Re-identification risk: {best_row['reidentification_risk']:.4f}")
    print(f"Attribute disclosure risk: {best_row['attribute_disclosure_risk']:.4f}")
    print(f"Information leakage: {best_row['information_leakage']:.4f}")
    print(f"Overfitting rate: {best_row['overfitting_rate']:.4f}")
    print(f"Underfitting rate: {best_row['underfitting_rate']:.4f}")
    print(f"Memorization risk rate: {best_row['memorization_risk_rate']:.4f}")

    print("\nBest masked ML metrics:")
    print(best_ml.round(4).to_string(index=False))

    print("\nTop 5 search results:")
    cols = [
        "method",
        "noise_std",
        "generalization_bin",
        "epsilon",
        "swap_fraction",
        "perturbation_strength",
        "privacy_score",
        "privacy_loss",
        "utility_score",
        "objective_score",
        "accuracy_drop",
        "roc_auc_drop",
    ]
    print(results[cols].head(5).round(4).to_string(index=False))

    print(f"\nArtifacts written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
