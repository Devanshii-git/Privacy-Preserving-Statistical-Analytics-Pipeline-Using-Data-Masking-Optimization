from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .analysis import numeric_projection


class VisualizationModule:
    def __init__(self, output_dir: str | Path = "outputs/figures") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def correlation_heatmap(self, df: pd.DataFrame, name: str) -> Path:
        corr = numeric_projection(df).corr().fillna(0)
        path = self.output_dir / f"{name}_correlation_heatmap.png"
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True)
        plt.title(f"{name.title()} Correlation Matrix")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path

    def accuracy_comparison(self, baseline_ml: pd.DataFrame, masked_ml: pd.DataFrame) -> Path:
        path = self.output_dir / "accuracy_comparison.png"
        baseline = baseline_ml.assign(dataset="Original")
        masked = masked_ml.assign(dataset="Best Masked")
        plot_df = pd.concat([baseline, masked], ignore_index=True)
        plt.figure(figsize=(8, 5))
        sns.barplot(data=plot_df, x="model", y="accuracy", hue="dataset")
        plt.ylim(0, 1)
        plt.title("Machine Learning Accuracy Comparison")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path

    def privacy_utility_tradeoff(self, results: pd.DataFrame) -> Path:
        path = self.output_dir / "privacy_utility_tradeoff.png"
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=results,
            x="utility_score",
            y="privacy_score",
            hue="method",
            size="objective_score",
            sizes=(40, 220),
            alpha=0.8,
        )
        best = results.iloc[0]
        plt.scatter([best["utility_score"]], [best["privacy_score"]], color="black", s=100, marker="X", label="Best")
        plt.xlim(0, 1.02)
        plt.ylim(0, 1.02)
        plt.title("Privacy vs Utility Trade-off")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path

    def metric_summary(self, results: pd.DataFrame) -> Path:
        path = self.output_dir / "top_config_scores.png"
        top = results.head(8).copy()
        top["rank"] = [f"#{idx + 1}" for idx in range(len(top))]
        plot_df = top.melt(
            id_vars=["rank", "method"],
            value_vars=["privacy_score", "utility_score", "objective_score"],
            var_name="metric",
            value_name="score",
        )
        plt.figure(figsize=(10, 5))
        sns.barplot(data=plot_df, x="rank", y="score", hue="metric")
        plt.ylim(0, 1)
        plt.title("Top Masking Configurations")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path

    def model_metric_comparison(self, baseline_ml: pd.DataFrame, masked_ml: pd.DataFrame) -> Path:
        path = self.output_dir / "model_metric_comparison.png"
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        baseline = baseline_ml.assign(dataset="Original")
        masked = masked_ml.assign(dataset="Best Masked")
        plot_df = pd.concat([baseline, masked], ignore_index=True)
        plot_df = plot_df.melt(
            id_vars=["dataset", "model"],
            value_vars=[metric for metric in metrics if metric in plot_df],
            var_name="metric",
            value_name="score",
        )
        plt.figure(figsize=(11, 6))
        sns.barplot(data=plot_df, x="metric", y="score", hue="dataset")
        plt.ylim(0, 1)
        plt.title("Model Utility Metrics")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path

    def information_loss(self, results: pd.DataFrame) -> Path:
        path = self.output_dir / "information_loss_by_method.png"
        plot_df = results.copy()
        plt.figure(figsize=(9, 5))
        sns.boxplot(data=plot_df, x="method", y="information_loss")
        sns.stripplot(data=plot_df, x="method", y="information_loss", color="black", alpha=0.35, size=3)
        plt.ylim(0, 1)
        plt.title("Information Loss by Transformation")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path

    def transformation_impact(self, validation: pd.DataFrame) -> Path:
        path = self.output_dir / "transformation_impact.png"
        metrics = ["ks_statistic", "kl_divergence", "variance_preservation"]
        plot_df = validation.melt(
            id_vars=["feature"],
            value_vars=[metric for metric in metrics if metric in validation],
            var_name="metric",
            value_name="value",
        )
        plt.figure(figsize=(10, 5))
        sns.barplot(data=plot_df, x="feature", y="value", hue="metric")
        plt.title("Statistical Transformation Impact")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path

    def generalization_diagnostics(self, baseline_ml: pd.DataFrame, masked_ml: pd.DataFrame) -> Path:
        path = self.output_dir / "generalization_diagnostics.png"
        baseline = baseline_ml.assign(dataset="Original")
        masked = masked_ml.assign(dataset="Best Masked")
        plot_df = pd.concat([baseline, masked], ignore_index=True)
        plot_df = plot_df.melt(
            id_vars=["dataset", "model"],
            value_vars=["train_accuracy", "validation_accuracy", "accuracy", "cv_accuracy_mean"],
            var_name="metric",
            value_name="score",
        )
        plt.figure(figsize=(11, 6))
        sns.lineplot(data=plot_df, x="metric", y="score", hue="model", style="dataset", marker="o")
        plt.ylim(0, 1)
        plt.title("Train, Validation, Test, and CV Accuracy")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path

    def roc_auc_comparison(self, baseline_ml: pd.DataFrame, masked_ml: pd.DataFrame) -> Path:
        path = self.output_dir / "roc_auc_comparison.png"
        baseline = baseline_ml.assign(dataset="Original")
        masked = masked_ml.assign(dataset="Best Masked")
        plot_df = pd.concat([baseline, masked], ignore_index=True)
        plt.figure(figsize=(8, 5))
        sns.barplot(data=plot_df, x="model", y="roc_auc", hue="dataset")
        plt.ylim(0, 1)
        plt.title("ROC-AUC Comparison")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path
