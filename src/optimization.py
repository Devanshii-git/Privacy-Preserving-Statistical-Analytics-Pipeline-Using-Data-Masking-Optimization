from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .analysis import machine_learning_evaluation, statistical_validation
from .evaluation import combined_objective, privacy_metrics, utility_metrics
from .masking import MaskingConfig, MaskingEngine


class OptimizationEngine:
    """Optimize privacy parameters with Optuna when available and robust fallbacks otherwise."""

    def __init__(
        self,
        masking_engine: MaskingEngine,
        quasi_identifiers: list[str],
        privacy_weight: float = 0.50,
        random_state: int = 42,
    ) -> None:
        self.masking_engine = masking_engine
        self.quasi_identifiers = quasi_identifiers
        self.privacy_weight = privacy_weight
        self.random_state = random_state

    def default_grid(self) -> list[MaskingConfig]:
        grid: list[MaskingConfig] = []
        for noise in [0.5, 3.0, 8.0]:
            grid.append(
                MaskingConfig(method="noise", noise_std=noise, gender_mask_probability=min(noise / 20, 0.5))
            )
        for bin_width in [10, 20, 30]:
            grid.append(MaskingConfig(method="generalization", generalization_bin=bin_width))
        for epsilon in [0.5, 1.0, 3.0]:
            grid.append(MaskingConfig(method="differential_privacy", epsilon=epsilon))
        for swap_fraction in [0.05, 0.20]:
            grid.append(MaskingConfig(method="data_swapping", swap_fraction=swap_fraction))
        for strength in [0.05, 0.20]:
            grid.append(MaskingConfig(method="feature_perturbation", perturbation_strength=strength))
        for noise in [1.5, 6.0]:
            for bin_width in [10, 20]:
                for epsilon in [0.5, 2.0]:
                    grid.append(
                        MaskingConfig(
                            method="hybrid",
                            noise_std=noise,
                            generalization_bin=bin_width,
                            epsilon=epsilon,
                            gender_mask_probability=min(noise / 20, 0.5),
                            swap_fraction=0.10,
                            perturbation_strength=0.10,
                        )
                    )
        return grid

    def randomized_search(self, n_iter: int = 24) -> list[MaskingConfig]:
        rng = np.random.default_rng(self.random_state)
        methods = np.array(
            ["noise", "generalization", "differential_privacy", "data_swapping", "feature_perturbation", "hybrid"]
        )
        configs: list[MaskingConfig] = []

        for _ in range(n_iter):
            method = str(rng.choice(methods, p=[0.16, 0.16, 0.20, 0.13, 0.13, 0.22]))
            configs.append(self._sample_config(method, rng))

        anchor_configs = [
            MaskingConfig(method="noise", noise_std=1.5, gender_mask_probability=0.075),
            MaskingConfig(method="generalization", generalization_bin=10),
            MaskingConfig(method="differential_privacy", epsilon=1.0),
            MaskingConfig(method="data_swapping", swap_fraction=0.10),
            MaskingConfig(method="feature_perturbation", perturbation_strength=0.10),
            MaskingConfig(
                method="hybrid",
                noise_std=3.0,
                generalization_bin=15,
                epsilon=2.0,
                gender_mask_probability=0.15,
                hybrid_dp_budget_fraction=0.7,
                swap_fraction=0.10,
                perturbation_strength=0.10,
            ),
        ]
        return self._deduplicate(anchor_configs + configs)

    def genetic_search(self, generations: int = 3, population_size: int = 10) -> list[MaskingConfig]:
        rng = np.random.default_rng(self.random_state + 7)
        methods = ["noise", "generalization", "differential_privacy", "data_swapping", "feature_perturbation", "hybrid"]
        population = [self._sample_config(str(rng.choice(methods)), rng) for _ in range(population_size)]
        configs = population.copy()

        for _ in range(generations):
            children = []
            for _ in range(population_size):
                left, right = rng.choice(population, size=2, replace=True)
                method = str(rng.choice([left.method, right.method]))
                child = MaskingConfig(
                    method=method,
                    noise_std=float(np.clip((left.noise_std + right.noise_std) / 2 + rng.normal(0, 0.7), 0.1, 12)),
                    generalization_bin=int(rng.choice([left.generalization_bin, right.generalization_bin, 5, 10, 15, 20, 25])),
                    epsilon=float(np.clip((left.epsilon + right.epsilon) / 2 + rng.normal(0, 0.35), 0.2, 6)),
                    gender_mask_probability=float(np.clip((left.gender_mask_probability + right.gender_mask_probability) / 2, 0, 0.65)),
                    hybrid_dp_budget_fraction=float(rng.choice([0.5, 0.6, 0.7, 0.8])),
                    swap_fraction=float(np.clip((left.swap_fraction + right.swap_fraction) / 2 + rng.normal(0, 0.03), 0, 0.45)),
                    perturbation_strength=float(
                        np.clip((left.perturbation_strength + right.perturbation_strength) / 2 + rng.normal(0, 0.03), 0, 0.45)
                    ),
                )
                children.append(child)
            population = children
            configs.extend(children)
        return self._deduplicate(configs)

    def optuna_search(
        self,
        original: pd.DataFrame,
        baseline_ml: pd.DataFrame,
        target: pd.Series,
        n_trials: int = 20,
    ) -> list[dict[str, object]]:
        try:
            import optuna
        except Exception:
            return []

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        trial_rows: list[dict[str, object]] = []

        def objective(trial: object) -> float:
            method = trial.suggest_categorical(
                "method",
                ["noise", "generalization", "differential_privacy", "data_swapping", "feature_perturbation", "hybrid"],
            )
            config = MaskingConfig(
                method=method,
                noise_std=trial.suggest_float("noise_std", 0.1, 12.0) if method in ["noise", "hybrid"] else 0.0,
                generalization_bin=trial.suggest_categorical("generalization_bin", [5, 10, 15, 20, 25, 30]) if method in ["generalization", "hybrid"] else 10,
                epsilon=trial.suggest_float("epsilon", 0.2, 6.0, log=True) if method in ["differential_privacy", "hybrid"] else 1.0,
                gender_mask_probability=trial.suggest_float("gender_mask_probability", 0.0, 0.65) if method in ["noise", "generalization", "hybrid"] else 0.0,
                hybrid_dp_budget_fraction=trial.suggest_float("hybrid_dp_budget_fraction", 0.5, 0.85) if method == "hybrid" else 0.7,
                swap_fraction=trial.suggest_float("swap_fraction", 0.0, 0.45) if method in ["data_swapping", "hybrid"] else 0.0,
                perturbation_strength=trial.suggest_float("perturbation_strength", 0.0, 0.45) if method in ["feature_perturbation", "hybrid"] else 0.0,
            )
            row, _ = self.evaluate_config(original, baseline_ml, target, config, optimizer_name="optuna")
            trial_rows.append(row)
            return float(row["objective_score"])

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return trial_rows

    def run(
        self,
        original: pd.DataFrame,
        baseline_ml: pd.DataFrame,
        target: pd.Series,
        configs: list[MaskingConfig] | None = None,
        n_trials: int = 20,
    ) -> tuple[pd.DataFrame, pd.DataFrame, MaskingConfig]:
        rows: list[dict[str, object]] = []
        best_score = -float("inf")
        best_masked = original.copy()
        best_config = MaskingConfig(method="noise", noise_std=0.0)

        candidate_configs = configs or self._deduplicate(
            self.default_grid() + self.randomized_search(n_iter=8) + self.genetic_search(generations=1, population_size=6)
        )
        for config in candidate_configs:
            row, masked = self.evaluate_config(original, baseline_ml, target, config, optimizer_name="grid_random_genetic")
            rows.append(row)
            if float(row["objective_score"]) > best_score:
                best_score = float(row["objective_score"])
                best_masked = masked
                best_config = config

        optuna_rows = self.optuna_search(original, baseline_ml, target, n_trials=n_trials)
        for row in optuna_rows:
            rows.append(row)
            if float(row["objective_score"]) > best_score:
                best_score = float(row["objective_score"])
                best_config = MaskingConfig(
                    method=str(row["method"]),
                    noise_std=float(row["noise_std"]),
                    generalization_bin=int(row["generalization_bin"]),
                    epsilon=float(row["epsilon"]),
                    gender_mask_probability=float(row["gender_mask_probability"]),
                    hybrid_dp_budget_fraction=float(row["hybrid_dp_budget_fraction"]),
                    swap_fraction=float(row["swap_fraction"]),
                    perturbation_strength=float(row["perturbation_strength"]),
                )
                best_masked = self.masking_engine.apply(original, best_config)

        results = pd.DataFrame(rows).sort_values("objective_score", ascending=False).reset_index(drop=True)
        return results, best_masked, best_config

    def evaluate_config(
        self,
        original: pd.DataFrame,
        baseline_ml: pd.DataFrame,
        target: pd.Series,
        config: MaskingConfig,
        optimizer_name: str,
    ) -> tuple[dict[str, object], pd.DataFrame]:
        masked = self.masking_engine.apply(original, config)
        masked_ml = machine_learning_evaluation(
            masked,
            target=target,
            random_state=self.random_state,
            tune_hyperparameters=False,
            include_xgboost=False,
            fast_mode=True,
        )
        privacy = privacy_metrics(original, masked, self.quasi_identifiers)
        utility = utility_metrics(original, masked, baseline_ml, masked_ml)
        score = combined_objective(
            privacy["privacy_score"],
            utility["utility_score"],
            privacy_weight=self.privacy_weight,
        )
        validation = statistical_validation(original, masked)
        mean_kl = float(validation["kl_divergence"].mean()) if not validation.empty else 0.0
        mean_ks = float(validation["ks_statistic"].mean()) if not validation.empty else 0.0
        mean_wasserstein = float(validation["wasserstein_distance"].mean()) if not validation.empty else 0.0
        variance_preservation = float(validation["variance_preservation"].mean()) if not validation.empty else 0.0

        row: dict[str, object] = {
            **asdict(config),
            "optimizer": optimizer_name,
            "config_label": config.label(),
            **privacy,
            **utility,
            "objective_score": score,
            "mean_ks_statistic": mean_ks,
            "mean_kl_divergence": mean_kl,
            "mean_wasserstein_distance": mean_wasserstein,
            "mean_variance_preservation": variance_preservation,
            "overfitting_rate": float(masked_ml["overfitting_flag"].mean()),
            "underfitting_rate": float(masked_ml["underfitting_flag"].mean()),
            "memorization_risk_rate": float(masked_ml["memorization_risk_flag"].mean()),
        }
        return row, masked

    def _sample_config(self, method: str, rng: np.random.Generator) -> MaskingConfig:
        noise = float(rng.choice([0.25, 0.5, 1.0, 1.5, 3.0, 6.0, 10.0]))
        bin_width = int(rng.choice([5, 10, 15, 20, 25, 30]))
        epsilon = float(rng.choice([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]))
        swap_fraction = float(rng.choice([0.0, 0.05, 0.10, 0.20, 0.35]))
        perturbation_strength = float(rng.choice([0.0, 0.05, 0.10, 0.20, 0.35]))
        return MaskingConfig(
            method=method,
            noise_std=noise,
            generalization_bin=bin_width,
            epsilon=epsilon,
            gender_mask_probability=float(min(noise / 20, 0.5)),
            hybrid_dp_budget_fraction=float(rng.choice([0.5, 0.6, 0.7, 0.8])),
            swap_fraction=swap_fraction,
            perturbation_strength=perturbation_strength,
        )

    def _deduplicate(self, configs: list[MaskingConfig]) -> list[MaskingConfig]:
        seen: set[str] = set()
        unique: list[MaskingConfig] = []
        for config in configs:
            label = config.label()
            if label not in seen:
                unique.append(config)
                seen.add(label)
        return unique
