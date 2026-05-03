from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import numeric_projection


def privacy_metrics(original: pd.DataFrame, masked: pd.DataFrame, quasi_identifiers: list[str]) -> dict[str, float]:
    qis = [column for column in quasi_identifiers if column in masked.columns]
    if not qis:
        return {
            "k_anonymity": 0.0,
            "average_equivalence_class_size": 0.0,
            "reidentification_risk": 1.0,
            "attribute_disclosure_risk": 1.0,
            "membership_inference_risk": 1.0,
            "information_leakage": 1.0,
            "privacy_score": 0.0,
            "privacy_loss": 1.0,
        }

    masked_qis = masked[qis].astype(str)
    grouped = masked_qis.groupby(qis, dropna=False).size()
    per_record_group_size = masked_qis.merge(
        grouped.rename("group_size").reset_index(),
        on=qis,
        how="left",
    )["group_size"]

    k_anonymity = float(grouped.min()) if len(grouped) else 0.0
    average_equivalence_class_size = float(grouped.mean()) if len(grouped) else 0.0
    reidentification_risk = float((1 / per_record_group_size).mean()) if len(per_record_group_size) else 1.0
    attribute_disclosure_risk = _attribute_disclosure_risk(original, masked, qis)
    membership_inference_risk = float(np.clip(0.65 * reidentification_risk + 0.35 * attribute_disclosure_risk, 0, 1))
    information_leakage = float(np.clip(0.55 * attribute_disclosure_risk + 0.45 * reidentification_risk, 0, 1))

    k_score = min(k_anonymity / 10, 1.0)
    equivalence_score = min(average_equivalence_class_size / 25, 1.0)
    privacy_score = float(
        np.clip(
            0.30 * k_score
            + 0.20 * equivalence_score
            + 0.20 * (1 - reidentification_risk)
            + 0.15 * (1 - attribute_disclosure_risk)
            + 0.15 * (1 - information_leakage),
            0,
            1,
        )
    )

    return {
        "k_anonymity": k_anonymity,
        "average_equivalence_class_size": average_equivalence_class_size,
        "reidentification_risk": reidentification_risk,
        "attribute_disclosure_risk": attribute_disclosure_risk,
        "membership_inference_risk": membership_inference_risk,
        "information_leakage": information_leakage,
        "privacy_score": privacy_score,
        "privacy_loss": 1 - privacy_score,
    }


def utility_metrics(
    original: pd.DataFrame,
    masked: pd.DataFrame,
    baseline_ml: pd.DataFrame,
    masked_ml: pd.DataFrame,
) -> dict[str, float]:
    baseline_accuracy = float(baseline_ml["accuracy"].mean())
    masked_accuracy = float(masked_ml["accuracy"].mean())
    baseline_auc = float(baseline_ml["roc_auc"].mean()) if "roc_auc" in baseline_ml else baseline_accuracy
    masked_auc = float(masked_ml["roc_auc"].mean()) if "roc_auc" in masked_ml else masked_accuracy
    accuracy_drop = max(0.0, baseline_accuracy - masked_accuracy)
    auc_drop = max(0.0, baseline_auc - masked_auc)
    accuracy_retention = min(masked_accuracy / baseline_accuracy, 1.0) if baseline_accuracy else 0.0
    auc_retention = min(masked_auc / baseline_auc, 1.0) if baseline_auc else 0.0
    correlation_preservation = _correlation_preservation(original, masked)
    covariance_preservation = _covariance_preservation(original, masked)
    statistical_similarity = _statistical_similarity(original, masked)
    information_loss = _information_loss(original, masked)
    generalization_quality = _generalization_quality(masked_ml)
    utility_score = float(
        np.clip(
            0.25 * accuracy_retention
            + 0.15 * auc_retention
            + 0.18 * correlation_preservation
            + 0.12 * covariance_preservation
            + 0.15 * statistical_similarity
            + 0.10 * generalization_quality
            + 0.05 * (1 - information_loss),
            0,
            1,
        )
    )

    return {
        "baseline_accuracy": baseline_accuracy,
        "masked_accuracy": masked_accuracy,
        "baseline_roc_auc": baseline_auc,
        "masked_roc_auc": masked_auc,
        "accuracy_drop": accuracy_drop,
        "roc_auc_drop": auc_drop,
        "information_loss": information_loss,
        "correlation_preservation": correlation_preservation,
        "covariance_preservation": covariance_preservation,
        "statistical_similarity": statistical_similarity,
        "generalization_quality": generalization_quality,
        "utility_score": utility_score,
        "utility_loss": 1 - utility_score,
    }


def combined_objective(privacy_score: float, utility_score: float, privacy_weight: float = 0.50) -> float:
    privacy_loss = 1 - privacy_score
    balance_bonus = 1 - abs(privacy_score - utility_score)
    return float(utility_score - privacy_loss + privacy_weight * 0.10 * balance_bonus)


def _attribute_disclosure_risk(original: pd.DataFrame, masked: pd.DataFrame, sensitive_columns: list[str]) -> float:
    risks = []
    original_projection = numeric_projection(original)
    masked_projection = numeric_projection(masked)
    for column in ["Age", "Income"]:
        if column in sensitive_columns and column in original_projection and column in masked_projection:
            a = original_projection[column].to_numpy(dtype=float)
            b = masked_projection[column].to_numpy(dtype=float)
            data_range = max(np.max(a) - np.min(a), 1e-9)
            normalized_distance = np.mean(np.abs(a - b)) / data_range
            risks.append(float(np.clip(1 - normalized_distance, 0, 1)))

    for column in ["Gender"]:
        if column in sensitive_columns and column in original.columns and column in masked.columns:
            risks.append(float((original[column].astype(str) == masked[column].astype(str)).mean()))
    return float(np.clip(np.mean(risks) if risks else 1.0, 0, 1))


def _information_loss(original: pd.DataFrame, masked: pd.DataFrame) -> float:
    losses = []
    original_projection = numeric_projection(original)
    masked_projection = numeric_projection(masked)
    for column in set(original_projection.columns).intersection(masked_projection.columns):
        a = original_projection[column].to_numpy(dtype=float)
        b = masked_projection[column].to_numpy(dtype=float)
        data_range = max(np.max(a) - np.min(a), 1e-9)
        losses.append(float(np.mean(np.abs(a - b)) / data_range))

    for column in ["Gender", "Education", "Occupation"]:
        if column in original.columns and column in masked.columns:
            losses.append(float((original[column].astype(str) != masked[column].astype(str)).mean()))
    return float(np.clip(np.mean(losses) if losses else 0.0, 0, 1))


def _correlation_preservation(original: pd.DataFrame, masked: pd.DataFrame) -> float:
    original_corr = numeric_projection(original).corr().fillna(0)
    masked_corr = numeric_projection(masked).corr().fillna(0)
    return _matrix_preservation(original_corr, masked_corr)


def _covariance_preservation(original: pd.DataFrame, masked: pd.DataFrame) -> float:
    original_cov = numeric_projection(original).cov().fillna(0)
    masked_cov = numeric_projection(masked).cov().fillna(0)
    return _matrix_preservation(original_cov, masked_cov)


def _matrix_preservation(a_df: pd.DataFrame, b_df: pd.DataFrame) -> float:
    columns = sorted(set(a_df.columns).intersection(b_df.columns))
    if len(columns) < 2:
        return 0.0
    a = a_df.loc[columns, columns].to_numpy()
    b = b_df.loc[columns, columns].to_numpy()
    upper = np.triu_indices_from(a, k=1)
    denom = max(float(np.mean(np.abs(a[upper]))), 1e-9)
    diff = np.mean(np.abs(a[upper] - b[upper])) / denom
    return float(np.clip(1 - diff, 0, 1))


def _statistical_similarity(original: pd.DataFrame, masked: pd.DataFrame) -> float:
    original_projection = numeric_projection(original)
    masked_projection = numeric_projection(masked)
    similarities = []
    for column in set(original_projection.columns).intersection(masked_projection.columns):
        a = original_projection[column].to_numpy(dtype=float)
        b = masked_projection[column].to_numpy(dtype=float)
        scale = max(np.std(a), 1e-9)
        normalized_mean_gap = abs(np.mean(a) - np.mean(b)) / scale
        normalized_std_gap = abs(np.std(a) - np.std(b)) / scale
        similarities.append(np.exp(-(normalized_mean_gap + normalized_std_gap) / 2))
    return float(np.clip(np.mean(similarities) if similarities else 0.0, 0, 1))


def _generalization_quality(masked_ml: pd.DataFrame) -> float:
    if masked_ml.empty:
        return 0.0
    gap = float(masked_ml["train_validation_gap"].mean()) if "train_validation_gap" in masked_ml else 0.0
    cv_std = float(masked_ml["cv_accuracy_std"].mean()) if "cv_accuracy_std" in masked_ml else 0.0
    overfit_rate = float(masked_ml["overfitting_flag"].mean()) if "overfitting_flag" in masked_ml else 0.0
    underfit_rate = float(masked_ml["underfitting_flag"].mean()) if "underfitting_flag" in masked_ml else 0.0
    penalty = 0.45 * min(gap / 0.2, 1) + 0.20 * min(cv_std / 0.08, 1) + 0.20 * overfit_rate + 0.15 * underfit_rate
    return float(np.clip(1 - penalty, 0, 1))
