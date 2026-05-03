from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .preprocessing import extract_binary_target, prepare_ml_frame


@dataclass
class StatisticalValidationResult:
    feature: str
    t_statistic: float
    approximate_p_value: float
    ks_statistic: float
    kl_divergence: float
    wasserstein_distance: float
    original_variance: float
    masked_variance: float
    variance_preservation: float
    mean_shift: float


def numeric_projection(df: pd.DataFrame) -> pd.DataFrame:
    projected = pd.DataFrame(index=df.index)
    if "Age" in df:
        projected["Age"] = df["Age"].map(_range_midpoint).fillna(pd.to_numeric(df["Age"], errors="coerce"))
    if "Income" in df:
        income_numeric = pd.to_numeric(df["Income"], errors="coerce")
        if income_numeric.isna().any():
            income_numeric = df["Income"].astype(str).str.lower().map({"low": 0.0, "high": 1.0})
        projected["Income"] = income_numeric
    for column in ["Gender", "Education", "Occupation"]:
        if column in df:
            projected[f"{column}_Code"] = pd.Categorical(df[column].astype(str)).codes
    return projected.apply(pd.to_numeric, errors="coerce").fillna(0)


def statistical_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    matrix = numeric_projection(df)
    return {
        "mean": matrix.mean(),
        "variance": matrix.var(),
        "covariance": matrix.cov().fillna(0),
        "correlation": matrix.corr(numeric_only=True).fillna(0),
    }


def machine_learning_evaluation(
    df: pd.DataFrame,
    target: pd.Series | None = None,
    random_state: int = 42,
    tune_hyperparameters: bool = True,
    include_xgboost: bool = True,
    fast_mode: bool = False,
) -> pd.DataFrame:
    """Train leakage-aware classifiers and return utility/generalization diagnostics.

    The function uses a 60/20/20 train-validation-test split, K-fold CV on the
    training partition, regularized model families, feature selection, and
    train-vs-validation diagnostics to expose overfitting and underfitting.
    """

    y = extract_binary_target(df) if target is None else target.astype(int).reset_index(drop=True)
    x = prepare_ml_frame(df, include_income_feature=False).reset_index(drop=True)
    x = x.reindex(sorted(x.columns), axis=1)

    if y.nunique() < 2 or len(x) < 20:
        return _empty_ml_rows()

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.40,
        random_state=random_state,
        stratify=y,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )

    cv = StratifiedKFold(n_splits=3 if fast_mode else 5, shuffle=True, random_state=random_state)
    rows = []
    specs = _model_specs(random_state, include_xgboost)
    if fast_mode:
        specs = [spec for spec in specs if spec["name"] in {"Logistic Regression", "Random Forest"}]
    for spec in specs:
        model = _select_model(spec, tune_hyperparameters, cv, x_train, y_train)
        model.fit(x_train, y_train)

        train_pred, train_prob = _predict(model, x_train)
        val_pred, val_prob = _predict(model, x_val)
        test_pred, test_prob = _predict(model, x_test)
        train_accuracy = accuracy_score(y_train, train_pred)
        validation_accuracy = accuracy_score(y_val, val_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        if fast_mode:
            cv_scores = np.array([validation_accuracy])
        else:
            cv_scores = cross_val_score(model, x_train, y_train, scoring="accuracy", cv=cv, n_jobs=1)
        validation_loss = _binary_log_loss(y_val, val_prob)
        test_loss = _binary_log_loss(y_test, test_prob)
        generalization_gap = max(0.0, train_accuracy - validation_accuracy)
        overfit_flag = bool(generalization_gap > 0.10 and validation_accuracy < train_accuracy * 0.92)
        underfit_flag = bool(train_accuracy < 0.62 and validation_accuracy < 0.62)
        sensitive_signal = _sensitive_signal_correlation(df, test_prob, x_test.index)

        rows.append(
            {
                "model": spec["name"],
                "accuracy": test_accuracy,
                "precision": precision_score(y_test, test_pred, zero_division=0),
                "recall": recall_score(y_test, test_pred, zero_division=0),
                "f1": f1_score(y_test, test_pred, zero_division=0),
                "rmse": math.sqrt(mean_squared_error(y_test, test_pred)),
                "roc_auc": _safe_roc_auc(y_test, test_prob),
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
                "train_validation_gap": generalization_gap,
                "validation_log_loss": validation_loss,
                "test_log_loss": test_loss,
                "cv_accuracy_mean": float(np.mean(cv_scores)),
                "cv_accuracy_std": float(np.std(cv_scores)),
                "feature_count": _selected_feature_count(model, x_train.shape[1]),
                "overfitting_flag": overfit_flag,
                "underfitting_flag": underfit_flag,
                "sensitive_signal_correlation": sensitive_signal,
                "memorization_risk_flag": bool(overfit_flag or sensitive_signal > 0.65),
                "best_params": json.dumps(_best_params(model), sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def statistical_validation(original: pd.DataFrame, masked: pd.DataFrame) -> pd.DataFrame:
    original_matrix = numeric_projection(original)
    masked_matrix = numeric_projection(masked)
    rows: list[StatisticalValidationResult] = []

    for column in sorted(set(original_matrix.columns).intersection(masked_matrix.columns)):
        a = original_matrix[column].to_numpy(dtype=float)
        b = masked_matrix[column].to_numpy(dtype=float)
        var_a = float(np.var(a, ddof=1))
        var_b = float(np.var(b, ddof=1))
        rows.append(
            StatisticalValidationResult(
                feature=column,
                t_statistic=_welch_t_statistic(a, b),
                approximate_p_value=_normal_approx_p_value(_welch_t_statistic(a, b)),
                ks_statistic=_ks_statistic(a, b),
                kl_divergence=_kl_divergence(a, b),
                wasserstein_distance=_wasserstein_distance(a, b),
                original_variance=var_a,
                masked_variance=var_b,
                variance_preservation=_variance_preservation(var_a, var_b),
                mean_shift=float(np.mean(b) - np.mean(a)),
            )
        )
    return pd.DataFrame([row.__dict__ for row in rows])


def _model_specs(random_state: int, include_xgboost: bool) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "name": "Logistic Regression",
            "estimator": Pipeline(
                [
                    ("select", SelectKBest(score_func=mutual_info_classif, k="all")),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1500,
                            solver="lbfgs",
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            "params": {"select__k": [8, "all"], "model__C": [0.1, 1.0, 3.0]},
        },
        {
            "name": "Random Forest",
            "estimator": Pipeline(
                [
                    ("select", SelectKBest(score_func=mutual_info_classif, k="all")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=160,
                            random_state=random_state,
                            n_jobs=-1,
                            class_weight="balanced_subsample",
                        ),
                    ),
                ]
            ),
            "params": {
                "select__k": [8, "all"],
                "model__max_depth": [6, 10],
                "model__min_samples_leaf": [5, 15],
            },
        },
        {
            "name": "SVM",
            "estimator": Pipeline(
                [
                    ("select", SelectKBest(score_func=mutual_info_classif, k="all")),
                    ("scale", StandardScaler()),
                    ("model", SVC(probability=True, class_weight="balanced", random_state=random_state)),
                ]
            ),
            "params": {"select__k": [8, "all"], "model__C": [0.5, 1.5], "model__gamma": ["scale"]},
        },
    ]

    if include_xgboost:
        try:
            from xgboost import XGBClassifier

            specs.append(
                {
                    "name": "XGBoost",
                    "estimator": Pipeline(
                        [
                            ("select", SelectKBest(score_func=mutual_info_classif, k="all")),
                            (
                                "model",
                                XGBClassifier(
                                    n_estimators=120,
                                    max_depth=3,
                                    learning_rate=0.08,
                                    subsample=0.85,
                                    colsample_bytree=0.85,
                                    eval_metric="logloss",
                                    random_state=random_state,
                                    n_jobs=-1,
                                ),
                            ),
                        ]
                    ),
                    "params": {
                        "select__k": [8, "all"],
                        "model__max_depth": [2, 4],
                        "model__learning_rate": [0.05, 0.1],
                    },
                }
            )
        except Exception:
            pass
    return specs


def _select_model(
    spec: dict[str, Any],
    tune_hyperparameters: bool,
    cv: StratifiedKFold,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    estimator = spec["estimator"]
    if not tune_hyperparameters:
        return estimator

    param_grid = _valid_param_grid(spec["params"], x_train.shape[1])
    search = GridSearchCV(
        estimator,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    return search.best_estimator_


def _valid_param_grid(params: dict[str, list[Any]], feature_count: int) -> dict[str, list[Any]]:
    valid = {}
    for key, values in params.items():
        if key.endswith("__k"):
            valid[key] = [value for value in values if value == "all" or int(value) <= feature_count]
            if not valid[key]:
                valid[key] = ["all"]
        else:
            valid[key] = values
    return valid


def _predict(model: Pipeline, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    predictions = model.predict(x)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        probabilities = 1 / (1 + np.exp(-scores))
    else:
        probabilities = predictions.astype(float)
    return predictions, np.asarray(probabilities, dtype=float)


def _safe_roc_auc(y_true: pd.Series, probabilities: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, probabilities))
    except ValueError:
        return 0.0


def _binary_log_loss(y_true: pd.Series, probabilities: np.ndarray) -> float:
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return float(log_loss(y_true, probabilities))


def _selected_feature_count(model: Pipeline, fallback: int) -> int:
    try:
        selector = model.named_steps.get("select")
        if selector is not None and hasattr(selector, "get_support"):
            return int(selector.get_support().sum())
    except Exception:
        pass
    return int(fallback)


def _best_params(model: Pipeline) -> dict[str, Any]:
    params = {}
    for key, value in model.get_params().items():
        if key.endswith("__C") or key.endswith("__max_depth") or key.endswith("__min_samples_leaf"):
            params[key] = value
        if key.endswith("__learning_rate") or key.endswith("__k") or key.endswith("__gamma"):
            params[key] = value
    return params


def _sensitive_signal_correlation(df: pd.DataFrame, probabilities: np.ndarray, test_index: pd.Index) -> float:
    projection = numeric_projection(df).reindex(test_index)
    sensitive_columns = [column for column in ["Age", "Gender_Code"] if column in projection]
    correlations = []
    for column in sensitive_columns:
        values = projection[column].to_numpy(dtype=float)
        if np.std(values) <= 1e-9 or np.std(probabilities) <= 1e-9:
            continue
        correlations.append(abs(float(np.corrcoef(values, probabilities)[0, 1])))
    return float(max(correlations) if correlations else 0.0)


def _empty_ml_rows() -> pd.DataFrame:
    models = ["Logistic Regression", "Random Forest", "SVM", "XGBoost"]
    return pd.DataFrame(
        [
            {
                "model": model,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "rmse": 1.0,
                "roc_auc": 0.0,
                "train_accuracy": 0.0,
                "validation_accuracy": 0.0,
                "train_validation_gap": 0.0,
                "validation_log_loss": 1.0,
                "test_log_loss": 1.0,
                "cv_accuracy_mean": 0.0,
                "cv_accuracy_std": 0.0,
                "feature_count": 0,
                "overfitting_flag": False,
                "underfitting_flag": True,
                "sensitive_signal_correlation": 0.0,
                "memorization_risk_flag": False,
                "best_params": "{}",
            }
            for model in models
        ]
    )


def _range_midpoint(value: object) -> float:
    text = str(value)
    if "-" not in text:
        return float("nan")
    try:
        left, right = text.split("-", 1)
        return (float(left) + float(right)) / 2
    except ValueError:
        return float("nan")


def _welch_t_statistic(a: np.ndarray, b: np.ndarray) -> float:
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    denom = math.sqrt(var_a / len(a) + var_b / len(b))
    if denom == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / denom)


def _normal_approx_p_value(t_statistic: float) -> float:
    return float(math.erfc(abs(t_statistic) / math.sqrt(2)))


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    values = np.sort(np.unique(np.concatenate([a, b])))
    if len(values) == 0:
        return 0.0
    cdf_a = np.searchsorted(np.sort(a), values, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), values, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _kl_divergence(a: np.ndarray, b: np.ndarray, bins: int = 20) -> float:
    low = min(float(np.min(a)), float(np.min(b)))
    high = max(float(np.max(a)), float(np.max(b)))
    if low == high:
        return 0.0
    p, edges = np.histogram(a, bins=bins, range=(low, high), density=True)
    q, _ = np.histogram(b, bins=edges, density=True)
    p = p + 1e-9
    q = q + 1e-9
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _wasserstein_distance(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import wasserstein_distance

        return float(wasserstein_distance(a, b))
    except Exception:
        a_sorted = np.sort(a)
        b_sorted = np.sort(b)
        n = min(len(a_sorted), len(b_sorted))
        if n == 0:
            return 0.0
        quantiles = np.linspace(0, 1, n)
        a_q = np.quantile(a_sorted, quantiles)
        b_q = np.quantile(b_sorted, quantiles)
        return float(np.mean(np.abs(a_q - b_q)))


def _variance_preservation(original_variance: float, masked_variance: float) -> float:
    denom = max(abs(original_variance), 1e-9)
    return float(np.clip(1 - abs(original_variance - masked_variance) / denom, 0, 1))
