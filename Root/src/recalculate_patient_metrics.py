"""Recalculate the final patient-level metrics from outer-test predictions.

This standalone script consumes the deidentified, one-row-per-patient table
released with the final 117-patient analysis. It applies each patient's saved
fold-specific validation threshold, calculates the pooled point estimates, and
reproduces the percentile confidence intervals from 5,000 patient-level
bootstrap samples.

Required CSV columns
--------------------
patient_id, outer_fold, true_label, prob_CME, threshold

An optional pred_label column is checked against the prediction reconstructed
from prob_CME >= threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


REQUIRED_COLUMNS = {
    "patient_id",
    "outer_fold",
    "true_label",
    "prob_CME",
    "threshold",
}
METRICS = [
    "accuracy",
    "auroc",
    "auprc",
    "sensitivity",
    "specificity",
    "precision",
    "recall",
    "f1",
]


def safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return AUROC, or NaN when a bootstrap sample contains one class."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, float | int]:
    """Calculate patient-level metrics using row-specific thresholds."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if not (y_true.shape == y_prob.shape == thresholds.shape):
        raise ValueError("Labels, probabilities, and thresholds must have equal shapes.")

    y_pred = (y_prob >= thresholds).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auroc": safe_auroc(y_true, y_prob),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_and_validate_predictions(
    path: Path,
    expected_patients: int,
    expected_positive: int,
    expected_negative: int,
    expected_folds: int,
) -> pd.DataFrame:
    """Load the released prediction table and enforce its statistical contract."""
    predictions = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {', '.join(missing)}")
    if predictions[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required prediction columns contain missing values.")

    predictions = predictions.copy()
    predictions["patient_id"] = predictions["patient_id"].astype(str)
    if predictions["patient_id"].duplicated().any():
        duplicates = int(predictions["patient_id"].duplicated().sum())
        raise ValueError(
            "The released table must contain one row per patient; "
            f"found {duplicates} duplicate patient identifiers."
        )

    for column in ["outer_fold", "true_label", "prob_CME", "threshold"]:
        predictions[column] = pd.to_numeric(predictions[column], errors="raise")
    if not np.allclose(predictions["outer_fold"], predictions["outer_fold"].astype(int)):
        raise ValueError("outer_fold must contain integers.")
    predictions["outer_fold"] = predictions["outer_fold"].astype(int)
    if not np.allclose(predictions["true_label"], predictions["true_label"].astype(int)):
        raise ValueError("true_label must contain integers.")
    predictions["true_label"] = predictions["true_label"].astype(int)

    if not set(predictions["true_label"].unique()).issubset({0, 1}):
        raise ValueError("true_label must contain only 0 and 1.")
    if not predictions["prob_CME"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("prob_CME values must be between 0 and 1.")
    if not predictions["threshold"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("threshold values must be between 0 and 1.")

    if len(predictions) != expected_patients:
        raise ValueError(
            f"Expected {expected_patients} patient rows; found {len(predictions)}."
        )
    positive = int(predictions["true_label"].sum())
    negative = int(len(predictions) - positive)
    if positive != expected_positive or negative != expected_negative:
        raise ValueError(
            "Expected class counts "
            f"{expected_positive} positive/{expected_negative} negative; "
            f"found {positive} positive/{negative} negative."
        )
    folds = sorted(predictions["outer_fold"].unique().tolist())
    if folds != list(range(1, expected_folds + 1)):
        raise ValueError(
            f"Expected outer folds 1 through {expected_folds}; found {folds}."
        )
    thresholds_per_fold = predictions.groupby("outer_fold")["threshold"].nunique()
    if not (thresholds_per_fold == 1).all():
        raise ValueError("Each outer fold must have exactly one applied threshold.")

    computed_predictions = (
        predictions["prob_CME"].to_numpy(dtype=float)
        >= predictions["threshold"].to_numpy(dtype=float)
    ).astype(int)
    if "pred_label" in predictions.columns:
        recorded_predictions = pd.to_numeric(
            predictions["pred_label"], errors="raise"
        ).to_numpy(dtype=int)
        if not np.array_equal(recorded_predictions, computed_predictions):
            raise ValueError(
                "Recorded pred_label values do not equal prob_CME >= threshold."
            )
    predictions["pred_label"] = computed_predictions
    return predictions


def patient_bootstrap_ci(
    predictions: pd.DataFrame,
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the pipeline's patient-level percentile bootstrap."""
    if iterations <= 0:
        raise ValueError("Bootstrap iterations must be positive.")
    rng = np.random.default_rng(seed)
    patient_ids = predictions["patient_id"].unique()
    grouped = {
        patient_id: group.copy()
        for patient_id, group in predictions.groupby("patient_id", sort=False)
    }
    bootstrap_rows: list[dict[str, float | int]] = []
    for iteration in range(iterations):
        sampled_ids = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        bootstrap = pd.concat(
            [grouped[patient_id] for patient_id in sampled_ids],
            ignore_index=True,
        )
        metrics = calculate_metrics(
            bootstrap["true_label"].to_numpy(dtype=int),
            bootstrap["prob_CME"].to_numpy(dtype=float),
            bootstrap["threshold"].to_numpy(dtype=float),
        )
        bootstrap_rows.append({"iteration": iteration, **metrics})

    samples = pd.DataFrame(bootstrap_rows)
    point = calculate_metrics(
        predictions["true_label"].to_numpy(dtype=int),
        predictions["prob_CME"].to_numpy(dtype=float),
        predictions["threshold"].to_numpy(dtype=float),
    )
    confidence_intervals = pd.DataFrame(
        [
            {
                "metric": metric,
                "estimate": float(point[metric]),
                "ci_lower": float(samples[metric].quantile(0.025)),
                "ci_upper": float(samples[metric].quantile(0.975)),
            }
            for metric in METRICS
        ]
    )
    return confidence_intervals, samples


def json_value(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict JSON values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_outputs(
    output_dir: Path,
    predictions: pd.DataFrame,
    point_metrics: dict[str, float | int],
    confidence_intervals: pd.DataFrame,
    bootstrap_samples: pd.DataFrame,
    iterations: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [
        {
            "outer_fold": int(fold),
            "threshold": float(group["threshold"].iloc[0]),
        }
        for fold, group in predictions.groupby("outer_fold", sort=True)
    ]
    payload = {
        "analysis_unit": "patient",
        "threshold_application": "row-specific fold-specific validation threshold",
        "patients": int(len(predictions)),
        "cme_positive_patients": int(predictions["true_label"].sum()),
        "cme_negative_patients": int(
            len(predictions) - predictions["true_label"].sum()
        ),
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(seed),
        "thresholds_by_outer_fold": thresholds,
        "point_metrics": {
            key: json_value(value) for key, value in point_metrics.items()
        },
        "confidence_intervals": confidence_intervals.to_dict(orient="records"),
    }
    with (output_dir / "recalculated_pooled_metrics.json").open("w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    confidence_intervals.to_csv(
        output_dir / "recalculated_patient_bootstrap_95ci.csv", index=False
    )
    bootstrap_samples.to_csv(
        output_dir / "recalculated_bootstrap_metric_samples.csv", index=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Deidentified one-row-per-patient outer-test prediction CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for recalculated metrics and bootstrap outputs.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=951)
    parser.add_argument("--expected-patients", type=int, default=117)
    parser.add_argument("--expected-positive", type=int, default=49)
    parser.add_argument("--expected-negative", type=int, default=68)
    parser.add_argument("--expected-folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_and_validate_predictions(
        args.predictions,
        expected_patients=args.expected_patients,
        expected_positive=args.expected_positive,
        expected_negative=args.expected_negative,
        expected_folds=args.expected_folds,
    )
    point_metrics = calculate_metrics(
        predictions["true_label"].to_numpy(dtype=int),
        predictions["prob_CME"].to_numpy(dtype=float),
        predictions["threshold"].to_numpy(dtype=float),
    )
    confidence_intervals, bootstrap_samples = patient_bootstrap_ci(
        predictions,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )

    print(f"Patients: {len(predictions)}")
    print(
        "Class counts: "
        f"{int(predictions['true_label'].sum())} CME-positive, "
        f"{int((predictions['true_label'] == 0).sum())} CME-negative"
    )
    for metric in METRICS:
        row = confidence_intervals.loc[
            confidence_intervals["metric"] == metric
        ].iloc[0]
        print(
            f"{metric:12s}: {point_metrics[metric]:.6f} "
            f"(95% CI {row.ci_lower:.6f}-{row.ci_upper:.6f})"
        )
    print(
        "Confusion matrix: "
        f"TN={point_metrics['tn']}, FP={point_metrics['fp']}, "
        f"FN={point_metrics['fn']}, TP={point_metrics['tp']}"
    )

    if args.output_dir is not None:
        write_outputs(
            args.output_dir,
            predictions,
            point_metrics,
            confidence_intervals,
            bootstrap_samples,
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        print(f"Saved recalculated outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
