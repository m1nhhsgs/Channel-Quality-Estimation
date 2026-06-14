#!/usr/bin/env python3
"""
Train an XGBoost regressor to predict future ESP-NOW packet loss.

Input CSV must be created with:
    python extract_xgboost_features.py raw.csv -o ml_dataset.csv --payload-size 28 --include-labels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET = "packet_loss_future_2s"
TRACE_COLUMNS = ["scenario_id", "window_start", "window_end"]
DEFAULT_DROP_COLUMNS = TRACE_COLUMNS + [TARGET]
STATE_LABELS = ["Good", "Critical"]


def is_nonempty(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def channel_state(loss_percent: float) -> str:
    return "Critical" if loss_percent >= 10.0 else "Good"


def split_dataset(df: pd.DataFrame, test_size: float, random_state: int):
    if "scenario_id" in df.columns and df["scenario_id"].nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, groups=df["scenario_id"]))
        return df.iloc[train_idx].copy(), df.iloc[test_idx].copy(), "group_by_scenario_id"

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    return train_df.copy(), test_df.copy(), "random_fallback_single_scenario"


def rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred) ** 0.5


def plot_training_curves(evals_result: dict, figures_dir: Path) -> Path | None:
    if not evals_result:
        return None

    train_key = "validation_0"
    test_key = "validation_1"
    metrics = sorted(set(evals_result.get(train_key, {})) | set(evals_result.get(test_key, {})))
    if not metrics:
        return None

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric in evals_result.get(train_key, {}):
            ax.plot(evals_result[train_key][metric], label=f"train/{metric}", linewidth=1.8)
        if metric in evals_result.get(test_key, {}):
            ax.plot(evals_result[test_key][metric], label=f"test/{metric}", linewidth=1.8)
        ax.set_title(metric.upper())
        ax.set_xlabel("Boosting round")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    path = figures_dir / "training_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_prediction_vs_actual(y_test, pred: list[float], figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, pred, s=10, alpha=0.45)
    max_value = max(float(max(y_test)), max(pred), 1.0)
    ax.plot([0, max_value], [0, max_value], color="crimson", linestyle="--", linewidth=1.4)
    ax.set_title("Prediction vs Actual")
    ax.set_xlabel("Actual packet_loss_future_2s (%)")
    ax.set_ylabel("Predicted packet_loss_future_2s (%)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = figures_dir / "prediction_vs_actual.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_residuals(y_test, pred: list[float], figures_dir: Path) -> Path:
    residuals = [float(actual) - float(predicted) for actual, predicted in zip(y_test, pred)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=50, color="#4477aa", alpha=0.85)
    ax.axvline(0, color="crimson", linestyle="--", linewidth=1.4)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Actual - Predicted")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = figures_dir / "residual_histogram.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_confusion_matrix_figure(cm: list[list[int]], figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("State Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(STATE_LABELS)), STATE_LABELS)
    ax.set_yticks(range(len(STATE_LABELS)), STATE_LABELS)
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = figures_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_feature_importance(importances: pd.DataFrame, figures_dir: Path) -> Path:
    top = importances.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["importance"], color="#228833")
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = figures_dir / "feature_importance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_results_summary(
    evals_result: dict,
    y_test,
    pred: list[float],
    cm: list[list[int]],
    importances: pd.DataFrame,
    metrics: dict,
    figures_dir: Path,
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    for metric in ("rmse", "mae"):
        if metric in evals_result.get("validation_0", {}):
            axes[0].plot(evals_result["validation_0"][metric], label=f"train/{metric}")
        if metric in evals_result.get("validation_1", {}):
            axes[0].plot(evals_result["validation_1"][metric], label=f"test/{metric}")
    axes[0].set_title("Train/Test Curves")
    axes[0].set_xlabel("Boosting round")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    max_value = max(float(max(y_test)), max(pred), 1.0)
    axes[1].scatter(y_test, pred, s=8, alpha=0.4)
    axes[1].plot([0, max_value], [0, max_value], color="crimson", linestyle="--")
    axes[1].set_title("Prediction vs Actual")
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].grid(alpha=0.25)

    residuals = [float(actual) - float(predicted) for actual, predicted in zip(y_test, pred)]
    axes[2].hist(residuals, bins=50, color="#4477aa", alpha=0.85)
    axes[2].axvline(0, color="crimson", linestyle="--")
    axes[2].set_title("Residuals")
    axes[2].grid(alpha=0.25)

    axes[3].imshow(cm, cmap="Blues")
    axes[3].set_title("Confusion Matrix")
    axes[3].set_xticks(range(len(STATE_LABELS)), STATE_LABELS)
    axes[3].set_yticks(range(len(STATE_LABELS)), STATE_LABELS)
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            axes[3].text(j, i, str(value), ha="center", va="center", fontsize=9)

    top = importances.head(10).iloc[::-1]
    axes[4].barh(top["feature"], top["importance"], color="#228833")
    axes[4].set_title("Top Feature Importance")
    axes[4].grid(axis="x", alpha=0.25)

    axes[5].axis("off")
    summary = (
        f"MAE: {metrics['mae']:.4f}\n"
        f"RMSE: {metrics['rmse']:.4f}\n"
        f"R2: {metrics['r2']:.4f}\n"
        f"State acc: {metrics['state_accuracy']:.4f}\n"
        f"F1 macro: {metrics['state_f1_macro']:.4f}\n"
        f"Train rows: {metrics['rows_train']}\n"
        f"Test rows: {metrics['rows_test']}"
    )
    axes[5].text(0.05, 0.95, summary, va="top", fontsize=13, family="monospace")
    axes[5].set_title("Metrics Summary")

    fig.tight_layout()
    path = figures_dir / "results.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost model for ESP-NOW channel-quality prediction.")
    parser.add_argument("dataset", nargs="?", help="Feature CSV with packet_loss_future_2s label.")
    parser.add_argument("--train-csv", help="Pre-split train CSV.")
    parser.add_argument("--test-csv", help="Pre-split test CSV.")
    parser.add_argument("--model-dir", default="models", help="Directory for saved model/config.")
    parser.add_argument("--results-dir", default="results", help="Directory for metrics and predictions.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    return parser.parse_args()


def load_labeled_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise SystemExit(f"{path}: missing target column '{TARGET}'. Recreate dataset with --include-labels.")
    return df.dropna(subset=[TARGET]).copy()


def prepare_feature_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[list[str], list[str], list[str], list[str]]:
    candidate_feature_columns = [
        col for col in train_df.columns
        if col not in DEFAULT_DROP_COLUMNS and col in test_df.columns
    ]

    numeric_feature_columns: list[str] = []
    categorical_feature_columns: list[str] = []
    dropped_empty_features: list[str] = []

    for col in candidate_feature_columns:
        has_train_value = is_nonempty(train_df[col]).any()
        has_test_value = is_nonempty(test_df[col]).any()
        if not has_train_value and not has_test_value:
            dropped_empty_features.append(col)
            continue

        train_numeric = pd.to_numeric(train_df[col], errors="coerce")
        test_numeric = pd.to_numeric(test_df[col], errors="coerce")
        if train_numeric.notna().any() and test_numeric.notna().any():
            train_df[col] = train_numeric
            test_df[col] = test_numeric
            numeric_feature_columns.append(col)
            continue

        for frame in (train_df, test_df):
            frame[col] = (
                frame[col]
                .fillna("unknown")
                .astype(str)
                .str.strip()
                .replace("", "unknown")
            )
        categorical_feature_columns.append(col)

    feature_columns = numeric_feature_columns + categorical_feature_columns
    if not feature_columns:
        raise SystemExit("No usable numeric feature columns found.")
    return feature_columns, numeric_feature_columns, categorical_feature_columns, dropped_empty_features


def feature_defaults(
    train_df: pd.DataFrame,
    numeric_feature_columns: list[str],
    categorical_feature_columns: list[str],
) -> dict[str, float | str]:
    defaults: dict[str, float | str] = {}
    for col in numeric_feature_columns:
        median_value = train_df[col].median()
        defaults[col] = 0.0 if pd.isna(median_value) else float(median_value)
    for col in categorical_feature_columns:
        values = train_df[col].fillna("unknown").astype(str).str.strip().replace("", "unknown")
        mode_values = values.mode()
        defaults[col] = str(mode_values.iloc[0]) if not mode_values.empty else "unknown"
    return defaults


def transformed_feature_names(
    preprocessor: ColumnTransformer,
    feature_columns: list[str],
) -> list[str]:
    try:
        names = list(preprocessor.get_feature_names_out())
    except Exception:  # noqa: BLE001 - fallback for older sklearn behavior.
        return feature_columns

    cleaned: list[str] = []
    for name in names:
        if "__" in name:
            name = name.split("__", 1)[1]
        cleaned.append(name)
    return cleaned


def main() -> int:
    args = parse_args()
    if bool(args.train_csv) != bool(args.test_csv):
        raise SystemExit("Use both --train-csv and --test-csv, or neither.")
    if not args.dataset and not args.train_csv:
        raise SystemExit("Provide a dataset CSV, or provide --train-csv and --test-csv.")

    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.train_csv and args.test_csv:
        train_df = load_labeled_csv(args.train_csv)
        test_df = load_labeled_csv(args.test_csv)
        split_method = "pre_split_csv"
    else:
        df = load_labeled_csv(args.dataset)
        train_df, test_df, split_method = split_dataset(df, args.test_size, args.random_state)

    (
        feature_columns,
        numeric_feature_columns,
        categorical_feature_columns,
        dropped_empty_features,
    ) = prepare_feature_columns(train_df, test_df)
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET].astype(float)
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET].astype(float)

    transformers = []
    if numeric_feature_columns:
        transformers.append(
            ("num", SimpleImputer(strategy="median"), numeric_feature_columns)
        )
    if categorical_feature_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_feature_columns,
            )
        )
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    model = XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric=["rmse", "mae"],
        random_state=args.random_state,
        n_jobs=-1,
    )

    X_train_prepared = preprocessor.fit_transform(X_train)
    X_test_prepared = preprocessor.transform(X_test)
    model.fit(
        X_train_prepared,
        y_train,
        eval_set=[(X_train_prepared, y_train), (X_test_prepared, y_test)],
        verbose=False,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pred = pipeline.predict(X_test)
    pred = [max(0.0, float(value)) for value in pred]

    actual_state = [channel_state(value) for value in y_test]
    pred_state = [channel_state(value) for value in pred]

    metrics = {
        "target": TARGET,
        "split_method": split_method,
        "rows_total": int(len(train_df) + len(test_df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "feature_columns": feature_columns,
        "numeric_feature_columns": numeric_feature_columns,
        "categorical_feature_columns": categorical_feature_columns,
        "dropped_empty_features": dropped_empty_features,
        "mae": mean_absolute_error(y_test, pred),
        "rmse": rmse(y_test, pred),
        "r2": r2_score(y_test, pred),
        "state_accuracy": accuracy_score(actual_state, pred_state),
        "state_f1_macro": f1_score(actual_state, pred_state, average="macro", zero_division=0),
        "state_labels": STATE_LABELS,
        "confusion_matrix": confusion_matrix(
            actual_state,
            pred_state,
            labels=STATE_LABELS,
        ).tolist(),
    }

    joblib.dump(pipeline, model_dir / "xgboost_packet_loss_pipeline.joblib")
    pipeline.named_steps["model"].save_model(model_dir / "xgboost_model.json")

    config = {
        "target": TARGET,
        "feature_columns": feature_columns,
        "numeric_feature_columns": numeric_feature_columns,
        "categorical_feature_columns": categorical_feature_columns,
        "feature_defaults": feature_defaults(
            train_df,
            numeric_feature_columns,
            categorical_feature_columns,
        ),
        "dropped_empty_features": dropped_empty_features,
        "trace_columns": TRACE_COLUMNS,
        "state_thresholds": {
            "Good": "packet_loss < 10%",
            "Critical": "packet_loss >= 10%",
        },
    }
    (model_dir / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    predictions = test_df[TRACE_COLUMNS].copy() if all(col in test_df.columns for col in TRACE_COLUMNS) else pd.DataFrame()
    predictions["actual_packet_loss_future_2s"] = list(y_test)
    predictions["predicted_packet_loss_future_2s"] = pred
    predictions["actual_state"] = actual_state
    predictions["predicted_state"] = pred_state
    predictions.to_csv(results_dir / "test_predictions.csv", index=False)

    importance_names = transformed_feature_names(preprocessor, feature_columns)
    importance_values = pipeline.named_steps["model"].feature_importances_
    if len(importance_names) != len(importance_values):
        importance_names = [f"feature_{idx}" for idx in range(len(importance_values))]

    importances = pd.DataFrame(
        {
            "feature": importance_names,
            "importance": importance_values,
        }
    ).sort_values("importance", ascending=False)
    importances.to_csv(results_dir / "feature_importance.csv", index=False)

    evals_result = pipeline.named_steps["model"].evals_result()
    eval_rows = []
    rounds = 0
    if evals_result:
        rounds = max(len(values) for dataset in evals_result.values() for values in dataset.values())
    for idx in range(rounds):
        row = {"round": idx}
        for dataset_name, dataset_metrics in evals_result.items():
            label = "train" if dataset_name == "validation_0" else "test"
            for metric_name, values in dataset_metrics.items():
                if idx < len(values):
                    row[f"{label}_{metric_name}"] = values[idx]
        eval_rows.append(row)
    if eval_rows:
        pd.DataFrame(eval_rows).to_csv(results_dir / "training_history.csv", index=False)

    figure_paths = [
        plot_training_curves(evals_result, figures_dir),
        plot_prediction_vs_actual(y_test, pred, figures_dir),
        plot_residuals(y_test, pred, figures_dir),
        plot_confusion_matrix_figure(metrics["confusion_matrix"], figures_dir),
        plot_feature_importance(importances, figures_dir),
        plot_results_summary(
            evals_result,
            y_test,
            pred,
            metrics["confusion_matrix"],
            importances,
            metrics,
            figures_dir,
        ),
    ]
    metrics["figure_files"] = [str(path) for path in figure_paths if path is not None]
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model to {model_dir / 'xgboost_packet_loss_pipeline.joblib'}")
    print(f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")
    print(f"State accuracy={metrics['state_accuracy']:.4f}, F1 macro={metrics['state_f1_macro']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
