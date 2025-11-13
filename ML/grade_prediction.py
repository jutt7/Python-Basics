"""
Predict final (G3) student grades with a higher fidelity model.

The training pipeline engineers informative features, encodes categoricals,
and fits a tuned RandomForest regressor. Pass ``--predict-json`` with either
an inline JSON string or a path to a JSON file describing one student to get a
custom prediction from the trained model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split


DATA_PATH = Path(__file__).with_name("student.csv")
# Columns that should be numeric but appear as quoted strings in the CSV.
NUMERIC_COLUMNS = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
    "G2",
    "G3",
]
GRADE_RANGE = (0, 20)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV once so everything else can be unit-tested easily."""
    return pd.read_csv(path)


def _clean_common_fields(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "absences" in data.columns:
        data["absences"] = data["absences"].clip(lower=0)

    for grade_col in ("G1", "G2", "G3"):
        if grade_col in data.columns:
            data[grade_col] = data[grade_col].clip(*GRADE_RANGE)

    if {"G1", "G2"}.issubset(data.columns):
        data["grade_average"] = (data["G1"] + data["G2"]) / 2
        data["grade_trend"] = data["G2"] - data["G1"]

    return data


def _encode_features(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = df.select_dtypes(include=["object"]).columns
    if categorical_cols.empty:
        return df
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean/engineer features and one-hot encode categoricals."""
    data = _clean_common_fields(df)
    data.dropna(subset=["G3"], inplace=True)

    y = data["G3"].astype(int)
    X = data.drop(columns=["G3"])

    X = _encode_features(X)
    X = X.fillna(0)
    return X, y


def preprocess_single(sample: dict[str, object], reference_columns: pd.Index) -> pd.DataFrame:
    """Apply the training-time preprocessing to a single custom record."""
    data = pd.DataFrame([sample])
    data = _clean_common_fields(data)
    if "G3" in data.columns:
        data = data.drop(columns=["G3"])

    features = _encode_features(data)
    features = features.fillna(0)
    features = features.reindex(columns=reference_columns, fill_value=0)
    return features


def build_model() -> GridSearchCV:
    """Hyper-parameter search over a RandomForest regressor."""
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [200, 400, 600],
        "max_depth": [12, 18, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    return GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )


def evaluate_predictions(
    y_true: pd.Series, y_pred: np.ndarray
) -> dict[str, float | np.ndarray | str]:
    """Return regression metrics plus accuracy after rounding."""
    rounded_pred = np.clip(np.rint(y_pred), *GRADE_RANGE).astype(int)
    metrics = {
        "rounded_accuracy": accuracy_score(y_true, rounded_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, rounded_pred),
        "classification_report": classification_report(
            y_true, rounded_pred, zero_division=0
        ),
    }
    return metrics


def show_model_summary(model: RandomForestRegressor, feature_names: pd.Index) -> None:
    """Print the most influential features to understand what drives accuracy."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(10)
    print("\nTop predictive features:")
    print(top_features.to_string(float_format=lambda x: f"{x:.4f}"))


def load_custom_input(arg_value: str) -> dict[str, object]:
    """Parse JSON input either from a path or inline string."""
    candidate_path = Path(arg_value)
    script_dir = Path(__file__).resolve().parent

    for possible in (candidate_path, script_dir / arg_value):
        if possible.exists():
            return json.loads(possible.read_text())

    return json.loads(arg_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and query the grade model")
    parser.add_argument(
        "--predict-json",
        dest="predict_json",
        help="JSON string or path describing one student for prediction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_data = load_data()
    X, y = preprocess_data(raw_data)
    feature_names = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    grid_search = build_model()
    grid_search.fit(X_train, y_train)
    best_model: RandomForestRegressor = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    predictions = best_model.predict(X_test)
    metrics = evaluate_predictions(y_test, predictions)

    print(f"\nRounded accuracy: {metrics['rounded_accuracy'] * 100:.2f}%")
    print(f"MAE: {metrics['mae']:.3f}")
    print(f"R^2: {metrics['r2']:.3f}")
    print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
    print(f"\nClassification Report:\n{metrics['classification_report']}")

    show_model_summary(best_model, feature_names)

    if args.predict_json:
        sample = load_custom_input(args.predict_json)
        sample_features = preprocess_single(sample, feature_names)
        sample_prediction = best_model.predict(sample_features)[0]
        rounded = int(np.clip(np.rint(sample_prediction), *GRADE_RANGE))
        print("\nCustom input prediction:")
        print(json.dumps(sample, indent=2))
        print(
            f"Predicted G3: {sample_prediction:.2f} (rounded to valid grade: {rounded})"
        )


if __name__ == "__main__":
    main()
