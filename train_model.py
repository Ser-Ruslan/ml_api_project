"""Скрипт обучения и сохранения модели логистической регрессии.

Пайплайн и препроцессинг повторяют шаги из ноутбука
EDA_Логистическая_регрессия_ОБНОВЛЕНО (1).ipynb.
"""

import json
from pathlib import Path
import pickle

import pandas as pd
import requests
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


DATA_URL = (
    "https://raw.githubusercontent.com/jamesrobertlloyd/dataset-space/"
    "master/data/class/raw/uci/heart/cleve.mod.txt"
)


def _download_if_missing(target_path: Path) -> None:
    if target_path.exists():
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(DATA_URL, timeout=60)
    resp.raise_for_status()
    target_path.write_bytes(resp.content)


def _remove_outliers_iqr(df: pd.DataFrame, columns: list[str], iqr_multiplier: float = 1.5) -> pd.DataFrame:
    data = df
    for col in columns:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        data = data[(data[col] >= lower) & (data[col] <= upper)]
    return data


def main() -> None:
    project_root = Path(__file__).resolve().parent
    models_dir = project_root / "app" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    data_dir = project_root / "data"
    data_path = data_dir / "cleve.mod"
    _download_if_missing(data_path)

    df = pd.read_csv(data_path, delim_whitespace=True, skiprows=20, header=None)
    column_names = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "diagnosis",
        "class",
    ]
    df.columns = column_names

    df_processed = df.copy()
    df_processed["target"] = (df_processed["class"] != "H").astype(int)

    # ca: '?' -> NaN -> median
    df_processed["ca"] = pd.to_numeric(df_processed["ca"], errors="coerce")
    if df_processed["ca"].isna().any():
        df_processed["ca"] = df_processed["ca"].fillna(df_processed["ca"].median())

    # Label Encoding for categorical columns
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
    label_encoders: dict[str, LabelEncoder] = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col + "_encoded"] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le

    # Remove duplicates
    df_processed = df_processed.drop_duplicates()

    # Remove outliers (IQR) as in notebook
    numeric_cols_to_check = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    df_processed = _remove_outliers_iqr(df_processed, numeric_cols_to_check, iqr_multiplier=1.5)

    feature_cols = numeric_cols_to_check + [col + "_encoded" for col in categorical_cols]
    X = df_processed[feature_cols].copy()
    y = df_processed["target"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs"),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Cross-validation (on train) as in notebook
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_acc = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="accuracy")
    cv_scores_f1 = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="f1")
    cv_scores_roc_auc = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="roc_auc")

    # Save artifacts
    pipeline_path = models_dir / "pipeline.pkl"
    encoders_path = models_dir / "label_encoders.pkl"
    config_path = models_dir / "config.json"

    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)

    with open(encoders_path, "wb") as f:
        pickle.dump(label_encoders, f)

    config = {
        "model_type": "LogisticRegression",
        "sklearn_version": sklearn.__version__,
        "feature_names": feature_cols,
        "feature_names_numeric": numeric_cols_to_check,
        "feature_names_categorical": categorical_cols,
        "metrics_test": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
        },
        "cv": {
            "accuracy_scores": cv_scores_acc.tolist(),
            "accuracy_mean": float(cv_scores_acc.mean()),
            "accuracy_std": float(cv_scores_acc.std()),
            "f1_scores": cv_scores_f1.tolist(),
            "f1_mean": float(cv_scores_f1.mean()),
            "f1_std": float(cv_scores_f1.std()),
            "roc_auc_scores": cv_scores_roc_auc.tolist(),
            "roc_auc_mean": float(cv_scores_roc_auc.mean()),
            "roc_auc_std": float(cv_scores_roc_auc.std()),
        },
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("Модель логистической регрессии успешно обучена и сохранена")
    print(f"Файл pipeline: {pipeline_path}")
    print(f"Файл encoders: {encoders_path}")
    print(f"Файл config:   {config_path}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test ROC-AUC:  {roc_auc:.4f}")


if __name__ == "__main__":
    main()
