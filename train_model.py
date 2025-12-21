"""
Скрипт обучения и сохранения модели линейной регрессии.
Модель обучается в текущей версии scikit-learn.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import sklearn


def main():
    # Корень проекта
    project_root = Path(__file__).resolve().parent
    models_dir = project_root / "app" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # === Генерация данных (6 признаков) ===
    X, y, true_coef = make_regression(
    n_samples=500,
    n_features=6,
    noise=15.0,
    random_state=42,
    coef=True
)


    # === Обучение модели ===
    model = LinearRegression()
    model.fit(X, y)

    # === Кросс-валидация (R2) ===
    r2_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    # === Имя файла модели с версией sklearn ===
    sklearn_version = sklearn.__version__
    model_filename = f"linear_regression_6_features_sklearn_{sklearn_version}.joblib"

    # === Сохранение модели ===
    joblib.dump(model, models_dir / model_filename)

    # === Сохранение информации о модели ===
    model_info = {
        "model_type": "LinearRegression",
        "n_features": 6,
        "sklearn_version": sklearn_version,
        "r2_cv_mean": float(r2_scores.mean()),
        "r2_cv_scores": r2_scores.tolist(),
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_)
    }

    with open(models_dir / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)

    print("Модель успешно обучена и сохранена")
    print(f"Версия scikit-learn: {sklearn_version}")
    print(f"Файл модели: {model_filename}")
    print(f"Средний R2 (CV): {model_info['r2_cv_mean']:.4f}")


if __name__ == "__main__":
    main()
