
# FastAPI Logistic Regression API (Heart Disease)

Проект демонстрирует простой API на FastAPI для бинарной классификации наличия заболеваний сердца
с помощью модели **логистической регрессии**.

## Структура проекта

```
ml_api_project/
├─ app/
│  ├─ server.py        # FastAPI приложение
│  ├─ ml_tools.py      # Утилиты работы с моделью
│  └─ models/
│     ├─ pipeline.pkl          # Сохранённый sklearn Pipeline (scaler + logistic regression)
│     ├─ label_encoders.pkl    # LabelEncoder для категориальных признаков
│     └─ config.json           # Метрики и конфигурация (feature_names и т.д.)
├─ client.py           # Простое консольное клиент-приложение
├─ tests/
│  └─ test_api.py      # Набор простых тестов (pytest)
├─ requirements.txt
├─ .env
└─ README.md
```

## Как настроить (локально)

1. Создайте виртуальное окружение:
   ```bash
   python -m venv venv
   venv\Scripts\activate    
   ```

2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## Как обучить модель

Скрипт `train_model.py`:

- скачивает датасет `cleve.mod` (если отсутствует) в `data/cleve.mod`
- выполняет препроцессинг (как в ноутбуке)
- обучает `LogisticRegression`
- сохраняет артефакты в `app/models/`

Запуск:
```bash
python train_model.py
```

3. Файл `.env` уже добавлен в корень проекта. 
   - `API_PREFIX` — префикс API (по умолчанию `/api/v1`)
   - `HOST`, `PORT` — адрес и порт сервера

## Как запустить сервер

```bash
uvicorn app.server:app --reload --host 0.0.0.0 --port 8000
```

Или используя переменные из `.env`:
```bash
uvicorn app.server:app --reload --host ${HOST} --port ${PORT}
```

## Эндпоинты

- `GET /ping` — проверка доступности сервера

- `POST /api/v1/prediction` — предсказание по «человеческим» полям пациента (рекомендуемый способ)

  Пример запроса:
  ```json
  {
    "age": 63,
    "sex": "male",
    "cp": "angina",
    "trestbps": 145,
    "chol": 233,
    "fbs": "true",
    "restecg": "hyp",
    "thalach": 150,
    "exang": "fal",
    "oldpeak": 2.3,
    "slope": "down",
    "ca": 0,
    "thal": "fix"
  }
  ```

  Пример ответа:
  ```json
  {
    "class": 0,
    "proba": 0.123
  }
  ```

- `GET /api/v1/prediction?features=...` — предсказание, если вы передаёте **уже подготовленные 13 чисел**
  в порядке `config.json -> feature_names`

- `GET /api/v1/model_info` — метрики и конфигурация модели (читается из `app/models/config.json`)

## Тесты

Запустить тесты:
```bash
pytest -q
```

## Клиент

Простой консольный клиент:
```bash
python client.py
```

## Примечания

- Артефакты модели (`pipeline.pkl`, `label_encoders.pkl`, `config.json`) находятся в `app/models/`.
- Код документирован комментариями.
- API использует версионирование: `/api/v1/...`.

