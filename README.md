
# FastAPI Linear Regression API

Проект демонстрирует простой API на FastAPI, который загружает заранее обученную модель
линейной регрессии (joblib/pkl) и возвращает предсказания.

## Структура проекта

```
ml_api_project/
├─ app/
│  ├─ server.py        # FastAPI приложение
│  ├─ ml_tools.py      # Утилиты работы с моделью
│  └─ models/
│     ├─ linear_regression_sklearn1_3_joblib.pkl   # Сохранённая модель
│     └─ model_info.json
├─ client.py           # Простое консольное клиент-приложение
├─ tests/
│  └─ test_model.py    # Набор простых тестов (pytest)
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

3. Файл `.env` уже добавлен в корень проекта. 
   - `MODEL_FILENAME` — имя файла модели в `app/models/`
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

- `GET /ping` — возвращает {{"status":"ok"}}
- `POST /api/v1/prediction` — принимает JSON {{"features":[...]}} и возвращает {{"prediction": <value>}}
- `GET /api/v1/prediction?features=1,2,3` — принимает фичи как query string (через запятую)
- `GET /api/v1/model_info` — возвращает коэффициенты модели и R2 (предварительно вычисленный)

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

- Модель и `model_info.json` находятся в `app/models/`.
- Код документирован комментариями.
- API использует версионирование: `/api/v1/...`.

