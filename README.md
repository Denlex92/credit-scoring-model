# Credit Scoring: автоматизированный пайплайн PD-модели

Сквозной пайплайн разработки, тестирования, развёртывания и мониторинга модели вероятности дефолта (PD) на датасете [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (UCI, 30 000 клиентов, 23 признака, целевая переменная `default.payment.next.month`).

```
 UCI ──► download ──► validate (Great Expectations) ──► prepare (очистка, split) ──► train (GridSearch + MLflow)
                                                                                         │
                     monitor (PSI) ◄── FastAPI /predict (Docker) ◄── models/*.pkl ◄──────┘
           DVC: версии данных и моделей, dvc repro   |   GitHub Actions: flake8, black, pytest, GE, docker
```

## Структура

```
credit-scoring-model/
├── data/
│   ├── raw/                  # исходные данные (DVC)
│   ├── processed/            # train.csv / test.csv (DVC, стадия prepare)
│   └── expectations/         # набор правил Great Expectations (JSON)
├── models/                   # credit_default_model.pkl + model_meta.json (DVC, стадия train)
├── notebooks/01_eda.ipynb    # разведочный анализ
├── reports/                  # метрики, отчёты валидации и дрифта, графики
├── src/
│   ├── config.py             # константы и загрузка params.yaml
│   ├── data/                 # download.py, make_dataset.py, validation.py
│   ├── features/             # build_features.py — Feature Engineering (sklearn-трансформер)
│   ├── models/               # pipeline.py, train.py, metrics.py, predict.py, compare_experiments.py
│   ├── api/app.py            # FastAPI
│   └── monitoring/           # drift.py (PSI), monitor.py (имитация потока + дрифт)
├── tests/                    # pytest: данные, признаки, пайплайн, метрики, GE, API, PSI
├── scripts/                  # build_and_run.sh / .ps1, check_api.py, sample_request.json
├── .github/workflows/ci-cd.yml
├── Dockerfile, dvc.yaml, params.yaml, requirements*.txt, pyproject.toml, .flake8
```

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.data.download          # data/raw/UCI_Credit_Card.csv (или положите CSV с Kaggle)
dvc init                             # один раз, если репозиторий ещё не инициализирован
dvc add data/raw/UCI_Credit_Card.csv # версия сырых данных
dvc repro                            # validate -> prepare -> train
```

Все гиперпараметры, сетки поиска и пороги мониторинга задаются в `params.yaml`. После изменения параметров `dvc repro` перезапускает только затронутые стадии, `dvc metrics show` и `dvc params diff` показывают изменения.

## Этапы

### 1. Данные и валидация
* `src/data/download.py` скачивает архив с UCI и сохраняет его в CSV формата Kaggle.
* `src/data/make_dataset.py` (стадия **prepare**) удаляет ID и дубликаты, переводит недокументированные коды `EDUCATION` (0, 5, 6) и `MARRIAGE` (0) в категорию «другое», приводит типы и делает стратифицированный split 80/20.
* `src/data/validation.py` (стадия **validate**) содержит набор правил Great Expectations (≈ 100 expectations): наличие колонок, отсутствие null, типы, диапазоны (`AGE` 18–100, `PAY_*` −2…9, `PAY_AMT* ≥ 0`), множества значений (`SEX`, `EDUCATION`, `MARRIAGE`, target), допустимое среднее `LIMIT_BAL`. При нарушении скрипт завершается с кодом 1 и останавливает `dvc repro` и CI. Тесты `tests/test_validation.py` проверяют, что аномалии ловятся, а `test_raw_data_file_is_valid` падает, если аномалии есть в реальном файле.

### 2. Feature Engineering
`src/features/build_features.py` строит признаки внутри sklearn-трансформера, поэтому одинаковый код работает при обучении, в API и в мониторинге:
* агрегаты истории платежей: максимальная и средняя просрочка, число месяцев с просрочкой (≥ 1 и ≥ 2);
* суммы: средний счёт, средний платёж, загрузка лимита (последняя и средняя), доля оплаченных счетов, тренд задолженности, лимит на год возраста;
* биннинг возраста `AGE_BIN` (7 групп).

### 3. Модель
`src/models/pipeline.py`: `FeatureEngineer → ColumnTransformer(SimpleImputer + StandardScaler | SimpleImputer + OneHotEncoder) → классификатор`.
`src/models/train.py`: `GridSearchCV` / `RandomizedSearchCV` со стратифицированной 3-fold CV по ROC-AUC. Порог классификации подбирается по максимуму F1 на out-of-fold предсказаниях train, тест в подборе не участвует. Метрики на отложенном тесте: ROC-AUC, Gini, KS, PR-AUC, Precision, Recall, F1 (при пороге 0.5 и при подобранном), ROC-кривая и матрица ошибок.

### 4. MLflow
Шесть экспериментов из `params.yaml`: логистическая регрессия без FE, с FE и со взвешиванием классов, Random Forest, Gradient Boosting, HistGradientBoosting. В каждый run логируются параметры, лучшие гиперпараметры, CV- и тестовые метрики, графики (ROC, матрица ошибок, важность признаков), таблица CV-результатов и модель (`mlflow.sklearn.log_model` с сигнатурой). Лучшая по CV модель регистрируется в Model Registry как `CreditDefaultModel`.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db      # http://127.0.0.1:5000
python -m src.models.compare_experiments             # таблица reports/experiments_comparison.md + график
```

### 5. DVC
`dvc.yaml` содержит три стадии: `validate → prepare → train`. Версионируются сырые данные (`dvc add`), обработанные данные и модель (outs стадий), метрики (`reports/metrics.json`) и графики. Хранилище для `dvc push` можно добавить так:

```bash
dvc remote add -d storage ../dvc-storage    # или S3 / Google Drive
dvc push
```

### 6. Тесты и CI
```bash
pytest -v
flake8 src tests scripts && black --check src tests scripts
```
GitHub Actions (`.github/workflows/ci-cd.yml`) запускается на каждый push и PR. Джобы: **lint** (flake8, black) → **test** (pytest) и **data-validation** (загрузка данных с UCI, Great Expectations, тест аномалий; отчёт сохраняется как артефакт) → **docker** (обучение быстрой модели, сборка образа, запуск контейнера, smoke-тест `/predict`).

### 7. Docker и API
```bash
bash scripts/build_and_run.sh          # Windows: powershell -ExecutionPolicy Bypass -File scripts/build_and_run.ps1
```
Скрипт собирает образ, запускает контейнер на порту 8000 и проверяет `/health` и `/predict`. Swagger UI: http://localhost:8000/docs.

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @scripts/sample_request.json
# {"default_prediction":1,"default_probability":0.43,"threshold":0.23,"model_name":"..."}
```
Без Docker: `uvicorn src.api.app:app --reload`.

### 8. Мониторинг дрифта
```bash
python -m src.monitoring.monitor                 # 500 «новых заявок» из test -> API, PSI против train
python -m src.monitoring.monitor --shift 0.6     # имитация кризиса: лимиты ниже, просрочки чаще
python -m src.monitoring.monitor --local         # без API
```
PSI считается для распределения предсказанной вероятности и ключевых признаков (`LIMIT_BAL`, `AGE`, `PAY_0`, `BILL_AMT1`, `PAY_AMT1`). Пороги: < 0.1 — OK, 0.1–0.25 — WARNING, > 0.25 — ALERT. Отчёт сохраняется в `reports/drift_report.json` и `reports/figures/drift_report.png`.

## Результаты

### Данные и валидация
* Сырые данные: 30 000 клиентов. После удаления 35 полных дубликатов осталось 29 965, из них train — 23 972, test — 5 993. Доля дефолтов в обеих частях **22,1%** (стратифицированный split).
* Great Expectations: **98 правил, 0 нарушений** на реальных данных UCI. Отчёт — `reports/data_validation.json`, набор правил — `data/expectations/credit_data_suite.json`.

### Эксперименты MLflow

Выбор лучшей модели — по CV ROC-AUC на train, тестовая выборка в выборе не участвует. Precision, Recall и F1 приведены при пороге, подобранном по максимуму F1 на out-of-fold предсказаниях.

| Эксперимент | Модель | FE | CV ROC-AUC | Test ROC-AUC | Gini | KS | Precision | Recall | F1 | Время, с |
|---|---|---|---|---|---|---|---|---|---|---|
| **random_forest_fe** | Random Forest | ✓ | **0.7883 ± 0.006** | 0.7780 | 0.556 | 0.421 | 0.525 | 0.547 | 0.536 | 65 |
| gradient_boosting_fe | Gradient Boosting | ✓ | 0.7878 ± 0.007 | **0.7789** | 0.558 | 0.430 | 0.547 | 0.528 | **0.537** | 53 |
| hist_gradient_boosting_fe | HistGradientBoosting | ✓ | 0.7877 ± 0.006 | 0.7784 | 0.557 | 0.426 | 0.528 | 0.530 | 0.529 | **4** |
| logreg_fe_balanced | LogReg, class_weight | ✓ | 0.7689 ± 0.007 | 0.7569 | 0.514 | 0.405 | 0.512 | 0.540 | 0.526 | 1 |
| logreg_fe | LogReg | ✓ | 0.7684 ± 0.007 | 0.7548 | 0.510 | 0.403 | 0.506 | 0.544 | 0.524 | 3 |
| logreg_raw | LogReg | — | 0.7255 ± 0.006 | 0.7185 | 0.437 | 0.373 | 0.525 | 0.476 | 0.499 | 5 |

**Выводы:**
1. **Feature Engineering — главный источник качества.** Логистическая регрессия на тех же данных с агрегатами истории платежей и загрузкой лимита: ROC-AUC 0.726 → 0.768 (+0.04, Gini +0.07). Это больше, чем дал переход к ансамблям деревьев (+0.02).
2. **Три ансамбля деревьев статистически неразличимы:** разница CV ROC-AUC 0.0006 при стандартном отклонении по фолдам ~0.006. Для продакшена разумнее **HistGradientBoosting**: то же качество при обучении в 15 раз быстрее. В пайплайне по правилу выбрана Random Forest с максимальным CV ROC-AUC.
3. **Переобучения нет:** разрыв CV и test ROC-AUC ~0.01, что сопоставимо с разбросом по фолдам.
4. **Порог важен.** При пороге 0.5 лучшая модель находит только 35% дефолтов (Recall 0.353, Precision 0.674). Подобранный порог 0.29 повышает Recall до 0.547 при Precision 0.525, F1 растёт с 0.463 до 0.536. В кредитном скоринге пропуск дефолта обычно дороже отказа надёжному клиенту, поэтому порог стоит выбирать по экономике бизнеса.
5. `class_weight="balanced"` почти не меняет ROC-AUC: ранжирование то же, сдвигается только откалиброванность вероятностей (оптимальный порог 0.60 вместо 0.30).

Итоговое качество (Gini ≈ 0.56, KS ≈ 0.42) соответствует типичному уровню поведенческих скоринговых моделей.

### Тесты и CI
* `pytest`: **51 тест** (данные, Feature Engineering, пайплайн, метрики, Great Expectations, API, PSI), все проходят. `flake8` и `black --check` без замечаний.
* GitHub Actions: lint → test и data-validation → docker (сборка образа и smoke-тест `/predict`).

### Мониторинг дрифта
500 записей из test прогоняются через модель и сравниваются с train:

| Переменная | PSI, обычные данные | PSI, имитация кризиса (`--shift 0.6`) |
|---|---|---|
| default_probability | 0.017 — OK | **0.886 — ALERT** |
| LIMIT_BAL | 0.019 — OK | **0.578 — ALERT** |
| PAY_0 | 0.003 — OK | **0.833 — ALERT** |
| PAY_AMT1 | 0.014 — OK | 0.212 — WARNING |
| AGE | 0.027 — OK | 0.027 — OK |
| BILL_AMT1 | 0.027 — OK | 0.027 — OK |
| ROC-AUC на новых данных | 0.735 | 0.636 |

На стабильных данных все PSI < 0.03. При сдвиге (лимиты ниже, просрочки чаще) мониторинг сигнализирует ALERT по изменённым признакам и по распределению скоров, а не изменявшиеся `AGE` и `BILL_AMT1` остаются в норме. Падение ROC-AUC подтверждает, что при таком алерте модель нужно переобучать.

Графики: `reports/figures/roc_curve.png`, `confusion_matrix.png`, `experiments_comparison.png`, `drift_report.png`. Полная таблица экспериментов — `reports/experiments_comparison.md`.
