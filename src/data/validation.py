"""Валидация входящих данных с помощью Great Expectations (API 1.x).

Набор правил (suite) описывает ожидания к сырым данным, которые поступают из
источника: наличие колонок, отсутствие пропусков, типы, допустимые диапазоны
и множества значений. Скрипт завершается с кодом 1, если хотя бы одно правило
нарушено, поэтому его можно использовать как шаг CI и стадию DVC.

    python -m src.data.validation data/raw/UCI_Credit_Card.csv --report reports/data_validation.json
"""

import argparse
import json
import os
import sys

import pandas as pd

from src.config import (
    BILL_COLS,
    PAY_AMT_COLS,
    PAY_STATUS_COLS,
    RAW_FEATURES,
    TARGET,
    load_params,
    resolve,
)

os.environ.setdefault("GX_ANALYTICS_ENABLED", "False")

SUITE_NAME = "credit_data_suite"
INT_TYPES = ["int64", "int32", "int16", "int8", "Int64"]
NUM_TYPES = INT_TYPES + ["float64", "float32", "Float64"]


def expectation_specs(require_target: bool = True) -> list:
    """Правила в виде (имя expectation, kwargs) — не зависят от версии GE."""
    cols = RAW_FEATURES + ([TARGET] if require_target else [])
    specs = [("ExpectTableRowCountToBeBetween", {"min_value": 1})]
    for c in cols:
        specs.append(("ExpectColumnToExist", {"column": c}))
        specs.append(("ExpectColumnValuesToNotBeNull", {"column": c}))
        types = INT_TYPES if c in ["SEX", "EDUCATION", "MARRIAGE", "AGE", TARGET] + PAY_STATUS_COLS else NUM_TYPES
        specs.append(("ExpectColumnValuesToBeInTypeList", {"column": c, "type_list": types}))
    specs += [
        ("ExpectColumnValuesToBeBetween", {"column": "LIMIT_BAL", "min_value": 1_000, "max_value": 10_000_000}),
        ("ExpectColumnValuesToBeBetween", {"column": "AGE", "min_value": 18, "max_value": 100}),
        ("ExpectColumnValuesToBeInSet", {"column": "SEX", "value_set": [1, 2]}),
        # в сырых данных встречаются недокументированные коды 0, 5, 6 (их чистит make_dataset)
        ("ExpectColumnValuesToBeInSet", {"column": "EDUCATION", "value_set": [0, 1, 2, 3, 4, 5, 6]}),
        ("ExpectColumnValuesToBeInSet", {"column": "MARRIAGE", "value_set": [0, 1, 2, 3]}),
        # LIMIT_BAL в среднем ~170k NT$; сильный сдвиг среднего — признак сбоя источника
        ("ExpectColumnMeanToBeBetween", {"column": "LIMIT_BAL", "min_value": 50_000, "max_value": 500_000}),
    ]
    for c in PAY_STATUS_COLS:
        specs.append(("ExpectColumnValuesToBeBetween", {"column": c, "min_value": -2, "max_value": 9}))
    for c in BILL_COLS:
        specs.append(("ExpectColumnValuesToBeBetween", {"column": c, "min_value": -1_000_000, "max_value": 10_000_000}))
    for c in PAY_AMT_COLS:
        specs.append(("ExpectColumnValuesToBeBetween", {"column": c, "min_value": 0, "max_value": 10_000_000}))
    if require_target:
        specs.append(("ExpectColumnValuesToBeInSet", {"column": TARGET, "value_set": [0, 1]}))
    return specs


def build_expectation_suite(require_target: bool = True):
    import great_expectations as gx
    import great_expectations.expectations as gxe

    suite = gx.ExpectationSuite(name=SUITE_NAME)
    for name, kwargs in expectation_specs(require_target):
        suite.add_expectation(getattr(gxe, name)(**kwargs))
    return suite


def validate_dataframe(df: pd.DataFrame, require_target: bool = True) -> dict:
    """Проверяет DataFrame набором правил. Возвращает сводку: success, число правил, нарушения."""
    import great_expectations as gx

    context = gx.get_context(mode="ephemeral")
    suite = context.suites.add(build_expectation_suite(require_target))
    asset = context.data_sources.add_pandas("pandas_source").add_dataframe_asset(name="credit_data")
    batch = asset.add_batch_definition_whole_dataframe("full_batch").get_batch(batch_parameters={"dataframe": df})
    result = batch.validate(suite)

    failed = []
    for r in result.results:
        if not r.success:
            cfg = r.expectation_config
            exp_type = getattr(cfg, "type", None) or getattr(cfg, "expectation_type", "")
            failed.append({"expectation": exp_type, "column": cfg.kwargs.get("column")})
    return {
        "success": bool(result.success),
        "evaluated_expectations": len(result.results),
        "failed_expectations": failed,
    }


def save_suite(path) -> None:
    suite = build_expectation_suite()
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(suite.to_json_dict(), f, indent=2, ensure_ascii=False, default=str)


def main() -> None:
    params = load_params()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=params["data"]["raw_path"])
    parser.add_argument("--report", default=params["data"]["validation_report"])
    parser.add_argument("--no-target", action="store_true", help="данные без целевой колонки")
    parser.add_argument("--save-suite", default="data/expectations/credit_data_suite.json")
    args = parser.parse_args()

    df = pd.read_csv(resolve(args.path))
    summary = validate_dataframe(df, require_target=not args.no_target)
    summary["path"] = str(args.path)
    save_suite(args.save_suite)

    report = resolve(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    status = "OK" if summary["success"] else "FAILED"
    n_failed = len(summary["failed_expectations"])
    print(f"Валидация {status}: правил {summary['evaluated_expectations']}, нарушено {n_failed}")
    for f in summary["failed_expectations"]:
        print(f"  - {f['expectation']} [{f['column']}]")
    sys.exit(0 if summary["success"] else 1)


if __name__ == "__main__":
    main()
