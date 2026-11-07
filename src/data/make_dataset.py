"""Стадия prepare: загрузка сырых данных, первичная очистка и разбиение на train/test.

python -m src.data.make_dataset
"""

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ID_COL, RAW_FEATURES, TARGET, load_params, resolve


def load_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": TARGET})
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Первичная очистка.

    * удаляем ID и полные дубликаты;
    * EDUCATION: значения 0, 5, 6 не описаны в документации -> категория 4 ("others");
    * MARRIAGE: значение 0 не описано -> категория 3 ("others");
    * приводим типы: категориальные и статусы платежей -> int, суммы -> float.
    """
    df = df.drop(columns=[ID_COL], errors="ignore").drop_duplicates().reset_index(drop=True)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    int_cols = ["SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    df[int_cols] = df[int_cols].astype("int64")
    float_cols = [c for c in RAW_FEATURES if c not in int_cols]
    df[float_cols] = df[float_cols].astype("float64")
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].astype("int64")
    return df


def split_data(df: pd.DataFrame, test_size: float, random_state: int):
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[TARGET])


def main() -> None:
    params = load_params()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default=params["data"]["raw_path"])
    parser.add_argument("output_dir", nargs="?", default=params["data"]["processed_dir"])
    args = parser.parse_args()

    df = clean_data(load_raw(resolve(args.input)))
    train, test = split_data(df, params["split"]["test_size"], params["split"]["random_state"])
    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    print(
        f"clean: {len(df)} строк | train: {len(train)} (доля дефолтов {train[TARGET].mean():.3f}) | "
        f"test: {len(test)} (доля дефолтов {test[TARGET].mean():.3f})"
    )


if __name__ == "__main__":
    main()
