"""Скачивание датасета Default of Credit Card Clients с UCI и сохранение в CSV.

Результат совпадает по формату с версией UCI_Credit_Card.csv с Kaggle
(колонка ID, 23 признака, целевая колонка default.payment.next.month).

    python -m src.data.download
"""

import argparse
import io
import urllib.request
import zipfile

import pandas as pd

from src.config import TARGET, load_params, resolve

UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"


def download_uci(url: str = UCI_URL) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=120) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    xls_name = next(n for n in archive.namelist() if n.lower().endswith((".xls", ".xlsx")))
    df = pd.read_excel(io.BytesIO(archive.read(xls_name)), header=1)
    return df.rename(columns={"default payment next month": TARGET})


def main() -> None:
    params = load_params()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=params["data"]["raw_path"])
    parser.add_argument("--force", action="store_true", help="перескачать, даже если файл есть")
    args = parser.parse_args()

    out = resolve(args.out)
    if out.exists() and not args.force:
        print(f"{out} уже существует, пропускаю загрузку")
        return
    df = download_uci()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Сохранено {df.shape[0]} строк, {df.shape[1]} колонок -> {out}")


if __name__ == "__main__":
    main()
