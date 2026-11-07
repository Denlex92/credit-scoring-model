import pandas as pd

from src.config import ID_COL, RAW_FEATURES, TARGET
from src.data.make_dataset import clean_data, load_raw, split_data


def test_clean_data_maps_undocumented_categories(raw_df):
    df = clean_data(raw_df)
    assert set(df["EDUCATION"].unique()) <= {1, 2, 3, 4}
    assert set(df["MARRIAGE"].unique()) <= {1, 2, 3}


def test_clean_data_drops_id_and_duplicates(raw_df):
    dup = pd.concat([raw_df, raw_df.iloc[:5].assign(ID=range(10_000, 10_005))], ignore_index=True)
    df = clean_data(dup)
    assert ID_COL not in df.columns
    assert len(df) == len(raw_df)


def test_clean_data_types(clean_df):
    assert clean_df["AGE"].dtype == "int64"
    assert clean_df["LIMIT_BAL"].dtype == "float64"
    assert clean_df[TARGET].dtype == "int64"
    assert list(clean_df.columns) == RAW_FEATURES + [TARGET]


def test_split_is_stratified(clean_df):
    train, test = split_data(clean_df, test_size=0.25, random_state=0)
    assert len(train) + len(test) == len(clean_df)
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.03


def test_load_raw_renames_uci_target(tmp_path, raw_df):
    path = tmp_path / "raw.csv"
    raw_df.rename(columns={TARGET: "default payment next month"}).to_csv(path, index=False)
    assert TARGET in load_raw(path).columns
