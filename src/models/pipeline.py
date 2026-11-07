"""Сборка sklearn Pipeline: Feature Engineering -> предобработка -> классификатор."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_RAW, RAW_FEATURES
from src.features.build_features import (
    ENGINEERED_CATEGORICAL,
    ENGINEERED_NUMERIC,
    FeatureEngineer,
)

MODELS = {
    "logreg": lambda rs: LogisticRegression(max_iter=2000, random_state=rs),
    "random_forest": lambda rs: RandomForestClassifier(n_jobs=1, random_state=rs),
    "gradient_boosting": lambda rs: GradientBoostingClassifier(random_state=rs),
    "hist_gradient_boosting": lambda rs: HistGradientBoostingClassifier(random_state=rs),
}


def feature_lists(feature_engineering: bool):
    numeric = [c for c in RAW_FEATURES if c not in CATEGORICAL_RAW]
    categorical = list(CATEGORICAL_RAW)
    if feature_engineering:
        numeric += ENGINEERED_NUMERIC
        categorical += ENGINEERED_CATEGORICAL
    return numeric, categorical


def create_preprocessor(numeric, categorical) -> ColumnTransformer:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, categorical)])


def create_pipeline(model: str = "logreg", feature_engineering: bool = True, random_state: int = 42) -> Pipeline:
    if model not in MODELS:
        raise ValueError(f"Неизвестная модель '{model}', доступны: {list(MODELS)}")
    numeric, categorical = feature_lists(feature_engineering)
    return Pipeline(
        [
            ("features", FeatureEngineer(enabled=feature_engineering)),
            ("preprocessor", create_preprocessor(numeric, categorical)),
            ("classifier", MODELS[model](random_state)),
        ]
    )
