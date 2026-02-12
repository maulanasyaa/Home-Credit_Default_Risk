import os
import sys
from pathlib import Path

import dotenv
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna_integration import LightGBMPruningCallback
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import create_engine

sys.path.append(str(Path(__file__).resolve().parent.parent))

from notebooks.config import encoded_features, normalized_features, to_remove_features

DATA_DB_URL = "postgresql://user:password@localhost:5432/home_credit_db"
OPTUNA_DB_URL = "postgresql://user:password@localhost:5432/optuna_db"


def get_or_create_study(study_name: str, DB_URL: str):
    sampler = TPESampler(multivariate=True)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=study_name,
        storage=DB_URL,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    return study


def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }

    pruning_callback = LightGBMPruningCallback(trial, "auc")

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            pruning_callback,
        ],
    )

    return model.best_score["valid_0"]["auc"]


def create_feature(df):
    # external source mean
    df["ext_source_avg"] = np.mean(
        df[["ext_source_1", "ext_source_2", "ext_source_3"]], axis=1
    )

    # debt to income ratio
    df["debt_to_income_ratio"] = df["amt_credit"] / df["amt_income_total"]

    # payment to income ratio
    df["payment_to_income_ratio"] = df["amt_annuity"] / df["amt_income_total"]

    # credit to goods ratio
    df["credit_to_goods_ratio"] = df["amt_credit"] / df["amt_goods_price"]

    # days employed percentage
    df["days_employed_percentage"] = df["days_employed"] / df["days_birth"]

    # income per person
    df["income_per_person"] = df["amt_income_total"] / df["cnt_fam_members"]

    return df


def data_prep(df):
    X = df.drop(columns=["sk_id_curr", "target"])
    y = df["target"]

    # select feature to scaled
    num_features = X.select_dtypes(include=np.number).columns
    cat_features = X.select_dtypes(include=np.object_).columns

    col_to_drop = [
        col for col in (normalized_features + encoded_features) if col in num_features
    ]

    col_to_scaled = df[num_features].drop(columns=col_to_drop).columns

    # preprocessing pipeline

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessing_pipeline = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, col_to_scaled),
            ("cat", categorical_transformer, cat_features),
        ]
    )

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
    X_test_transformed = preprocessing_pipeline.transform(X_test)

    return X_train_transformed, X_test_transformed, y_train, y_test


def main():
    # data ingestion
    engine = create_engine(DATA_DB_URL)
    query_app = """
            select * from application_train_clean
        """

    df = pd.read_sql(query_app, engine)
    df = df.drop(columns=to_remove_features)

    # feature engineering
    df = create_feature(df)

    # process data
    X_train, X_test, y_train, y_test = data_prep(df)

    # optuna study
    study = get_or_create_study("lgbm_home_credit_v2", DB_URL=OPTUNA_DB_URL)

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=100
    )

    print(f"Best AUC: {study.best_value}")


if __name__ == "__main__":
    main()

# to run optuna dashboard
# uv run optuna-dashboard postgresql+pg8000://user:password@localhost:5432/optuna_db
