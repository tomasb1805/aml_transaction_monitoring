# Python libraries
from typing import NamedTuple

import numpy as np
import pandas as pd

# Machine learning libraries
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# project modules
from src.config import DUCKDB_PATH, SQL_DIR, MODEL_TABLE_PATH
from src.database import connect
from scripts.customer_risk_segmentation import assign_customer_segments


# configuration
TARGET = 'is_laundering'

# chronological split (fractions of transactions ordered by time): train / validation / test.
# AML models score the future, so a random split would leak tomorrow's patterns into training.
SPLIT_QUANTILES = (0.60, 0.80)

# share of legitimate training transactions kept (all laundering cases are kept).
# 1.0 = no undersampling: the imbalance ablation (imbalance_ablation.py) found that the
# full training window without re-weighting gives the best validation Average Precision.
# Validation and test windows are never sampled.
NEG_SAMPLE_RATE = 1.0
RANDOM_STATE = 13

BEHAVIOURAL_COLS = [
    'orig_avg_daily_inflow', 'orig_inflow_stability_ratio', 'orig_spend_to_income_ratio',
    'orig_days_net_negative', 'orig_cash_withdrawal_ratio', 'orig_crypto_outflow_ratio',
    'benef_avg_daily_inflow', 'benef_spend_to_income_ratio', 'benef_days_net_negative',
]

NUMERICAL_FEATURES = [
    # transaction
    'amount_paid_usd', 'is_self_transfer', 'is_cross_bank', 'is_cross_currency', 'hour_of_day',
    # KYC
    'orig_entity_n_accounts', 'benef_entity_n_accounts', 'orig_prior_days_observed',
    # behavioural profile
    *BEHAVIOURAL_COLS,
    # originator velocity / network
    'orig_out_count_24h', 'orig_out_amount_usd_24h', 'orig_out_distinct_cp_24h',
    'orig_in_count_24h', 'orig_in_amount_usd_24h', 'orig_in_distinct_cp_24h',
    'orig_out_distinct_cp_cum', 'orig_in_distinct_cp_cum', 'orig_legs_cum',
    'orig_distinct_formats_cum', 'orig_mins_since_prev_leg', 'orig_out_count_prior',
    'orig_amount_zscore',
    # beneficiary velocity / network
    'benef_in_count_24h', 'benef_in_amount_usd_24h', 'benef_in_distinct_cp_24h',
    'benef_out_count_24h', 'benef_out_distinct_cp_24h', 'benef_in_distinct_cp_cum',
    'benef_out_distinct_cp_cum', 'benef_legs_cum',
    # relationship
    'pair_count_24h', 'pair_count_prior',
    # rules engine output
    'rule_hit_count',
    # interaction features
    'pass_through_ratio', 'amount_to_avg_inflow', 'orig_payee_concentration_24h',
    'orig_flow_balance_24h', 'is_new_payee',
    # missing-history indicators
    *[f'{col}_missing' for col in BEHAVIOURAL_COLS],
]

CATEGORICAL_FEATURES = [
    'payment_format', 'payment_currency', 'receiving_currency',
    'orig_entity_type', 'benef_entity_type', 'customer_segment',
]


class ModelDataset(NamedTuple):
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    df_val: pd.DataFrame
    df_test: pd.DataFrame


def export_model_table(db_path=DUCKDB_PATH, out_path=MODEL_TABLE_PATH) -> None:
    """
    Runs the modelling-table join in DuckDB and writes the result to parquet.
    """
    query = (SQL_DIR / 'db_join_modeling_table.sql').read_text()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with connect(db_path, read_only=True) as con:
        con.execute(f"COPY ({query}) TO '{out_path}' (FORMAT parquet, COMPRESSION zstd)")
        n_rows = con.execute(f"SELECT COUNT(*) FROM '{out_path}'").fetchone()[0]

    print(f"Modelling table exported: {n_rows:,} rows → {out_path.name}")


def load_model_table(table_path=MODEL_TABLE_PATH) -> pd.DataFrame:
    """
    Load the modelling table to be preprocessed.
    """
    return pd.read_parquet(table_path)


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accounts with no prior history (first day seen) have no behavioural profile.
    Add the "_missing" flag so the model understands the absence is meaningful.
    """
    df = df.copy()
    for col in BEHAVIOURAL_COLS:
        df[f'{col}_missing'] = df[col].isna().astype('int8')

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer interaction features capturing laundering mechanics.
    """
    df = df.copy()
    # value sent on relative to value received in the last 24h (layering / pass-through)
    df['pass_through_ratio']           = df['amount_paid_usd'] / (df['orig_in_amount_usd_24h'] + 1)
    # payment size relative to the originator's usual daily inflow
    df['amount_to_avg_inflow']         = df['amount_paid_usd'] / (df['orig_avg_daily_inflow'] + 1)
    # repeat payments per distinct beneficiary in 24h (1 = every payment to a new party)
    df['orig_payee_concentration_24h'] = df['orig_out_count_24h'] / df['orig_out_distinct_cp_24h'].clip(lower=1)
    # net direction of the originator's 24h flows, -1 (all inflow) to +1 (all outflow)
    df['orig_flow_balance_24h']        = (
        (df['orig_out_amount_usd_24h'] - df['orig_in_amount_usd_24h']) /
        (df['orig_out_amount_usd_24h'] + df['orig_in_amount_usd_24h'] + 1)
    )
    df['is_new_payee']                 = (df['pair_count_prior'] == 0).astype('int8')

    return df


def temporal_split(df: pd.DataFrame) -> tuple:
    """
    Boolean masks for the chronological train / validation / test windows.
    """
    cut_val, cut_test = df['ts'].quantile(list(SPLIT_QUANTILES))

    train = df['ts'] < cut_val
    val   = (df['ts'] >= cut_val) & (df['ts'] < cut_test)
    test  = df['ts'] >= cut_test

    for name, mask in [('train', train), ('validation', val), ('test', test)]:
        window = df.loc[mask, 'ts']
        print(f"{name:<11} {window.min():%Y-%m-%d %H:%M} → {window.max():%Y-%m-%d %H:%M}  "
              f"{mask.sum():>10,} txns  {df.loc[mask, TARGET].sum():>6,} laundering "
              f"({df.loc[mask, TARGET].mean():.3%})")

    return train, val, test


def undersample_negatives(X: pd.DataFrame, y: pd.Series,
                          rate: float = NEG_SAMPLE_RATE) -> tuple:
    """
    Keeps every laundering case and a random `rate` share of legitimate transactions.
    """
    if rate >= 1.0:
        return X, y

    rng = np.random.default_rng(RANDOM_STATE)
    keep = (y == 1).to_numpy() | (rng.random(len(y)) < rate)
    print(f"Training sample: {keep.sum():,} of {len(y):,} txns "
          f"(all {int(y.sum()):,} laundering + {rate:.0%} of legitimate)")

    return X[keep], y[keep]


def preprocess_df(df: pd.DataFrame, neg_sample_rate: float = NEG_SAMPLE_RATE) -> ModelDataset:
    """
    Preprocess the DataFrame for model instantiation.
    """
    train, val, test = temporal_split(df)

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, y_train = undersample_negatives(X[train], y[train], rate=neg_sample_rate)

    preprocessing = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), NUMERICAL_FEATURES),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), CATEGORICAL_FEATURES)
        ]
    )

    return ModelDataset(
        X_train=X_train, X_val=X[val], X_test=X[test],
        y_train=y_train, y_val=y[val], y_test=y[test],
        preprocessor=preprocessing,
        df_val=df[val], df_test=df[test],
    )


def build_dataset(export: bool = True, neg_sample_rate: float = NEG_SAMPLE_RATE) -> ModelDataset:
    """
    Export (optionally), load and prepare the modelling table end to end.
    """
    if export:
        export_model_table()

    df = load_model_table()
    df = assign_customer_segments(df)
    df = add_missing_flags(df)
    df = feature_engineering(df)

    return preprocess_df(df, neg_sample_rate=neg_sample_rate)


if __name__ == "__main__":
    data = build_dataset()
    print(data.X_train.shape, data.X_val.shape, data.X_test.shape)
