"""
ETL pipeline for the IBM AML synthetic dataset.

    extract → standardise → validate → transform → load (DuckDB `raw` schema)

Three sources are processed:
    Trans.csv     -  the transaction ledger (one row per payment, labelled `is_laundering`)
    accounts.csv  -  account/KYC reference data (bank, entity, entity type)
    Patterns.txt  -  laundering attempts grouped by typology (FAN-OUT, CYCLE, ...),
                     used downstream to measure detection rates per typology
"""
import csv
import re

import numpy as np
import pandas as pd

from src.config import DUCKDB_PATH, SQL_DIR, raw_file
from src.database import connect


# configuration
# the raw header repeats 'Account' for both sides of the payment, so names are set explicitly
TRANSACTION_COLUMNS = [
    'timestamp', 'from_bank', 'from_account', 'to_bank', 'to_account',
    'amount_received', 'receiving_currency', 'amount_paid', 'payment_currency',
    'payment_format', 'is_laundering'
]

ACCOUNT_RENAME_MAP = {
    'Bank Name':      'bank_name',
    'Bank ID':        'bank_id',
    'Account Number': 'account_number',
    'Entity ID':      'entity_id',
    'Entity Name':    'entity_name',
}

TIMESTAMP_FORMAT = '%Y/%m/%d %H:%M'
PAYMENT_FORMATS  = {'ACH', 'Bitcoin', 'Cash', 'Cheque', 'Credit Card', 'Reinvestment', 'Wire'}
BASE_CURRENCY    = 'US Dollar'

# a day whose volume is below 1% of the median daily volume is a simulator artefact:
# normal activity stops and only the tails of in-flight laundering attempts remain
TAIL_VOLUME_RATIO = 0.01

TRANSACTION_COL_ORDER = [
    'transaction_id', 'ts', 'txn_date',
    'from_bank', 'from_account', 'from_account_id',
    'to_bank', 'to_account', 'to_account_id',
    'amount_received', 'receiving_currency', 'amount_paid', 'payment_currency',
    'payment_format', 'amount_paid_usd', 'amount_received_usd',
    'is_self_transfer', 'is_cross_bank', 'is_cross_currency',
    'hour_of_day', 'day_of_week', 'is_simulation_tail', 'is_laundering'
]


# functions definitions
def normalise_bank_id(bank: pd.Series) -> pd.Series:
    """
    Bank IDs carry leading zeros in the ledger ('010') but not in the
    accounts file ('10'): strip them so both sources share one key.
    """
    stripped = bank.str.lstrip('0')
    return stripped.where(stripped != '', '0')


def make_account_id(bank: pd.Series, account: pd.Series) -> pd.Series:
    """
    Account numbers are only unique within a bank: the composite key is bank_account.
    """
    return normalise_bank_id(bank) + '_' + account


def extract_transactions(path) -> pd.DataFrame:
    """
    Load the raw transactions CSV. Every column is read as text so that
    bank IDs and account numbers keep their exact source representation.
    """
    return pd.read_csv(path, names=TRANSACTION_COLUMNS, header=0, dtype=str)


def extract_accounts(path) -> pd.DataFrame:
    """
    Load the raw accounts (KYC reference) CSV.
    """
    return pd.read_csv(path, dtype=str)


def extract_patterns(path) -> pd.DataFrame:
    """
    Parse the laundering-patterns file: blocks of transactions wrapped in
    'BEGIN LAUNDERING ATTEMPT - <TYPOLOGY>' / 'END LAUNDERING ATTEMPT' lines.
    """
    rows, typology, attempt_id = [], None, 0

    with open(path, newline='') as f:
        for line in f:
            line = line.strip()
            if line.startswith('BEGIN LAUNDERING ATTEMPT'):
                attempt_id += 1
                typology = re.match(r'BEGIN LAUNDERING ATTEMPT - ([A-Z\-]+)', line).group(1)
            elif line.startswith('END LAUNDERING ATTEMPT'):
                typology = None
            elif typology and line:
                rows.append([attempt_id, typology] + next(csv.reader([line])))

    return pd.DataFrame(rows, columns=['attempt_id', 'typology'] + TRANSACTION_COLUMNS)


def standardise_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse timestamps and amounts, cast the label and build composite account keys.
    """
    df = df.copy()
    df['ts']              = pd.to_datetime(df['timestamp'], format=TIMESTAMP_FORMAT)
    df['amount_received'] = df['amount_received'].astype('float64')
    df['amount_paid']     = df['amount_paid'].astype('float64')
    df['is_laundering']   = df['is_laundering'].astype('int8')
    df['from_account_id'] = make_account_id(df['from_bank'], df['from_account'])
    df['to_account_id']   = make_account_id(df['to_bank'], df['to_account'])

    return df.drop(columns='timestamp')


def standardise_accounts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename to snake_case, build the composite account key and extract the
    KYC entity type ('Corporation #33520' → 'corporation').
    """
    df = df.rename(columns=ACCOUNT_RENAME_MAP)
    df['account_id']  = make_account_id(df['bank_id'], df['account_number'])
    df['entity_type'] = (
        df['entity_name'].str.extract(r'^(.*?) #', expand=False)
        .str.lower().str.replace(' ', '_')
        .fillna('unknown')
    )

    return df


def validate_transactions(df: pd.DataFrame) -> None:
    """
    Execute data quality checks on the transaction ledger.
    """
    # checks for Null values
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        raise ValueError(f"Null values detected:\n{null_cols.to_string()}")

    # checks the label is binary
    unexpected = set(df['is_laundering'].unique()) - {0, 1}
    if unexpected:
        raise ValueError(f"'is_laundering' contains unexpected values: {unexpected}")

    # checks payment formats against the known channel list
    unexpected = set(df['payment_format'].unique()) - PAYMENT_FORMATS
    if unexpected:
        raise ValueError(f"Unknown payment formats: {unexpected}")

    # checks for negative amounts
    if (df[['amount_paid', 'amount_received']] < 0).any().any():
        raise ValueError("Negative transaction amounts detected")

    # repeated identical rows are kept: same-minute repeat payments are plausible activity
    n_dupes = df.duplicated().sum()
    print(f"Validation passed — {len(df):,} transactions, {n_dupes:,} exact duplicate rows kept")


def deduplicate_accounts(df: pd.DataFrame) -> pd.DataFrame:
    """
    A handful of bank IDs collide once leading zeros are stripped:
    keep the first record per composite key so joins stay one-to-one.
    """
    n_dupes = df['account_id'].duplicated().sum()
    if n_dupes:
        print(f"Dropping {n_dupes} duplicate account keys")

    return df.drop_duplicates(subset='account_id', keep='first')


def derive_fx_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives a USD conversion rate per currency from the data itself:
    cross-currency payments settled in USD reveal the simulator's exchange rate
    (amount_received / amount_paid). The median makes it robust to rounding noise.
    """
    to_usd = df[(df['receiving_currency'] == BASE_CURRENCY) &
                (df['payment_currency'] != BASE_CURRENCY) &
                (df['amount_paid'] > 0)]
    from_usd = df[(df['payment_currency'] == BASE_CURRENCY) &
                  (df['receiving_currency'] != BASE_CURRENCY) &
                  (df['amount_received'] > 0)]

    direct  = (to_usd['amount_received'] / to_usd['amount_paid']).groupby(to_usd['payment_currency'])
    inverse = (from_usd['amount_paid'] / from_usd['amount_received']).groupby(from_usd['receiving_currency'])

    rates = pd.concat([direct.median(), inverse.median()], axis=1, keys=['direct', 'inverse'])
    counts = pd.concat([direct.size(), inverse.size()], axis=1).sum(axis=1)

    fx = pd.DataFrame({
        'usd_rate':       rates['direct'].fillna(rates['inverse']),
        'n_observations': counts.astype('int64'),
    })
    fx.loc[BASE_CURRENCY] = [1.0, (df['payment_currency'] == BASE_CURRENCY).sum()]

    currencies = set(df['payment_currency']) | set(df['receiving_currency'])
    missing = currencies - set(fx.index)
    if missing:
        raise ValueError(f"No USD exchange rate could be derived for: {missing}")

    return fx.rename_axis('currency').reset_index()


def engineer_features(df: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """
    Derive row-level attributes: USD-normalised amounts, payment-route flags
    and calendar features.
    """
    df = df.copy()
    usd_rate = fx.set_index('currency')['usd_rate']

    df['amount_paid_usd']     = df['amount_paid'] * df['payment_currency'].map(usd_rate)
    df['amount_received_usd'] = df['amount_received'] * df['receiving_currency'].map(usd_rate)
    df['is_self_transfer']    = (df['from_account_id'] == df['to_account_id']).astype('int8')
    df['is_cross_bank']       = (normalise_bank_id(df['from_bank']) != normalise_bank_id(df['to_bank'])).astype('int8')
    df['is_cross_currency']   = (df['payment_currency'] != df['receiving_currency']).astype('int8')
    df['txn_date']            = df['ts'].dt.date
    df['hour_of_day']         = df['ts'].dt.hour.astype('int8')
    df['day_of_week']         = df['ts'].dt.dayofweek.astype('int8')

    return df


def flag_simulation_tail(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags the trailing days where the simulator only emits the remainder of
    laundering attempts (daily volume collapses, laundering share jumps to ~60%).
    These rows stay in the raw layer but are excluded from modelling.
    """
    df = df.copy()
    daily = df.groupby('txn_date').size()
    tail_days = daily[daily < TAIL_VOLUME_RATIO * daily.median()].index

    df['is_simulation_tail'] = df['txn_date'].isin(tail_days).astype('int8')
    if len(tail_days):
        print(f"Simulation tail: {len(tail_days)} days from {min(tail_days)} "
              f"({df['is_simulation_tail'].sum():,} rows) flagged")

    return df


def assign_transaction_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order the ledger chronologically and assign a surrogate primary key.
    """
    df = df.sort_values('ts', kind='stable').reset_index(drop=True)
    df.insert(0, 'transaction_id', np.arange(len(df), dtype='int64'))

    return df


def build_entity_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Number of accounts held by the same legal entity (multi-account customers).
    """
    df = df.copy()
    df['entity_n_accounts'] = df.groupby('entity_id')['account_id'].transform('size').astype('int32')

    return df


def standardise_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the pattern rows into the same keys used by the transaction ledger.
    """
    df = df.copy()
    df['ts']              = pd.to_datetime(df['timestamp'], format=TIMESTAMP_FORMAT)
    df['amount_paid']     = df['amount_paid'].astype('float64')
    df['from_account_id'] = make_account_id(df['from_bank'], df['from_account'])
    df['to_account_id']   = make_account_id(df['to_bank'], df['to_account'])

    return df[['attempt_id', 'typology', 'ts', 'from_account_id', 'to_account_id',
               'amount_paid', 'payment_currency', 'payment_format']]


def load_to_duckdb(transactions: pd.DataFrame, accounts: pd.DataFrame,
                   patterns: pd.DataFrame, fx: pd.DataFrame,
                   db_path=DUCKDB_PATH) -> None:
    """
    (Re)create the raw schema from the DDL script and bulk-insert each table.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tables = {
        'raw.transactions':        transactions,
        'raw.accounts':            accounts,
        'raw.laundering_patterns': patterns,
        'analytics.fx_rates':      fx,
    }

    with connect(db_path) as con:
        con.execute((SQL_DIR / 'db_tables_creation_script.sql').read_text())

        for table, frame in tables.items():
            columns = [c[0] for c in con.execute(f"DESCRIBE {table}").fetchall()]
            con.register('frame_view', frame[columns])
            con.execute(f"INSERT INTO {table} SELECT * FROM frame_view")
            con.unregister('frame_view')
            print(f"Loaded {len(frame):>10,} rows into {table}")


# pipeline
def run_pipeline() -> pd.DataFrame:
    """
    Execute the full ETL pipeline and load the raw layer into DuckDB.
    """
    df = extract_transactions(raw_file('Trans.csv'))
    df = standardise_transactions(df)
    validate_transactions(df)

    fx = derive_fx_rates(df)
    df = engineer_features(df, fx)
    df = flag_simulation_tail(df)
    df = assign_transaction_ids(df)
    df = df[TRANSACTION_COL_ORDER]

    accounts = extract_accounts(raw_file('accounts.csv'))
    accounts = standardise_accounts(accounts)
    accounts = deduplicate_accounts(accounts)
    accounts = build_entity_counts(accounts)

    patterns = standardise_patterns(extract_patterns(raw_file('Patterns.txt')))

    load_to_duckdb(df, accounts, patterns, fx)
    print(f"=== ETL pipeline complete — {len(df):,} transactions loaded into {DUCKDB_PATH.name} ===")

    return df


if __name__ == "__main__":
    run_pipeline()
