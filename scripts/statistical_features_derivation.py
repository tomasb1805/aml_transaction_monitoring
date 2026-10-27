"""
Derives the statistical behavioural features used for transaction monitoring.

The heavy lifting runs inside DuckDB (SQL_scripts/db_transaction_features.sql)
because rolling-window aggregates over ~10M account legs are far faster as SQL
window functions than as pandas group-bys.

Account-level behavioural profile (prior days only), the AML counterpart of the
credit-risk Plaid features:
    avg_daily_inflow        -  mean daily incoming value (USD)
    inflow_stability_ratio  -  1 minus the coefficient of variation of daily inflows
                               (higher = steadier, range 0-1)
    spend_to_income_ratio   -  total outflows / total inflows (≈1 for pass-through accounts)
    days_net_negative       -  number of days where outflows exceeded inflows
    cash_withdrawal_ratio   -  share of outgoing value paid in cash
    crypto_outflow_ratio    -  share of outgoing value paid in Bitcoin

Transaction-level velocity / network features (rolling 24h and cumulative):
    fan-out and fan-in degree, inflow and outflow value, payment-size z-score,
    time since previous activity, repeat payments to the same beneficiary.
"""
import time

from src.config import DUCKDB_PATH, SQL_DIR
from src.database import connect


FEATURE_SQL = SQL_DIR / 'db_transaction_features.sql'


def run_feature_sql(db_path=DUCKDB_PATH, sql_path=FEATURE_SQL) -> None:
    """
    Executes the feature-derivation SQL script statement by statement.
    """
    statements = [s.strip() for s in sql_path.read_text().split(';') if s.strip()]

    with connect(db_path) as con:
        for statement in statements:
            table = statement.split('TABLE', 1)[1].split()[0] if 'CREATE' in statement else '?'
            start = time.perf_counter()
            con.execute(statement)
            n_rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"Built {table:<38} {n_rows:>12,} rows  ({time.perf_counter() - start:5.1f}s)")


def pipeline() -> None:
    """
    Orchestrates the feature derivation workflow.
    """
    run_feature_sql()
    print("=== Statistical feature derivation complete ===")


if __name__ == "__main__":
    pipeline()
