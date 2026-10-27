"""
Rule-based transaction monitoring (TM) scenarios: the deterministic first line of
detection in a typical AML programme. Each scenario encodes a known money-laundering
red flag; a transaction hitting any scenario raises an alert.

Thresholds are fixed, industry-style defaults (USD 10,000 mirrors the US Currency
Transaction Report threshold) and are deliberately NOT tuned on the labels, so the
rules act as an honest benchmark for the statistical model downstream.

Output: analytics.rule_alerts — one row per transaction with a 0/1 flag per scenario,
the number of scenarios hit and a readable list of triggered scenario codes.
"""
from src.config import DUCKDB_PATH
from src.database import connect


# configuration
REPORTING_THRESHOLD_USD = 10_000

TM_SCENARIOS = {
    'TM01_STRUCTURING': (
        "Payment just below the reporting threshold while the originator's 24h total exceeds it",
        f"amount_paid_usd >= 0.8 * {REPORTING_THRESHOLD_USD} AND amount_paid_usd < {REPORTING_THRESHOLD_USD} "
        f"AND orig_out_amount_usd_24h >= {REPORTING_THRESHOLD_USD}"
    ),
    'TM02_RAPID_MOVEMENT': (
        "Funds received and sent on within 24h (pass-through / mule behaviour)",
        "is_self_transfer = 0 AND orig_in_amount_usd_24h >= 5000 "
        "AND amount_paid_usd BETWEEN 0.8 * orig_in_amount_usd_24h AND 1.2 * orig_in_amount_usd_24h"
    ),
    'TM03_FAN_OUT': (
        "Originator paid 10+ distinct beneficiaries in 24h totalling over the threshold",
        f"orig_out_distinct_cp_24h >= 10 AND orig_out_amount_usd_24h >= {REPORTING_THRESHOLD_USD}"
    ),
    'TM04_FAN_IN': (
        "Beneficiary received from 5+ distinct originators in 24h totalling over the threshold",
        f"benef_in_distinct_cp_24h >= 5 AND benef_in_amount_usd_24h >= {REPORTING_THRESHOLD_USD}"
    ),
    'TM05_HIGH_RISK_CHANNEL': (
        "Cash or crypto payment above the threshold",
        f"payment_format IN ('Cash', 'Bitcoin') AND amount_paid_usd >= {REPORTING_THRESHOLD_USD}"
    ),
    'TM06_NEW_PAYEE_LARGE': (
        "First-ever payment to a beneficiary above the threshold",
        f"pair_count_prior = 0 AND is_self_transfer = 0 AND amount_paid_usd >= {REPORTING_THRESHOLD_USD}"
    ),
    'TM07_AMOUNT_SPIKE': (
        "Payment 3+ standard deviations above the originator's own history (5+ prior payments)",
        "orig_out_count_prior >= 5 AND orig_amount_zscore >= 3"
    ),
    'TM08_CROSS_CURRENCY_LARGE': (
        "Currency-converting payment above the threshold",
        f"is_cross_currency = 1 AND amount_paid_usd >= {REPORTING_THRESHOLD_USD}"
    ),
}


def build_rules_sql(scenarios: dict = TM_SCENARIOS) -> str:
    """
    Compiles the scenario dictionary into a single CREATE TABLE statement.
    """
    flags = ",\n       ".join(
        f"CASE WHEN {predicate} THEN 1 ELSE 0 END::SMALLINT AS {code}"
        for code, (_, predicate) in scenarios.items()
    )
    hit_count = " + ".join(f"(CASE WHEN {predicate} THEN 1 ELSE 0 END)"
                           for _, predicate in scenarios.values())
    rule_list = ", ".join(f"CASE WHEN {predicate} THEN '{code}' END"
                          for code, (_, predicate) in scenarios.items())

    return f"""
CREATE OR REPLACE TABLE analytics.rule_alerts AS
SELECT transaction_id,
       {flags},
       ({hit_count})::SMALLINT AS rule_hit_count,
       concat_ws(', ', {rule_list}) AS rules_triggered
FROM raw.transactions
JOIN analytics.transaction_features USING (transaction_id)
"""


def run_rules(db_path=DUCKDB_PATH) -> None:
    """
    Executes every TM scenario and prints the alert volume and hit rate per rule
    (above-the-line view on the full, labelled history excluding the simulation tail).
    """
    with connect(db_path) as con:
        con.execute(build_rules_sql())

        summary = con.execute(f"""
            SELECT {", ".join(f"SUM({c}) AS {c}" for c in TM_SCENARIOS)},
                   {", ".join(f"SUM({c} * is_laundering) AS {c}_tp" for c in TM_SCENARIOS)},
                   SUM((rule_hit_count > 0)::INT)                 AS any_rule,
                   SUM((rule_hit_count > 0)::INT * is_laundering) AS any_rule_tp,
                   SUM(is_laundering)                             AS positives
            FROM analytics.rule_alerts
            JOIN raw.transactions USING (transaction_id)
            WHERE is_simulation_tail = 0
        """).fetchdf().iloc[0]

    print(f"{'scenario':<28}{'alerts':>12}{'true pos.':>12}{'precision':>11}{'recall':>9}")
    for code in list(TM_SCENARIOS) + ['any_rule']:
        alerts, tp = summary[code], summary[f'{code}_tp']
        precision = tp / alerts if alerts else 0
        print(f"{code:<28}{alerts:>12,.0f}{tp:>12,.0f}{precision:>11.2%}{tp / summary['positives']:>9.1%}")


if __name__ == "__main__":
    run_rules()
    print("=== Transaction monitoring rules complete ===")
