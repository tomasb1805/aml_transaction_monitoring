-- Modelling table: one row per transaction, joining the ledger with
-- KYC attributes, the prior-day behavioural profile of both parties,
-- point-in-time velocity features and the TM rule outputs.
-- The simulation tail (see scripts/etl.py) is excluded.
WITH pattern_typology AS (
    -- laundering typology from the Patterns file — used for evaluation only, never as a feature
    SELECT ts, from_account_id, to_account_id, amount_paid, payment_currency, payment_format,
           ANY_VALUE(typology) AS typology
    FROM raw.laundering_patterns
    GROUP BY ALL
)
SELECT
    t.transaction_id,
    t.ts,
    t.from_account_id,
    t.to_account_id,
    t.amount_paid_usd,
    t.amount_received_usd,
    t.payment_format,
    t.payment_currency,
    t.receiving_currency,
    t.is_self_transfer,
    t.is_cross_bank,
    t.is_cross_currency,
    t.hour_of_day,
    -- KYC
    fa.entity_type                  AS orig_entity_type,
    fa.entity_n_accounts            AS orig_entity_n_accounts,
    ta.entity_type                  AS benef_entity_type,
    ta.entity_n_accounts            AS benef_entity_n_accounts,
    -- originator behavioural profile (prior days)
    op.prior_days_observed          AS orig_prior_days_observed,
    op.avg_daily_inflow             AS orig_avg_daily_inflow,
    op.inflow_stability_ratio       AS orig_inflow_stability_ratio,
    op.spend_to_income_ratio        AS orig_spend_to_income_ratio,
    op.days_net_negative            AS orig_days_net_negative,
    op.cash_withdrawal_ratio        AS orig_cash_withdrawal_ratio,
    op.crypto_outflow_ratio         AS orig_crypto_outflow_ratio,
    -- beneficiary behavioural profile (prior days)
    bp.avg_daily_inflow             AS benef_avg_daily_inflow,
    bp.spend_to_income_ratio        AS benef_spend_to_income_ratio,
    bp.days_net_negative            AS benef_days_net_negative,
    -- velocity / network features
    f.* EXCLUDE (transaction_id),
    -- TM rules engine
    r.* EXCLUDE (transaction_id),
    -- evaluation-only columns
    pt.typology                     AS eval_typology,
    t.is_laundering
FROM raw.transactions t
JOIN analytics.transaction_features f
  ON f.transaction_id = t.transaction_id
JOIN analytics.rule_alerts r
  ON r.transaction_id = t.transaction_id
LEFT JOIN raw.accounts fa
  ON fa.account_id = t.from_account_id
LEFT JOIN raw.accounts ta
  ON ta.account_id = t.to_account_id
LEFT JOIN analytics.account_behaviour_profile op
  ON op.account_id = t.from_account_id AND op.txn_date = t.txn_date
LEFT JOIN analytics.account_behaviour_profile bp
  ON bp.account_id = t.to_account_id AND bp.txn_date = t.txn_date
LEFT JOIN pattern_typology pt
  ON  pt.ts = t.ts
  AND pt.from_account_id  = t.from_account_id
  AND pt.to_account_id    = t.to_account_id
  AND pt.amount_paid      = t.amount_paid
  AND pt.payment_currency = t.payment_currency
  AND pt.payment_format   = t.payment_format
WHERE t.is_simulation_tail = 0
ORDER BY t.transaction_id
