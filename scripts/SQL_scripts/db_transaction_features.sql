-- Point-in-time behavioural features for transaction monitoring.
-- Every feature is computed only from activity strictly before the transaction,
-- so the model never sees the future (no temporal leakage across the train/test split).
-- Timestamps have minute resolution and the order of payments booked within the same
-- minute is unknown, so same-minute activity is treated as not-yet-observed:
-- every window ends at INTERVAL 1 MINUTE PRECEDING (earlier minutes only), and the
-- screened payment's own contribution is added back explicitly where intended.
-- This also keeps a self-transfer's own incoming leg out of the originator's inflow.


-- 1. Daily cash-flow per account (self-transfers excluded: they move no money between parties)
CREATE OR REPLACE TABLE analytics.account_daily_activity AS
WITH legs AS (
    SELECT from_account_id                                                  AS account_id,
           txn_date,
           0.0                                                              AS inflow_usd,
           amount_paid_usd                                                  AS outflow_usd,
           CASE WHEN payment_format = 'Cash'    THEN amount_paid_usd ELSE 0 END AS cash_outflow_usd,
           CASE WHEN payment_format = 'Bitcoin' THEN amount_paid_usd ELSE 0 END AS crypto_outflow_usd
    FROM raw.transactions
    WHERE is_self_transfer = 0
    UNION ALL
    SELECT to_account_id, txn_date, amount_received_usd, 0.0, 0.0, 0.0
    FROM raw.transactions
    WHERE is_self_transfer = 0
)
SELECT account_id,
       txn_date,
       SUM(inflow_usd)         AS inflow_usd,
       SUM(outflow_usd)        AS outflow_usd,
       SUM(cash_outflow_usd)   AS cash_outflow_usd,
       SUM(crypto_outflow_usd) AS crypto_outflow_usd
FROM legs
GROUP BY account_id, txn_date;


-- 2. Behavioural profile per account-day, built from all PRIOR days only
--    (the account's "known history" when a transaction on that day is screened).
--    Inactive days count as zero-flow days, from the account's first active day onward.
CREATE OR REPLACE TABLE analytics.account_behaviour_profile AS
WITH calendar AS (
    SELECT DISTINCT txn_date FROM raw.transactions
),
first_seen AS (
    SELECT account_id, MIN(txn_date) AS first_date
    FROM analytics.account_daily_activity
    GROUP BY account_id
),
daily AS (
    SELECT f.account_id,
           c.txn_date,
           COALESCE(a.inflow_usd, 0)         AS inflow_usd,
           COALESCE(a.outflow_usd, 0)        AS outflow_usd,
           COALESCE(a.cash_outflow_usd, 0)   AS cash_outflow_usd,
           COALESCE(a.crypto_outflow_usd, 0) AS crypto_outflow_usd
    FROM first_seen f
    JOIN calendar c
      ON c.txn_date >= f.first_date
    LEFT JOIN analytics.account_daily_activity a
      ON a.account_id = f.account_id AND a.txn_date = c.txn_date
)
SELECT account_id,
       txn_date,
       COUNT(*) OVER w                                                          AS prior_days_observed,
       AVG(inflow_usd) OVER w                                                   AS avg_daily_inflow,
       GREATEST(0, 1 - STDDEV_SAMP(inflow_usd) OVER w
                     / NULLIF(AVG(inflow_usd) OVER w, 0))                       AS inflow_stability_ratio,
       SUM(outflow_usd) OVER w / NULLIF(SUM(inflow_usd) OVER w, 0)              AS spend_to_income_ratio,
       SUM(CASE WHEN inflow_usd - outflow_usd < 0 THEN 1 ELSE 0 END) OVER w     AS days_net_negative,
       SUM(cash_outflow_usd) OVER w / NULLIF(SUM(outflow_usd) OVER w, 0)        AS cash_withdrawal_ratio,
       SUM(crypto_outflow_usd) OVER w / NULLIF(SUM(outflow_usd) OVER w, 0)      AS crypto_outflow_ratio
FROM daily
WINDOW w AS (PARTITION BY account_id ORDER BY txn_date
             ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING);


-- 3. Velocity and network features per account "leg".
--    Every transaction produces an 'out' leg (originator) and an 'in' leg (beneficiary),
--    so both parties' recent activity can be measured at the moment of the payment.
CREATE OR REPLACE TABLE analytics.account_leg_velocity AS
WITH legs AS (
    SELECT transaction_id, from_account_id AS account_id, to_account_id AS counterparty_id,
           ts, 'out' AS direction, amount_paid_usd AS amount_usd, payment_format
    FROM raw.transactions
    UNION ALL
    SELECT transaction_id, to_account_id, from_account_id,
           ts, 'in', amount_received_usd, payment_format
    FROM raw.transactions
),
prior AS (
    -- activity in earlier minutes only (rolling 24h and since first seen)
    SELECT transaction_id, direction, amount_usd, ts,
           COUNT(*)                        FILTER (WHERE direction = 'out') OVER w24  AS out_count_24h,
           SUM(amount_usd)                 FILTER (WHERE direction = 'out') OVER w24  AS out_amount_usd_24h,
           COUNT(DISTINCT counterparty_id) FILTER (WHERE direction = 'out') OVER w24  AS out_distinct_cp_24h,
           COUNT(*)                        FILTER (WHERE direction = 'in')  OVER w24  AS in_count_24h,
           SUM(amount_usd)                 FILTER (WHERE direction = 'in')  OVER w24  AS in_amount_usd_24h,
           COUNT(DISTINCT counterparty_id) FILTER (WHERE direction = 'in')  OVER w24  AS in_distinct_cp_24h,
           COUNT(DISTINCT counterparty_id) FILTER (WHERE direction = 'out') OVER wcum AS out_distinct_cp_cum,
           COUNT(DISTINCT counterparty_id) FILTER (WHERE direction = 'in')  OVER wcum AS in_distinct_cp_cum,
           COUNT(*)                                                         OVER wcum AS legs_cum,
           COUNT(DISTINCT payment_format)                                   OVER wcum AS distinct_formats_cum,
           COUNT(*)                        FILTER (WHERE direction = 'out') OVER wcum AS out_count_prior,
           AVG(amount_usd)                 FILTER (WHERE direction = 'out') OVER wcum AS out_amount_mean_prior,
           STDDEV_SAMP(amount_usd)         FILTER (WHERE direction = 'out') OVER wcum AS out_amount_std_prior,
           MAX(ts)                                                          OVER wcum AS prev_leg_ts,
           -- was this counterparty / payment format already seen in earlier minutes?
           COUNT(*) OVER (PARTITION BY account_id, counterparty_id, direction ORDER BY ts
                          RANGE BETWEEN INTERVAL 24 HOURS PRECEDING AND INTERVAL 1 MINUTE PRECEDING) AS cp_seen_24h,
           COUNT(*) OVER (PARTITION BY account_id, counterparty_id, direction ORDER BY ts
                          RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL 1 MINUTE PRECEDING)        AS cp_seen_cum,
           COUNT(*) OVER (PARTITION BY account_id, payment_format ORDER BY ts
                          RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL 1 MINUTE PRECEDING)        AS format_seen_cum
    FROM legs
    WINDOW w24  AS (PARTITION BY account_id ORDER BY ts
                    RANGE BETWEEN INTERVAL 24 HOURS PRECEDING AND INTERVAL 1 MINUTE PRECEDING),
           wcum AS (PARTITION BY account_id ORDER BY ts
                    RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL 1 MINUTE PRECEDING)
)
SELECT transaction_id,
       direction,
       -- rolling 24h activity: earlier minutes + the screened payment itself
       out_count_24h + (direction = 'out')::INT                                         AS out_count_24h,
       COALESCE(out_amount_usd_24h, 0) + CASE WHEN direction = 'out' THEN amount_usd ELSE 0 END AS out_amount_usd_24h,
       out_distinct_cp_24h + (direction = 'out' AND cp_seen_24h = 0)::INT               AS out_distinct_cp_24h,
       in_count_24h + (direction = 'in')::INT                                           AS in_count_24h,
       COALESCE(in_amount_usd_24h, 0) + CASE WHEN direction = 'in' THEN amount_usd ELSE 0 END   AS in_amount_usd_24h,
       in_distinct_cp_24h + (direction = 'in' AND cp_seen_24h = 0)::INT                 AS in_distinct_cp_24h,
       -- cumulative network degree since the account was first seen
       out_distinct_cp_cum + (direction = 'out' AND cp_seen_cum = 0)::INT               AS out_distinct_cp_cum,
       in_distinct_cp_cum + (direction = 'in' AND cp_seen_cum = 0)::INT                 AS in_distinct_cp_cum,
       legs_cum + 1                                                                     AS legs_cum,
       distinct_formats_cum + (format_seen_cum = 0)::INT                                AS distinct_formats_cum,
       -- the account's own historical payment-size baseline (earlier minutes only)
       out_count_prior,
       out_amount_mean_prior,
       out_amount_std_prior,
       date_diff('minute', prev_leg_ts, ts)                                             AS mins_since_prev_leg
FROM prior;


-- 4. One row per transaction: originator and beneficiary features side by side
CREATE OR REPLACE TABLE analytics.transaction_features AS
WITH pair AS (
    SELECT transaction_id,
           -- earlier minutes + the screened payment itself
           1 + COUNT(*) OVER (PARTITION BY from_account_id, to_account_id ORDER BY ts
                              RANGE BETWEEN INTERVAL 24 HOURS PRECEDING AND INTERVAL 1 MINUTE PRECEDING) AS pair_count_24h,
           -- earlier minutes only (0 = first payment to this beneficiary)
           COUNT(*) OVER (PARTITION BY from_account_id, to_account_id ORDER BY ts
                          RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL 1 MINUTE PRECEDING)         AS pair_count_prior
    FROM raw.transactions
)
SELECT t.transaction_id,
       -- originator: rolling 24h
       o.out_count_24h                                                  AS orig_out_count_24h,
       o.out_amount_usd_24h                                             AS orig_out_amount_usd_24h,
       o.out_distinct_cp_24h                                            AS orig_out_distinct_cp_24h,
       o.in_count_24h                                                   AS orig_in_count_24h,
       COALESCE(o.in_amount_usd_24h, 0)                                 AS orig_in_amount_usd_24h,
       o.in_distinct_cp_24h                                             AS orig_in_distinct_cp_24h,
       -- originator: history
       o.out_distinct_cp_cum                                            AS orig_out_distinct_cp_cum,
       o.in_distinct_cp_cum                                             AS orig_in_distinct_cp_cum,
       o.legs_cum                                                       AS orig_legs_cum,
       o.distinct_formats_cum                                           AS orig_distinct_formats_cum,
       o.mins_since_prev_leg                                            AS orig_mins_since_prev_leg,
       o.out_count_prior                                                AS orig_out_count_prior,
       (t.amount_paid_usd - o.out_amount_mean_prior)
           / NULLIF(o.out_amount_std_prior, 0)                          AS orig_amount_zscore,
       -- beneficiary: rolling 24h and history
       b.in_count_24h                                                   AS benef_in_count_24h,
       b.in_amount_usd_24h                                              AS benef_in_amount_usd_24h,
       b.in_distinct_cp_24h                                             AS benef_in_distinct_cp_24h,
       b.out_count_24h                                                  AS benef_out_count_24h,
       b.out_distinct_cp_24h                                            AS benef_out_distinct_cp_24h,
       b.in_distinct_cp_cum                                             AS benef_in_distinct_cp_cum,
       b.out_distinct_cp_cum                                            AS benef_out_distinct_cp_cum,
       b.legs_cum                                                       AS benef_legs_cum,
       -- originator → beneficiary relationship
       p.pair_count_24h,
       p.pair_count_prior
FROM raw.transactions t
JOIN analytics.account_leg_velocity o
  ON o.transaction_id = t.transaction_id AND o.direction = 'out'
JOIN analytics.account_leg_velocity b
  ON b.transaction_id = t.transaction_id AND b.direction = 'in'
JOIN pair p
  ON p.transaction_id = t.transaction_id;
