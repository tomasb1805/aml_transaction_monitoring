-- DuckDB schema for the AML transaction-monitoring pipeline.
-- raw       : cleaned source data loaded by scripts/etl.py
-- analytics : derived features, rule alerts and the modelling table

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Raw layer tables
CREATE OR REPLACE TABLE raw.transactions (
    transaction_id          BIGINT          PRIMARY KEY,
    ts                      TIMESTAMP       NOT NULL,
    txn_date                DATE            NOT NULL,
    from_bank               VARCHAR         NOT NULL,
    from_account            VARCHAR         NOT NULL,
    from_account_id         VARCHAR         NOT NULL,
    to_bank                 VARCHAR         NOT NULL,
    to_account              VARCHAR         NOT NULL,
    to_account_id           VARCHAR         NOT NULL,
    amount_received         DOUBLE          NOT NULL CHECK (amount_received >= 0),
    receiving_currency      VARCHAR         NOT NULL,
    amount_paid             DOUBLE          NOT NULL CHECK (amount_paid >= 0),
    payment_currency        VARCHAR         NOT NULL,
    payment_format          VARCHAR         NOT NULL,
    amount_paid_usd         DOUBLE          NOT NULL,
    amount_received_usd     DOUBLE          NOT NULL,
    is_self_transfer        SMALLINT        NOT NULL CHECK (is_self_transfer IN (0, 1)),
    is_cross_bank           SMALLINT        NOT NULL CHECK (is_cross_bank IN (0, 1)),
    is_cross_currency       SMALLINT        NOT NULL CHECK (is_cross_currency IN (0, 1)),
    hour_of_day             SMALLINT        NOT NULL,
    day_of_week             SMALLINT        NOT NULL,
    is_simulation_tail      SMALLINT        NOT NULL CHECK (is_simulation_tail IN (0, 1)),
    is_laundering           SMALLINT        NOT NULL CHECK (is_laundering IN (0, 1))
);

CREATE OR REPLACE TABLE raw.accounts (
    account_id              VARCHAR         PRIMARY KEY,
    bank_id                 VARCHAR         NOT NULL,
    bank_name               VARCHAR,
    account_number          VARCHAR         NOT NULL,
    entity_id               VARCHAR         NOT NULL,
    entity_name             VARCHAR,
    entity_type             VARCHAR         NOT NULL,
    entity_n_accounts       INTEGER         NOT NULL
);

CREATE OR REPLACE TABLE raw.laundering_patterns (
    attempt_id              INTEGER         NOT NULL,
    typology                VARCHAR         NOT NULL,
    ts                      TIMESTAMP       NOT NULL,
    from_account_id         VARCHAR         NOT NULL,
    to_account_id           VARCHAR         NOT NULL,
    amount_paid             DOUBLE          NOT NULL,
    payment_currency        VARCHAR         NOT NULL,
    payment_format          VARCHAR         NOT NULL
);

-- Analytics layer tables
CREATE OR REPLACE TABLE analytics.fx_rates (
    currency                VARCHAR         PRIMARY KEY,
    usd_rate                DOUBLE          NOT NULL,   -- 1 unit of currency in USD
    n_observations          INTEGER         NOT NULL
);
