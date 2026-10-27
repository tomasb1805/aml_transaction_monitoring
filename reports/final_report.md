AML Transaction Monitoring and Alert Triage Automation

## 1. Abstract

This project builds an automated anti-money-laundering (AML) transaction-monitoring pipeline on the IBM synthetic AML dataset (HI-Small: 5.08 million payments, 0.10% laundering). It re-engineers the statistical approach of the earlier credit-risk scoring project for financial-crime detection. The steps are rule-based customer segmentation, behavioural statistics, an XGBoost classifier tuned with randomised search, ROC and precision-recall evaluation, SHAP interpretability, and a three-tier decision policy. These are embedded in a typical AML workflow: ingestion and KYC enrichment, rule-based detection scenarios, risk scoring, alert triage, and investigator-ready explanations.

On an out-of-time test window, the model-driven alert queue catches **82.2%** of laundering payments with **9,487** alerts. The rules engine raises **175,200** alerts and catches only **56.4%**, so the model cuts alert volume by 95% and lifts precision 27-fold.

## 2. Data Architecture and ETL

Data is acquired with `kagglehub`, transformed in Python, and stored in an embedded DuckDB database layered into `raw` and `analytics` schemas, mirroring the PostgreSQL design of the credit-risk project.

#### Data acquisition (kaggle_data_client.py):
Downloads the transaction ledger, the account/KYC reference file and the laundering-pattern file of the selected variant only. The full six-variant dataset (~41 GB) is not required.

#### Transaction ETL (etl.py):
Function-per-stage pipeline: explicit column naming (the raw header repeats `Account`), composite account keys with normalised bank IDs (100% KYC join coverage), validation (nulls, binary label, payment formats, non-negative amounts), and USD normalisation. FX rates are derived from the data itself as the median `amount_received / amount_paid` over cross-currency payments settled in USD.

#### Simulation tail:
After 10 September 2022 the simulator stops generating normal activity and only completes in-flight laundering attempts (1,108 rows, ~60% laundering). These days are flagged in the raw layer and excluded from modelling, as they would otherwise inflate out-of-time metrics.

#### Database Schema:
`raw.transactions`, `raw.accounts`, `raw.laundering_patterns`, plus `analytics.fx_rates`, `analytics.account_daily_activity`, `analytics.account_behaviour_profile`, `analytics.account_leg_velocity`, `analytics.transaction_features` and `analytics.rule_alerts`. A SQL join (`db_join_modeling_table.sql`) assembles the modelling table, exported to parquet.

## 3. Feature Engineering

All behavioural features are **point-in-time**: they are computed only from activity strictly before each payment, using DuckDB window functions that end one minute before the payment. The payment's own contribution is added back explicitly. Timestamps have minute resolution and the order of payments within a minute is unknown, so same-minute activity is treated as not yet observed. This is material on this dataset: 36% of laundering payments share their minute with an ACH self-transfer by the same account, against 1.6% of legitimate payments. An earlier version whose windows included the current minute partly learned that artefact (test Average Precision 0.537 with same-minute activity, 0.419 without, under otherwise identical settings).

#### Behavioural Statistics (statistical_features_derivation.py):
The five cash-flow statistics of the credit-risk project are re-derived per account, over all prior days:

- avg_daily_inflow

- inflow_stability_ratio [computed as: $\max(0,\ 1 - \sigma_{\text{inflow}} / \mu_{\text{inflow}})$ ]

- spend_to_income_ratio

- days_net_negative

- cash_withdrawal_ratio (plus crypto_outflow_ratio)

#### Velocity and Network Features:
Rolling 24-hour and cumulative statistics for both the originator and the beneficiary: payment counts and values, distinct counterparties (fan-out / fan-in degree), inflow received before sending on, time since previous activity, and the payment-size z-score against the account's own history:

$z = \dfrac{\text{amount} - \bar{x}_{\text{prior}}}{s_{\text{prior}}}$

The number of prior payments between the same originator and beneficiary identifies **new payees**.

#### Customer Risk Segmentation (customer_risk_segmentation.py):
The counterpart of the credit-risk personas: a rule hierarchy assigns each originator a segment from its behaviour on the transaction date (new_relationship, crypto_user, cash_intensive, pass_through, high_degree_hub, net_spender), and the residual mainstream segment is refined by KYC entity type. Accounts with no history yet receive `_missing` flags for every behavioural statistic, preserving the structural information as the original project did.

#### Interaction Features (build_model_dataset.py):
- pass_through_ratio = amount / (inflow received in the last 24h + 1)
- amount_to_avg_inflow = amount / (average daily inflow + 1)
- orig_payee_concentration_24h = payments / distinct beneficiaries in 24h
- orig_flow_balance_24h = (outflow − inflow) / (outflow + inflow + 1)
- is_new_payee

## 4. Detection Strategy

#### Rule-based Transaction Monitoring (transaction_monitoring_rules.py):
Eight industry-style scenarios (structuring, rapid movement, fan-out, fan-in, high-risk channel, large payment to a new payee, statistical amount spike, large cross-currency payment) with fixed USD 10,000-based thresholds. They are not tuned on labels, so they act as an honest benchmark. Their hit count is also passed to the model.

#### Statistical Model (train_xgb_model.py):
An XGBClassifier inside a Scikit-Learn pipeline. A ColumnTransformer applies median imputation to continuous variables, and mode imputation with one-hot encoding to categorical variables.

#### Chronological Validation:
Payments are split by time into train (60%), validation (20%) and test (20%) windows, matching the protocol of the IBM benchmark paper. A shuffled split would let the model learn from laundering patterns that occur after the payments it is scored on.

#### Imbalance Handling:
The credit-risk project re-weighted the minority class to a balanced prior. Here four strategies were compared in an ablation (`imbalance_ablation.py`). Each was tuned over the same 15 hyperparameter candidates, and the winner was chosen by validation Average Precision, a rule fixed before running:

| Strategy | `scale_pos_weight` | Validation AP |
|---|---|---|
| Full training window (3,046,186 rows), no re-weighting | 1 | **0.598** |
| 20% of legitimate payments, no re-weighting | 1 | 0.579 |
| Full training window, sqrt re-weighting | 36.4 | 0.563 |
| 20% of legitimate payments, balanced re-weighting | 265.1 | 0.448 |

Undersampling followed by `scale_pos_weight = negatives / positives` does not penalise the minority class twice: in expectation it equals balanced weighting on the full data. But a balanced prior was clearly the worst choice for ranking quality at a 0.08% laundering rate. The final model therefore trains on the full training window without re-weighting. The validation and test windows are never sampled.

#### Hyperparameter Optimization:
RandomizedSearchCV (30 candidates, same search space as the credit-risk model) with a PredefinedSplit: each candidate is fitted on the training window and scored on the later validation window. The objective is **Average Precision**, which reflects performance on the 0.1% minority class far better than ROC-AUC. The best candidate is refitted on the training window only.

Selected configuration: 400 trees, depth 6, learning rate 0.05, subsample 0.8, colsample 0.8, min_child_weight 3, α = 0, λ = 1.

## 5. Model Evaluation

Evaluated out-of-time on the test window (8–10 September 2022: 1,015,751 payments, 1,143 laundering).

#### ROC-AUC:
0.987 (validation 0.988). The rules engine's operating point (17.2% false-positive rate, 56.4% recall) lies far inside the model's ROC curve.

#### Precision-Recall (Average Precision):
AP = 0.571 (validation 0.609) against a random baseline of 0.11%, a **507×** lift.

#### Rules engine vs model:

| Alerting strategy | Alerts | Alert rate | Laundering caught | Precision | Recall |
|---|---|---|---|---|---|
| TM rules (any scenario) | 175,200 | 17.25% | 645 | 0.37% | 56.4% |
| Model alert queue (REVIEW + ESCALATE) | 9,487 | 0.93% | 939 | 9.9% | 82.2% |
| Model ESCALATE tier only | 742 | 0.07% | 544 | 73.3% | 47.6% |

#### Rule rationalisation:
TM06 (large payment to a new payee) is the strongest individual scenario (14.3% precision, 31.6% recall). TM03 (fan-out) generates 104,012 alerts at 0.14% precision because large hub accounts trip it constantly. TM05 (cash/crypto) caught 2 laundering payments in 28,932 alerts, and TM08 (cross-currency) caught none, since no laundering in this dataset converts currency. In a real programme these findings would feed scenario tuning or retirement.

#### Typology coverage:
Detection rates for the model's alert queue are 93–100% across all eight labelled typologies (FAN-OUT, FAN-IN, CYCLE, STACK, BIPARTITE, SCATTER-GATHER, GATHER-SCATTER, RANDOM), against 52–69% for the rules. Laundering payments not attributed to a labelled pattern (40% of test laundering) are the hardest: 59% detected, against 47% for the rules.

## 6. Model Interpretability and Alert Triage

To meet the explainability expectations placed on AML systems, SHAP (SHapley Additive exPlanations) is integrated in the pipeline (explain_model.py).

#### Global Interpretability:
The beeswarm and dependence plots identify the macro-level drivers of laundering risk: ACH payments, first-time payees (`pair_count_prior = 0`), originators reactivating after a long gap (dormant-account behaviour), originators that have paid many distinct beneficiaries, and payment size. The amount effect is non-monotonic: risk peaks around USD 10,000, the currency-reporting threshold, and falls for very small and very large payments: the signature of funds moved in mid-sized amounts that stay below the radar.

#### Local Interpretability:
Waterfall plots break individual risk scores into the base value and specific feature contributions, one per triage tier. Every alert in the investigator queue (`reports/alerts/alert_queue_test_window.csv`) carries its triggered TM scenarios and its top three SHAP reason codes, computed exactly with XGBoost's native TreeSHAP, as a starting point for the investigation narrative.

#### Automated Triage Policy:
Risk scores are mapped to discrete investigation actions. Their calibration as probabilities has not been verified, so the thresholds are calibrated on the validation window against operating targets instead of fixed probability cut-offs:

$\text{score} < 0.011$: CLEAR (no alert)

$0.011 \leq \text{score} < 0.342$: REVIEW (Level-1 analyst), sized so that alerts capture 80% of validation laundering

$\text{score} \geq 0.342$: ESCALATE (Level-2 investigation / potential SAR), the lowest score with 80% validation precision

On the test window, ESCALATE holds 742 payments at 73% precision (47.6% of all laundering). REVIEW holds 8,745 payments at 4.5%, and 17.8% of laundering is auto-cleared. The gap between the 80% validation precision target and the 73% achieved out-of-time is the kind of drift a production deployment would monitor.

## 7. Conclusion

The re-engineered pipeline shows that the statistical toolkit of the credit-risk model transfers directly to financial-crime detection once the problem is framed correctly. The prediction unit becomes the payment; features become strictly point-in-time behavioural statistics; validation becomes chronological; the tuning objective becomes average precision; the class-imbalance treatment is chosen empirically rather than inherited; and decision thresholds are calibrated to investigation capacity rather than set by hand. Layered on top of conventional monitoring rules, the model raises 95% fewer alerts while catching 26 percentage points more laundering. SHAP reason codes keep every alert explainable to investigators and auditors.

Caveats: the data is synthetic and spans only 18 days, and some strong signals (e.g. the concentration of laundering in ACH payments) are likely artefacts of the generator. The methodology transfers to real transaction data, but the weights and thresholds would need re-fitting.
