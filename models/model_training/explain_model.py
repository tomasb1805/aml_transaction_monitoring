import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from models.model_training.alert_triage import triage_decision
from models.model_training.plot_style import apply_style, save_figure, TIER_COLOURS
from src.config import ALERTS_DIR, PLOTS_DIR


# extract readable feature names from the pipeline

def get_feature_names(pipeline) -> list:
    """
    Extracts transformed column names from the ColumnTransformer.
    Strips sklearn prefixes (num__, cat__) for clean SHAP labels.
    """
    raw_names = (
        pipeline.named_steps['preprocessing']
        .get_feature_names_out()
    )

    clean = [n.replace('num__', '').replace('cat__', '') for n in raw_names]
    return clean


def get_shap_input(pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Transforms X through the preprocessor only, returning a numpy array
    that matches the feature space the model was trained on.
    """
    return pipeline.named_steps['preprocessing'].transform(X)



# SHAP setup

def build_shap_explainer(pipeline, X_train: pd.DataFrame):
    """
    Builds a TreeExplainer using the fitted model extracted from the pipeline.
    Uses a background sample of 500 rows as the reference population.
    """
    model        = pipeline.named_steps['model']
    X_train_t    = get_shap_input(pipeline, X_train)
    background   = shap.sample(X_train_t, 500, random_state=11)

    explainer    = shap.TreeExplainer(model, background)
    return explainer


def compute_shap_values(explainer, pipeline, X: pd.DataFrame):
    """
    Computes SHAP Explanation object with clean feature names attached.
    Returns an shap.Explanation object ready for all plot types.
    """
    X_t           = get_shap_input(pipeline, X)
    feature_names = get_feature_names(pipeline)

    shap_values = explainer(X_t)
    shap_values.feature_names = feature_names   # attach clean names
    return shap_values


def explanation_sample(tiers: np.ndarray, n_random: int = 1000,
                       n_alerts: int = 1000, seed: int = 11) -> np.ndarray:
    """
    Positional indices for the SHAP sample: a random slice of all screened
    transactions plus a slice of raised alerts. With ~0.1% laundering, a purely
    random sample would contain almost no alerts to explain.
    """
    rng = np.random.default_rng(seed)
    alerted = np.flatnonzero(tiers != 'CLEAR')

    random_idx = rng.choice(len(tiers), min(n_random, len(tiers)), replace=False)
    alert_idx  = rng.choice(alerted, min(n_alerts, len(alerted)), replace=False)

    return np.unique(np.r_[random_idx, alert_idx])



# SHAP Beeswarm Summary

def plot_shap_summary(shap_values, max_display: int = 20):
    """
    Global feature importance beeswarm plot.
    Each dot is one transaction; colour = feature value (red=high, blue=low).
    Ranked by mean |SHAP| — shows which features drive laundering risk most.
    """
    apply_style()
    fig = plt.figure()
    shap.plots.beeswarm(
        shap_values,
        max_display=max_display,
        show=False
    )
    plt.title('SHAP Summary — Global Feature Impact on Laundering Risk')
    save_figure(fig, 'SHAP-Beeswarm-plot.png')



# Top Features - plot for individual contributions to model

def plot_shap_dependence(shap_values, pipeline, X_sample: pd.DataFrame,
                         top_n: int = 5):
    """
    Dependence plots for the top_n most important features.
    Shows how each feature's value relates to its SHAP contribution,
    coloured by the feature that most interacts with it (auto-selected).
    """
    apply_style()
    X_t           = get_shap_input(pipeline, X_sample)
    feature_names = get_feature_names(pipeline)
    shap_matrix   = shap_values.values

    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    top_indices   = np.argsort(mean_abs_shap)[::-1][:top_n]

    # the top features change between runs: drop dependence plots from previous runs
    for old_plot in PLOTS_DIR.glob('SHAP-Dependence-*.png'):
        old_plot.unlink()

    for rank, idx in enumerate(top_indices, start=1):
        feat_name = feature_names[idx]
        fig, ax = plt.subplots(figsize=(7, 4))
        shap.dependence_plot(
            ind=idx,
            shap_values=shap_matrix,
            features=X_t,
            feature_names=feature_names,
            interaction_index='auto',
            ax=ax,
            show=False
        )
        # money and count features span several orders of magnitude: log-compress the axis
        values = X_t[:, idx]
        if values.min() >= 0 and values.max() > 1_000:
            ax.set_xscale('symlog', linthresh=1)
            ax.set_xlim(left=0)
        ax.set_title(f'SHAP Dependence — {feat_name}')
        save_figure(fig, f'SHAP-Dependence-{rank}-{feat_name}.png')



# Waterfall Plot for individual transactions
# explains how their features contributed to the
# final triage outcome/decision

def plot_waterfall(shap_values, y_proba: np.ndarray, thresholds: dict,
                   txn_idx: int, transaction_id: int):
    """
    Waterfall plot for a single transaction.
    Shows exactly why the model gave this specific risk score,
    with the triage decision banner at the top.
    """
    prob     = y_proba[txn_idx]
    decision = str(triage_decision(prob, thresholds))

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.sca(ax)

    shap.plots.waterfall(
        shap_values[txn_idx],
        max_display=12,
        show=False
    )

    plt.title(
        f'Transaction #{transaction_id} — Laundering Risk Score: {prob:.3f}  |  '
        f'Decision: {decision}',
        fontsize=11,
        color=TIER_COLOURS[decision],
        fontweight='bold'
    )
    save_figure(plt.gcf(), f'Waterfall-{decision.title()}.png')


def plot_example_transactions(shap_values, df_sample: pd.DataFrame,
                              y_proba: np.ndarray, thresholds: dict):
    """
    Plots waterfall explanations for three representative transactions:
    one from each triage tier (Clear / Review / Escalate).
    For the alert tiers a confirmed laundering case is preferred, since that is
    the explanation an investigator would document.
    """
    tiers      = triage_decision(y_proba, thresholds)
    laundering = df_sample['is_laundering'].to_numpy() == 1

    for tier, label in [('CLEAR',    'Auto-Clear Example'),
                        ('REVIEW',   'L1 Review Example'),
                        ('ESCALATE', 'L2 Escalation Example')]:
        in_tier = tiers == tier
        preferred = np.flatnonzero(in_tier & laundering) if tier != 'CLEAR' else np.flatnonzero(in_tier & ~laundering)
        candidates = preferred if len(preferred) else np.flatnonzero(in_tier)
        if not len(candidates):
            print(f"No sampled transaction in tier {tier}")
            continue

        idx = candidates[0]
        print(f"\n--- {label} ---")
        plot_waterfall(shap_values, y_proba, thresholds, idx,
                       int(df_sample['transaction_id'].iloc[idx]))



# Alert queue with reason codes for investigators

def reason_codes(pipeline, X: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """
    Top-k features pushing each transaction towards laundering, as readable
    reason codes. Uses XGBoost's native TreeSHAP (pred_contribs), which is exact
    for the fitted trees and fast enough to explain every alert in the queue.
    """
    feature_names = np.array(get_feature_names(pipeline))
    X_t = get_shap_input(pipeline, X)

    booster  = pipeline.named_steps['model'].get_booster()
    contribs = booster.predict(xgb.DMatrix(X_t), pred_contribs=True)[:, :-1]   # drop bias term

    top = np.argsort(-contribs, axis=1)[:, :top_k]
    reasons = {}
    for k in range(top_k):
        idx    = top[:, k]
        values = X_t[np.arange(len(X_t)), idx]
        impact = contribs[np.arange(len(X_t)), idx]
        reasons[f'reason_{k + 1}'] = [
            f"{name} = {value:,.4g} (+{effect:.2f})" if effect > 0 else ''
            for name, value, effect in zip(feature_names[idx], values, impact)
        ]

    return pd.DataFrame(reasons, index=X.index)


def build_alert_queue(pipeline, X_test: pd.DataFrame, df_test: pd.DataFrame,
                      y_proba: np.ndarray, thresholds: dict,
                      filename: str = 'alert_queue_test_window.csv') -> pd.DataFrame:
    """
    Investigator work queue: every REVIEW / ESCALATE alert on the test window,
    ranked by risk score, with triggered TM scenarios and model reason codes.
    The ground-truth label is attached for evaluation only.
    """
    tiers   = triage_decision(y_proba, thresholds)
    alerted = tiers != 'CLEAR'

    queue = df_test.loc[alerted, [
        'transaction_id', 'ts', 'from_account_id', 'to_account_id', 'amount_paid_usd',
        'payment_format', 'payment_currency', 'customer_segment', 'rules_triggered'
    ]].copy()
    queue.insert(1, 'triage_tier', tiers[alerted])
    queue.insert(2, 'risk_score', y_proba[alerted].round(4))
    queue = queue.join(reason_codes(pipeline, X_test[alerted]))
    queue['ground_truth_is_laundering'] = df_test.loc[alerted, 'is_laundering']
    queue['ground_truth_typology']      = df_test.loc[alerted, 'eval_typology']

    queue = queue.sort_values(['triage_tier', 'risk_score'], ascending=[True, False])

    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    queue.to_csv(ALERTS_DIR / filename, index=False)
    print(f"Alert queue: {len(queue):,} alerts → reports/alerts/{filename}")

    return queue
