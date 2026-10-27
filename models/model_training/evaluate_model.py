# Python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# metrics libraries
from sklearn.metrics import (roc_auc_score, auc, roc_curve,
                             precision_recall_curve, average_precision_score)

from models.model_training.alert_triage import TIERS
from models.model_training.plot_style import (apply_style, save_figure, SURFACE, TEXT_MUTED,
                                              TEXT_SECONDARY, MODEL_COLOUR, RULES_COLOUR,
                                              TIER_COLOURS)
from scripts.transaction_monitoring_rules import TM_SCENARIOS


TARGET = 'is_laundering'


def evaluate_model(fitted_pipeline, X, y, label: str = 'Test') -> tuple:
    """
    Computes ROC-AUC and Average Precision using the optimized estimator.
    """
    y_pred_proba = fitted_pipeline.predict_proba(X)[:, 1]
    metrics = {
        'roc_auc':           roc_auc_score(y, y_pred_proba),
        'average_precision': average_precision_score(y, y_pred_proba),
        'baseline':          y.mean(),
    }
    metrics['lift'] = metrics['average_precision'] / metrics['baseline']

    print(f"{label:<10} ROC-AUC: {metrics['roc_auc']:.4f} | Average Precision: "
          f"{metrics['average_precision']:.4f} | baseline {metrics['baseline']:.4%} | "
          f"lift {metrics['lift']:.0f}x")

    return y_pred_proba, metrics


def operating_point(alert, y) -> dict:
    """
    Confusion-matrix summary of a binary alerting decision.
    """
    alert, y = np.asarray(alert, dtype=bool), np.asarray(y) == 1
    tp, fp = (alert & y).sum(), (alert & ~y).sum()

    return {
        'alerts':          int(alert.sum()),
        'alert_rate':      alert.mean(),
        'true_positives':  int(tp),
        'false_positives': int(fp),
        'precision':       tp / max(alert.sum(), 1),
        'recall':          tp / y.sum(),
        'fpr':             fp / (~y).sum(),
    }


def rule_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Alert volume, precision and recall of every TM scenario on the given window.
    """
    rows = {code: operating_point(df[code] == 1, df[TARGET]) for code in TM_SCENARIOS}
    rows['ANY_SCENARIO'] = operating_point(df['rule_hit_count'] > 0, df[TARGET])

    table = pd.DataFrame(rows).T
    table.insert(0, 'description', [TM_SCENARIOS[c][0] for c in TM_SCENARIOS] + ['Alert if any scenario fires'])

    return table


def triage_summary(df: pd.DataFrame, tiers: np.ndarray) -> pd.DataFrame:
    """
    Volume and laundering capture per triage tier.
    """
    summary = (
        pd.DataFrame({'tier': tiers, TARGET: df[TARGET].to_numpy()})
        .groupby('tier')[TARGET].agg(transactions='size', laundering='sum')
        .reindex(TIERS)
        .fillna(0)
    )
    summary['laundering_rate']    = summary['laundering'] / summary['transactions'].clip(lower=1)
    summary['share_of_laundering'] = summary['laundering'] / summary['laundering'].sum()

    return summary


def compare_rules_vs_model(df: pd.DataFrame, tiers: np.ndarray) -> pd.DataFrame:
    """
    Head-to-head of the rules engine and the model-driven alert queue.
    """
    y = df[TARGET]
    return pd.DataFrame({
        'TM rules (any scenario)':      operating_point(df['rule_hit_count'] > 0, y),
        'Model alerts (REVIEW + ESCALATE)': operating_point(tiers != 'CLEAR', y),
        'Model ESCALATE only':          operating_point(tiers == 'ESCALATE', y),
    }).T


def typology_detection(df: pd.DataFrame, tiers: np.ndarray) -> pd.DataFrame:
    """
    Share of laundering payments caught per typology, model alert queue vs rules engine.
    Laundering not listed in the Patterns file is reported as UNATTRIBUTED.
    """
    laundering = df[TARGET].to_numpy() == 1
    frame = pd.DataFrame({
        'typology':      df.loc[laundering, 'eval_typology'].fillna('UNATTRIBUTED').to_numpy(),
        'model_alerted': (tiers[laundering] != 'CLEAR'),
        'rules_alerted': (df.loc[laundering, 'rule_hit_count'] > 0).to_numpy(),
    })

    return (
        frame.groupby('typology')
        .agg(laundering_txns=('model_alerted', 'size'),
             model_recall=('model_alerted', 'mean'),
             rules_recall=('rules_alerted', 'mean'))
        .sort_values('model_recall')
    )


def plot_roc_auc(y_test, y_pred_proba, rules_point: dict):
    """
    Plots the ROC curve with the rules engine's operating point for reference.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    ax.plot([0, 1], [0, 1], color=TEXT_MUTED, linewidth=1, label='Random guess')
    ax.plot(fpr, tpr, color=MODEL_COLOUR, label=f'XGBoost (AUC = {roc_auc:.3f})')
    ax.scatter(rules_point['fpr'], rules_point['recall'], s=70, color=RULES_COLOUR,
               edgecolor=SURFACE, linewidth=2, zorder=3, label='TM rules (any scenario)')

    ax.set_xlabel('False Positive Rate  (legitimate payments alerted)')
    ax.set_ylabel('True Positive Rate  (laundering payments alerted)')
    ax.set_title('ROC Curve — XGBoost Laundering Detection (test window)')
    ax.legend(loc='lower right')
    save_figure(fig, 'ROC-AUC-curve.png')


def plot_precision_recall(y_test, y_pred_proba, rules_point: dict, thresholds: dict):
    """
    Plots the Precision-Recall curve, the rules engine's operating point and
    where the calibrated triage thresholds land on the test window.
    """
    apply_style()
    avg_prec = average_precision_score(y_test, y_pred_proba)
    baseline = y_test.mean()
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(y=baseline, color=TEXT_MUTED, linewidth=1,
               label=f'Random classifier (baseline = {baseline:.2%})')
    ax.plot(recall, precision, color=MODEL_COLOUR, label=f'XGBoost (AP = {avg_prec:.3f})')
    ax.scatter(rules_point['recall'], rules_point['precision'], s=70, color=RULES_COLOUR,
               edgecolor=SURFACE, linewidth=2, zorder=3, label='TM rules (any scenario)')

    for tier, key in [('REVIEW', 'review'), ('ESCALATE', 'escalate')]:
        point = operating_point(y_pred_proba >= thresholds[key], y_test)
        ax.scatter(point['recall'], point['precision'], s=70, marker='D', color=TIER_COLOURS[tier],
                   edgecolor=SURFACE, linewidth=2, zorder=4)
        offset, align = ((-12, -30), 'right') if tier == 'ESCALATE' else ((30, 48), 'center')
        ax.annotate(f"{tier} threshold\n{point['precision']:.0%} precision · {point['recall']:.0%} recall",
                    (point['recall'], point['precision']), textcoords='offset points',
                    xytext=offset, ha=align, fontsize=8.5, color=TEXT_SECONDARY)

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Recall  (share of laundering payments alerted)')
    ax.set_ylabel('Precision  (share of alerts that are laundering)')
    ax.set_title('Precision-Recall Curve — XGBoost Laundering Detection (test window)')
    ax.legend(loc='upper right')
    save_figure(fig, 'Precision-Recall-curve.png')

    print(f"Average Precision (AP): {avg_prec:.4f}")
    return avg_prec


def plot_typology_detection(table: pd.DataFrame):
    """
    Horizontal grouped bars: share of laundering caught per typology, model vs rules.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    y = np.arange(len(table))
    height = 0.38
    ax.barh(y + height / 2, table['model_recall'], height=height - 0.03,
            color=MODEL_COLOUR, label='Model alert queue (REVIEW + ESCALATE)')
    ax.barh(y - height / 2, table['rules_recall'], height=height - 0.03,
            color=RULES_COLOUR, label='TM rules (any scenario)')

    for yi, value in zip(y, table['model_recall']):
        ax.text(value + 0.01, yi + height / 2, f'{value:.0%}', va='center',
                fontsize=8.5, color=TEXT_SECONDARY)

    ax.set_yticks(y, [f"{t}  (n={n})" for t, n in zip(table.index, table['laundering_txns'])])
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Share of laundering payments alerted (recall)')
    ax.set_title('Detection Rate by Laundering Typology (test window)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    save_figure(fig, 'Typology-detection.png')
