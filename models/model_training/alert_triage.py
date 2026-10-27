"""
Alert triage policy — maps a laundering risk score to an investigation decision.

    CLEAR     -  no alert raised, transaction auto-cleared
    REVIEW    -  alert routed to a Level-1 analyst for review
    ESCALATE  -  alert routed to Level-2 investigation (potential SAR filing)

Thresholds are not hard-coded: they are calibrated on the validation window
(never on the test window) against two operating targets:
    ESCALATE  -  at least ESCALATE_MIN_PRECISION of escalated alerts are true laundering
    REVIEW    -  REVIEW + ESCALATE together capture REVIEW_TARGET_RECALL of laundering
"""
import json

import numpy as np
from sklearn.metrics import precision_recall_curve

from src.config import MODEL_DIR


# example of a triage policy
ESCALATE_MIN_PRECISION = 0.80   # L2 investigations are costly: 4 in 5 escalations should be genuine
REVIEW_TARGET_RECALL   = 0.80   # the alert queue as a whole should catch 4 in 5 laundering payments

TIERS = ['CLEAR', 'REVIEW', 'ESCALATE']
THRESHOLDS_PATH = MODEL_DIR / 'alert_thresholds.json'


def calibrate_thresholds(y_val, p_val,
                         escalate_precision: float = ESCALATE_MIN_PRECISION,
                         review_recall: float = REVIEW_TARGET_RECALL) -> dict:
    """
    Derives the REVIEW and ESCALATE score thresholds from the validation PR curve.
    """
    precision, recall, thresholds = precision_recall_curve(y_val, p_val)
    precision, recall = precision[:-1], recall[:-1]          # align with thresholds

    # lowest score at which escalations reach the precision target
    meets_precision = np.flatnonzero(precision >= escalate_precision)
    escalate = thresholds[meets_precision[0]] if len(meets_precision) else thresholds[-1]

    # highest score at which the alert queue still reaches the recall target
    meets_recall = np.flatnonzero(recall >= review_recall)
    review = thresholds[meets_recall[-1]] if len(meets_recall) else thresholds[0]
    review = min(review, escalate)

    calibrated = {'review': float(review), 'escalate': float(escalate),
                  'escalate_min_precision': escalate_precision,
                  'review_target_recall': review_recall}
    print(f"Calibrated on validation — REVIEW ≥ {review:.4f} | ESCALATE ≥ {escalate:.4f}")

    return calibrated


def triage_decision(probability, thresholds: dict) -> np.ndarray:
    """
    Maps laundering risk scores to triage tiers (vectorised).
    """
    probability = np.asarray(probability)
    return np.select(
        [probability >= thresholds['escalate'], probability >= thresholds['review']],
        ['ESCALATE', 'REVIEW'],
        default='CLEAR'
    )


def save_thresholds(thresholds: dict, path=THRESHOLDS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(thresholds, indent=2))


def load_thresholds(path=THRESHOLDS_PATH) -> dict:
    return json.loads(path.read_text())
