"""
Class-imbalance ablation.

Laundering is ~0.08% of training transactions. Four ways of handling that imbalance are
compared on equal terms: each is tuned over the same hyperparameter candidates
(same RandomizedSearchCV seed and n_iter), fitted on the training window and scored
on the validation window.

    undersample_20pct_balanced  -  20% of legitimate txns, scale_pos_weight = neg/pos (≈265)
    undersample_20pct_none      -  20% of legitimate txns, no re-weighting
    full_data_none              -  all training txns, no re-weighting
    full_data_sqrt              -  all training txns, scale_pos_weight = sqrt(neg/pos) (≈36)

Selection rule (fixed before running): highest validation Average Precision.
Test-window metrics are recorded for information only and play no part in the choice.

    python -m models.model_training.imbalance_ablation --n-iter 15
"""
import argparse
import json
import time

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import BASE_DIR
from models.model_training.build_model_dataset import build_dataset, undersample_negatives
from models.model_training.train_xgb_model import build_pipeline, hyper_tuning, positive_class_weight


CONFIGS = {
    'undersample_20pct_balanced': (0.20, 'balanced'),
    'undersample_20pct_none':     (0.20, 'none'),
    'full_data_none':             (1.00, 'none'),
    'full_data_sqrt':             (1.00, 'sqrt'),
}

OUTPUT_PATH = BASE_DIR / 'reports' / 'tables' / 'imbalance_ablation.csv'


def run_ablation(n_iter: int = 15) -> pd.DataFrame:
    """
    Tunes and scores every imbalance configuration; returns one row per configuration.
    """
    data = build_dataset(export=False, neg_sample_rate=1.0)
    rows = []

    for name, (rate, weighting) in CONFIGS.items():
        print(f"\n=== {name} ===")
        start = time.perf_counter()

        X_train, y_train = undersample_negatives(data.X_train, data.y_train, rate=rate)
        pipeline = build_pipeline(y_train, data.preprocessor, class_weighting=weighting)
        tuned, search = hyper_tuning(pipeline, X_train, y_train, data.X_val, data.y_val, n_iter=n_iter)

        p_val  = tuned.predict_proba(data.X_val)[:, 1]
        p_test = tuned.predict_proba(data.X_test)[:, 1]

        rows.append({
            'configuration':       name,
            'neg_sample_rate':     rate,
            'class_weighting':     weighting,
            'scale_pos_weight':    round(positive_class_weight(y_train, weighting), 1),
            'train_rows':          len(y_train),
            'val_average_precision': average_precision_score(data.y_val, p_val),
            'val_roc_auc':         roc_auc_score(data.y_val, p_val),
            'test_average_precision': average_precision_score(data.y_test, p_test),
            'test_roc_auc':        roc_auc_score(data.y_test, p_test),
            'minutes':             (time.perf_counter() - start) / 60,
            'best_params':         json.dumps({k.replace('model__', ''): v for k, v in search.best_params_.items()}),
        })
        print(pd.DataFrame(rows).drop(columns='best_params').tail(1).to_string(index=False))

    results = pd.DataFrame(rows).sort_values('val_average_precision', ascending=False)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{results.drop(columns='best_params').to_string(index=False, float_format=lambda v: f'{v:.4f}')}")
    print(f"\nSelected (highest validation AP): {results.iloc[0]['configuration']}")
    print(f"Saved → reports/tables/{OUTPUT_PATH.name}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-iter', type=int, default=15,
                        help='hyperparameter candidates per configuration (default: 15)')
    args = parser.parse_args()

    run_ablation(n_iter=args.n_iter)
