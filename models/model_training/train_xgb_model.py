# Python libraries
import time

import joblib
import numpy as np
import pandas as pd

# Machine learning + eval libraries
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import MODEL_PATH


# positive-class weight scheme passed to XGBoost's scale_pos_weight:
#   'balanced'  -  negatives / positives (both classes carry equal total weight)
#   'sqrt'      -  square root of that ratio (partial re-weighting)
#   'none'      -  1.0 (no re-weighting)
# 'none' won the imbalance ablation (imbalance_ablation.py): re-weighting to a balanced
# prior lowered validation Average Precision from 0.598 to 0.563 (sqrt) and 0.448 (balanced, 20% sample).
CLASS_WEIGHTING = 'none'


def positive_class_weight(y_train, scheme: str = CLASS_WEIGHTING) -> float:
    """
    scale_pos_weight for the given scheme, computed on the training data actually fitted.
    """
    ratio = (len(y_train) - y_train.sum()) / y_train.sum()
    weights = {'balanced': ratio, 'sqrt': np.sqrt(ratio), 'none': 1.0}

    return float(weights[scheme])


def features_selection(pipeline, top_n: int = 10) -> pd.DataFrame:
    """
    Ranks the transformed features by XGBoost gain importance.
    """
    feature_names = (
        pipeline.named_steps['preprocessing']
        .get_feature_names_out()
    )
    importances = pipeline.named_steps['model'].feature_importances_

    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False)
    print(fi_df.head(top_n).to_string(index=False))

    return fi_df


def build_pipeline(y_train, preprocessing, class_weighting: str = CLASS_WEIGHTING):
    """
    Constructs the classification pipeline.
    """
    scale_pos_weight = positive_class_weight(y_train, class_weighting)
    print(f"scale_pos_weight ({class_weighting}): {scale_pos_weight:.1f}")

    pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('model', XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            eval_metric='aucpr',
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=11
        ))
    ])

    return pipeline


def hyper_tuning(pipeline, X_train, y_train, X_val, y_val, n_iter: int = 20):
    """
    Process hyperparameter tuning using RandomizedSearchCV.

    Folds are chronological rather than shuffled: every candidate is fitted on the
    training window and scored on the later validation window (PredefinedSplit).
    The search maximises Average Precision (area under the PR curve), the standard
    objective when the positive class is ~0.1% of transactions.
    The best candidate is then refitted on the training window only, so the
    validation window stays unseen for alert-threshold calibration.
    """
    param_grid = {
        'model__n_estimators':     [100, 200, 400],
        'model__max_depth':        [3, 4, 5, 6],
        'model__learning_rate':    [0.01, 0.05, 0.1],
        'model__subsample':        [0.6, 0.8, 1.0],
        'model__colsample_bytree': [0.6, 0.8, 1.0],
        'model__min_child_weight': [1, 3, 5],
        'model__reg_alpha':        [0, 0.1, 1.0],
        'model__reg_lambda':       [1.0, 2.0, 5.0]
    }

    X_search = pd.concat([X_train, X_val])
    y_search = pd.concat([y_train, y_val])
    test_fold = np.r_[np.full(len(X_train), -1), np.zeros(len(X_val))]

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='average_precision',
        cv=PredefinedSplit(test_fold),
        random_state=11,
        n_jobs=1,            # XGBoost already uses every core per fit
        refit=False,
        verbose=1
    )

    start = time.perf_counter()
    search.fit(X_search, y_search)

    print(f"Search time: {(time.perf_counter() - start) / 60:.1f} min")
    print(f"Validation Best Average Precision: {search.best_score_:.4f}")
    print(f"Optimal Parameters: {search.best_params_}")

    tuned_pipeline = pipeline.set_params(**search.best_params_)
    tuned_pipeline.fit(X_train, y_train)

    return tuned_pipeline, search


def save_model(pipeline, path=MODEL_PATH) -> None:
    """
    Serialises the fitted preprocessing + model pipeline.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Model saved → {path}")
