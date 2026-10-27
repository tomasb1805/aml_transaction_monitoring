"""
Master entrypoint — runs the AML transaction-monitoring pipeline end to end.

    python run_pipeline.py                # full run: download → ETL → features → rules → model
    python run_pipeline.py --skip-data    # reuse the DuckDB database, rerun the modelling stages
    python run_pipeline.py --n-iter 10    # quicker hyperparameter search
"""
import argparse
import json

from src.config import BASE_DIR
from src.kaggle_data_client import download_dataset
from scripts import etl, statistical_features_derivation, transaction_monitoring_rules

from models.model_training.build_model_dataset import build_dataset
from models.model_training.train_xgb_model import (
    build_pipeline,
    hyper_tuning,
    features_selection,
    save_model
)
from models.model_training.alert_triage import (
    calibrate_thresholds,
    triage_decision,
    save_thresholds
)
from models.model_training.evaluate_model import (
    evaluate_model,
    operating_point,
    plot_roc_auc,
    plot_precision_recall,
    rule_performance,
    triage_summary,
    compare_rules_vs_model,
    typology_detection,
    plot_typology_detection
)
from models.model_training.explain_model import (
    build_shap_explainer,
    explanation_sample,
    compute_shap_values,
    plot_shap_summary,
    plot_shap_dependence,
    plot_example_transactions,
    build_alert_queue
)


TABLES_DIR = BASE_DIR / 'reports' / 'tables'


def run_data_stages() -> None:
    """
    Steps 1-4: ingestion, ETL, behavioural features and the TM rules engine.
    """
    download_dataset()
    etl.run_pipeline()
    statistical_features_derivation.pipeline()
    transaction_monitoring_rules.run_rules()


def save_table(table, name: str) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / f'{name}.csv')
    print(f"\n{name}\n{table.to_string(float_format=lambda v: f'{v:,.4f}')}")


def run_model_stages(n_iter: int) -> dict:
    """
    Steps 5-9: modelling table, training, threshold calibration, evaluation, explainability.
    """
    data = build_dataset(export=True)

    # Pass the outputs of data preparing step directly into the training functions
    model_pipeline = build_pipeline(data.y_train, data.preprocessor)
    tuned_pipeline, search = hyper_tuning(model_pipeline, data.X_train, data.y_train,
                                          data.X_val, data.y_val, n_iter=n_iter)
    features_selection(tuned_pipeline)
    save_model(tuned_pipeline)

    # alert thresholds are calibrated on the validation window only
    p_val, val_metrics = evaluate_model(tuned_pipeline, data.X_val, data.y_val, 'Validation')
    thresholds = calibrate_thresholds(data.y_val, p_val)
    save_thresholds(thresholds)

    # tuned_pipeline flows into the out-of-time test evaluation
    y_proba, test_metrics = evaluate_model(tuned_pipeline, data.X_test, data.y_test, 'Test')
    tiers = triage_decision(y_proba, thresholds)
    rules_point = operating_point(data.df_test['rule_hit_count'] > 0, data.y_test)

    plot_roc_auc(data.y_test, y_proba, rules_point)
    plot_precision_recall(data.y_test, y_proba, rules_point, thresholds)

    comparison = compare_rules_vs_model(data.df_test, tiers)
    typology   = typology_detection(data.df_test, tiers)
    save_table(rule_performance(data.df_test), 'rule_performance_test')
    save_table(triage_summary(data.df_test, tiers), 'triage_summary_test')
    save_table(comparison, 'rules_vs_model_test')
    save_table(typology, 'typology_detection_test')
    plot_typology_detection(typology)

    # Both tuned_pipeline and X_train/X_test flow in here
    explainer   = build_shap_explainer(tuned_pipeline, data.X_train)
    sample_idx  = explanation_sample(tiers)
    X_sample    = data.X_test.iloc[sample_idx]
    df_sample   = data.df_test.iloc[sample_idx]
    shap_values = compute_shap_values(explainer, tuned_pipeline, X_sample)

    plot_shap_summary(shap_values)
    plot_shap_dependence(shap_values, tuned_pipeline, X_sample, top_n=5)
    plot_example_transactions(shap_values, df_sample, y_proba[sample_idx], thresholds)
    build_alert_queue(tuned_pipeline, data.X_test, data.df_test, y_proba, thresholds)

    metrics = {
        'validation':       val_metrics,
        'test':             test_metrics,
        'best_params':      search.best_params_,
        'thresholds':       thresholds,
        'rules_vs_model':   comparison.to_dict(orient='index'),
        'n_transactions':   {'train': int(len(data.X_train)),
                             'validation': int(len(data.X_val)), 'test': int(len(data.X_test))},
    }
    (BASE_DIR / 'reports' / 'metrics.json').write_text(json.dumps(metrics, indent=2, default=float))

    return metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--skip-data', action='store_true',
                        help='skip download, ETL, feature derivation and rules (reuse data/aml.duckdb)')
    parser.add_argument('--n-iter', type=int, default=30,
                        help='RandomizedSearchCV iterations (default: 30)')
    args = parser.parse_args()

    if not args.skip_data:
        run_data_stages()

    run_model_stages(n_iter=args.n_iter)
    print("\n=== AML transaction-monitoring pipeline complete ===")
