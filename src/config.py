"""
Project-wide configuration: file paths, dataset selection and database location.

Every value can be overridden through the `.env` file in the project root,
so the pipeline runs out-of-the-box with the defaults below.
"""
import os
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)


def _path(env_key: str, default: Path) -> Path:
    return Path(os.getenv(env_key, default)).expanduser()


# data source
KAGGLE_HANDLE   = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
DATASET_VARIANT = os.getenv("AML_DATASET_VARIANT", "HI-Small")   # HI/LI x Small/Medium/Large
DATASET_FILES   = ("Trans.csv", "accounts.csv", "Patterns.txt")

# directories
RAW_DIR       = _path("RAW_DATA_DIR",       BASE_DIR / "data" / "raw")
PROCESSED_DIR = _path("PROCESSED_DATA_DIR", BASE_DIR / "data" / "processed")
DUCKDB_PATH   = _path("DUCKDB_PATH",        BASE_DIR / "data" / "aml.duckdb")
SQL_DIR       = BASE_DIR / "scripts" / "SQL_scripts"
MODEL_DIR     = BASE_DIR / "models" / "artifacts"
PLOTS_DIR     = BASE_DIR / "reports" / "output_plots"
ALERTS_DIR    = BASE_DIR / "reports" / "alerts"

MODEL_TABLE_PATH = PROCESSED_DIR / "modeling_table.parquet"
MODEL_PATH       = MODEL_DIR / "xgb_aml_pipeline.pkl"


def raw_file(kind: str) -> Path:
    """
    Path of a raw dataset file for the selected variant, e.g. raw_file('Trans.csv').
    """
    return RAW_DIR / f"{DATASET_VARIANT}_{kind}"
