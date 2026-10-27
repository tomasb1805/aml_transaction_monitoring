"""
Downloads the IBM Transactions for Anti-Money Laundering (AML) dataset from Kaggle.

The full dataset ships six variants (HI/LI x Small/Medium/Large, ~41 GB uncompressed).
Only the files of the configured variant (default: HI-Small, ~0.5 GB) are downloaded,
straight into data/raw/ instead of the kagglehub cache.
"""
import kagglehub

from src.config import KAGGLE_HANDLE, DATASET_VARIANT, DATASET_FILES, RAW_DIR, raw_file


def download_dataset(force: bool = False) -> list:
    """
    Downloads the transactions, accounts and laundering-pattern files
    of the selected dataset variant. Files already on disk are skipped.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    for kind in DATASET_FILES:
        target = raw_file(kind)
        if target.exists() and not force:
            print(f"Already downloaded: {target.name}")
        else:
            kagglehub.dataset_download(
                KAGGLE_HANDLE,
                path=target.name,
                output_dir=str(RAW_DIR),
                force_download=force,
            )
            print(f"Downloaded: {target.name}")
        paths.append(target)

    return paths


if __name__ == "__main__":
    files = download_dataset()
    print(f"Path to dataset files ({DATASET_VARIANT}):", RAW_DIR)
