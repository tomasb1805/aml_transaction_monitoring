"""
DuckDB connection helper shared by every pipeline stage.
"""
import duckdb

from src.config import DUCKDB_PATH


def connect(db_path=DUCKDB_PATH, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """
    Opens the project database with console progress bars disabled.
    """
    con = duckdb.connect(str(db_path), read_only=read_only)
    con.execute("SET enable_progress_bar = false")

    return con
