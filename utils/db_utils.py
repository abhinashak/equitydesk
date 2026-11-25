"""
util/connection.py
──────────────────
Single place that builds the DuckDB connection and its views from `sqls/all.sql`.

    from util.connection import get_connection
    conn = get_connection()
    df = conn.execute("SELECT ...").df()

* `initialize_database()` builds a fresh in-memory database by running DB_SQL_PATH
  (following `.read other.sql` includes).
* `get_connection()` initializes once per process and hands back a cursor on that shared
  database, so every caller sees the views without re-running the script, and each
  caller / Streamlit session gets its own cursor (a DuckDB connection is not thread-safe).
* `reset_connection()` drops the shared database; the next `get_connection()` rebuilds it
  (use after editing the .sql files).
"""

import threading
from pathlib import Path

import duckdb

DB_SQL_PATH = "sqls/all.sql"

_lock = threading.Lock()
_connection = None            # shared in-memory database, created on first use


def _run_sql_script(connection, path, _seen=None):
    """Runs a .sql file, following `.read other.sql` includes."""
    path = Path(path).expanduser().resolve()
    _seen = _seen or set()
    if path in _seen:
        return
    _seen = _seen | {path}
    buf = []

    def flush():
        sql = "\n".join(buf).strip()
        buf.clear()
        if sql:
            connection.execute(sql)

    for line in path.read_text().splitlines():
        stripped = line.strip()
        at_boundary = not buf or "\n".join(buf).rstrip().endswith(";")
        if stripped.startswith(".") and at_boundary:
            flush()
            parts = stripped.split(maxsplit=1)
            if parts[0] == ".read" and len(parts) == 2:
                target = Path(parts[1].strip().strip("'\"")).expanduser()
                _run_sql_script(connection, target, _seen)
        else:
            buf.append(line)
    flush()


def initialize_database(sql_path: str = DB_SQL_PATH):
    """Create a new in-memory DuckDB database and run `sql_path` against it."""
    if not Path(sql_path).exists():
        raise FileNotFoundError(
            f"'{sql_path}' not found (looked relative to {Path.cwd()}). "
            "Start the app from the project root or fix DB_SQL_PATH.")
    con = duckdb.connect()
    _run_sql_script(con, sql_path)
    return con


def get_connection():
    """Cursor on the shared, initialized database (the views from DB_SQL_PATH are ready to query)."""
    global _connection
    with _lock:
        if _connection is None:
            _connection = initialize_database()
        return _connection.cursor()


def reset_connection() -> None:
    """Close the shared database so the next get_connection() re-runs DB_SQL_PATH."""
    global _connection
    with _lock:
        if _connection is not None:
            _connection.close()
            _connection = None
