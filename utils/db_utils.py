
"""
utils/db_utils.py
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

import re
import threading
from contextlib import contextmanager
from pathlib import Path

import duckdb

from utils.file_utils import raise_fd_limit
from utils.scoring import compute_scores

DB_SQL_PATH = "sqls/all.sql"

FUNDAMENTAL_VIEWS = [
    ("quarterly_results",      "Quarterly"),
    ("profit_loss",            "P&L"),
    ("cash_flows",             "Cash Flow"),
    ("ratios",                 "Ratios"),
    ("shareholding_quarterly_metric", "Shareholding"),
    ("balance_sheet",          "Balance Sheet"),
]

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
    raise_fd_limit()               # multi-parquet pipelines can exceed macOS' 256 open files
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

from functools import lru_cache

DEFAULT_INDUSTRY_META: dict[str, str] = {"color": "#6b7280", "icon": "📦"}


@lru_cache(maxsize=8)
def load_color_codes(namespace: str = "industry") -> dict[str, dict[str, str]]:
    """Load color codes and icons for a given namespace from DuckDB `color_codes`."""
    query = """
        SELECT instrument, color_hexcode, icon
        FROM color_codes
        WHERE namespace = ?
    """
    try:
        conn = get_connection()
        df = conn.execute(query, [namespace]).df()
        mapping = {
            str(row["instrument"]).strip(): {
                "color": str(row["color_hexcode"]).strip(),
                "icon":  str(row["icon"]).strip(),
            }
            for _, row in df.iterrows()
        }
        if namespace == "industry":
            mapping.setdefault("Other", DEFAULT_INDUSTRY_META.copy())
        return mapping
    except Exception as exc:
        print(f"[db_utils] Failed to load color_codes (namespace={namespace!r}): {exc}")
        return {"Other": DEFAULT_INDUSTRY_META.copy()} if namespace == "industry" else {}


def load_defined_industry() -> dict[str, dict[str, str]]:
    """Return {industry: {'color': ..., 'icon': ...}} from `color_codes`."""
    return load_color_codes("industry")


def get_industry_meta(industry: str) -> dict[str, str]:
    """Return color + icon metadata for an industry, falling back to 'Other'."""
    defined = load_defined_industry()
    return defined.get(industry, defined.get("Other", DEFAULT_INDUSTRY_META))

import datetime
import pandas as pd
import datetime
import time
from typing import Callable
import pandas as pd

def _emit_sql_log(
        log_sql: Callable | None,
        label: str,
        sql: str,
        df: pd.DataFrame | None,
        elapsed_ms: float | None,
        error: str | None = None,
) -> None:
    if log_sql is not None:
        log_sql(label, sql, df, elapsed_ms, error)


def _price_history_table(is_benchmark: bool) -> str:
    """Resolve the source table name for benchmark vs. holding price history."""
    return "benchmark_prices" if is_benchmark else "ticker_prices_extended"


def fetch_price_history(
        nse_symbols: list[str],
        days: int,
        is_benchmark: bool = False,
        end_date: datetime.date | None = None,
        log_sql: Callable | None = None,
) -> pd.DataFrame:
    """Parameterized price-history lookup for holdings or benchmark series."""
    if not nse_symbols:
        return pd.DataFrame(columns=["date", "ticker", "close"])

    table_name = _price_history_table(is_benchmark)
    placeholders = ", ".join(["?"] * len(nse_symbols))
    params: list = [*nse_symbols, int(days)]

    end_clause = ""
    if end_date is not None:
        end_clause = "AND Date <= ?"
        params.append(end_date)

    sql = f"""
        SELECT
            Date       AS date,
            nse_symbol AS ticker,
            Close      AS close
        FROM {table_name}
        WHERE nse_symbol IN ({placeholders})
          AND Date >= CURRENT_DATE - (? * INTERVAL '1 day')
          {end_clause}
        ORDER BY nse_symbol, Date
    """
    label = (
            f"get_price_history → {table_name} ({len(nse_symbols)} symbols, {days} days"
            + (f", to {end_date}" if end_date else "")
            + ")"
    )
    t0 = time.perf_counter()
    try:
        conn = get_connection()
        df = conn.execute(sql, params).df()
        _emit_sql_log(log_sql, label, sql, df, (time.perf_counter() - t0) * 1000)
        return df
    except Exception as exc:
        _emit_sql_log(
            log_sql,
            f"{label} (FAILED)",
            sql,
            None,
            (time.perf_counter() - t0) * 1000,
            str(exc),
            )
        raise


def fetch_ticker_metadata(
        symbols: list[str],
        log_sql: Callable | None = None,
) -> pd.DataFrame:
    """Parameterized metadata lookup backed by the initialized view."""
    cleaned = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
    if not cleaned:
        return pd.DataFrame(
            columns=["matched_symbol", "industry", "market_cap", "market_cap_amt", "p_e", "dividend_yield"]
        )

    placeholders = ", ".join(["?"] * len(cleaned))
    sql = f"""
        SELECT
            nse_symbol AS matched_symbol,
            industry,
            market_cap,
            market_cap_amt,
            p_e,
            dividend_yield
        FROM v_ticker_metadata
        WHERE nse_symbol IN ({placeholders})
    """
    label = f"ticker metadata → v_ticker_metadata ({len(cleaned)} symbols)"
    t0 = time.perf_counter()
    try:
        conn = get_connection()
        df = conn.execute(sql, cleaned).df()
        _emit_sql_log(log_sql, label, sql, df, (time.perf_counter() - t0) * 1000)
        return df
    except Exception as exc:
        _emit_sql_log(
            log_sql,
            f"{label} (FAILED)",
            sql,
            None,
            (time.perf_counter() - t0) * 1000,
            str(exc),
            )
        raise

@lru_cache(maxsize=16)
def _load_csv_table_cached(resolved_path: str) -> pd.DataFrame:
    path = Path(resolved_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration CSV not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def load_csv_table(csv_path: str) -> pd.DataFrame:
    """Load a configuration CSV once and return a fresh DataFrame copy.

    Paths are resolved from the caller's working directory when relative.
    The cache is keyed by the resolved path; restart the app after editing CSVs.
    """
    path = str(Path(csv_path).expanduser().resolve())
    return _load_csv_table_cached(path).copy()



# ═══════════════════════════════════════════════════════════════════════════════
# Cursor helper, schema inspection and read-only execution  (SQL Lab, table_widget)
# ═══════════════════════════════════════════════════════════════════════════════

@contextmanager
def db_cursor():
    """Yield a private cursor on the shared database and close it on exit."""
    cur = get_connection()
    try:
        yield cur
    finally:
        cur.close()


def quote_ident(name: str) -> str:
    """Quote an SQL identifier, escaping embedded double quotes."""
    return '"' + str(name).replace('"', '""') + '"'


def list_tables() -> list[str]:
    """Names of all tables and views in the main schema, sorted."""
    with db_cursor() as cur:
        rows = cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_catalog = current_database() AND table_schema = 'main' "
            "ORDER BY table_name"
        ).fetchall()
    return [r[0] for r in rows]


def _require_table(cur, table: str) -> None:
    """Raise ValueError unless `table` is an existing table/view (guards identifier use)."""
    found = cur.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_catalog = current_database() AND table_schema = 'main' "
        "AND table_name = ?",
        [table],
    ).fetchone()
    if found is None:
        raise ValueError(f"Unknown table or view: {table!r}")


def ddl_for_table(table: str) -> str:
    """A `CREATE TABLE` sketch (column names + types) for a table or view."""
    with db_cursor() as cur:
        _require_table(cur, table)
        cols = cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_catalog = current_database() AND table_schema = 'main' "
            "AND table_name = ? ORDER BY ordinal_position",
            [table],
        ).fetchall()
    col_defs = ",\n    ".join(f"{quote_ident(c)} {t}" for c, t in cols)
    return f"CREATE TABLE {quote_ident(table)} (\n    {col_defs}\n);"


def sample_rows(table: str, n: int = 5) -> pd.DataFrame:
    """First `n` rows (1-1000) of a table or view."""
    n = max(1, min(int(n), 1000))
    with db_cursor() as cur:
        _require_table(cur, table)
        return cur.execute(f"SELECT * FROM {quote_ident(table)} LIMIT {n}").df()


class ReadOnlyQueryError(ValueError):
    """Raised when a query is rejected by the read-only guard."""


_READ_ONLY_KEYWORDS = frozenset(
    {"select", "with", "from", "describe", "show", "summarize", "explain", "table"}
)

# Statement types that change state; checked via duckdb's `extract_statements` so that
# `WITH ... DELETE` and multi-statement input are caught, not just the first keyword.
_WRITE_STATEMENT_TYPES = frozenset({
    "INSERT", "UPDATE", "DELETE", "CREATE", "CREATE_FUNC", "ALTER", "DROP", "COPY",
    "COPY_DATABASE", "ATTACH", "DETACH", "SET", "VARIABLE_SET", "LOAD", "EXPORT",
    "VACUUM", "TRANSACTION", "PREPARE", "EXECUTE", "CALL", "PRAGMA", "EXTENSION",
    "ANALYZE", "MERGE_INTO",
})


def _strip_leading_comments(sql: str) -> str:
    s = sql.lstrip()
    while True:
        if s.startswith("--"):
            nl = s.find("\n")
            s = "" if nl == -1 else s[nl + 1:].lstrip()
        elif s.startswith("/*"):
            end = s.find("*/")
            s = "" if end == -1 else s[end + 2:].lstrip()
        else:
            return s


def _assert_single_read_statement(cur, sql: str) -> None:
    extract = getattr(cur, "extract_statements", None)
    if extract is None:                         # older duckdb: conservative fallback
        if ";" in sql.strip().rstrip(";"):
            raise ReadOnlyQueryError("Run one statement at a time.")
        return
    statements = extract(sql)
    if len(statements) != 1:
        raise ReadOnlyQueryError("Run exactly one statement at a time.")
    stype = getattr(statements[0], "type", None)
    if stype is not None:
        name = str(getattr(stype, "name", stype)).split(".")[-1].upper()
        if name in _WRITE_STATEMENT_TYPES:
            raise ReadOnlyQueryError(f"{name} statements are not allowed - read-only queries only.")


def run_readonly_query(sql: str, params: list | tuple | None = None) -> pd.DataFrame:
    """
    Run one read-only statement on a private cursor and return a DataFrame.

    Allowed: SELECT / WITH / FROM / TABLE / DESCRIBE / SHOW / SUMMARIZE / EXPLAIN.
    Raises ReadOnlyQueryError for anything else; DuckDB errors for bad SQL propagate.
    Guards the shared database against writes; it does not sandbox file access from
    table functions such as read_csv('/path').
    """
    if not sql or not sql.strip():
        raise ReadOnlyQueryError("Enter a query to run.")
    match = re.match(r"[\s(]*([A-Za-z_]+)", _strip_leading_comments(sql))
    first = match.group(1).lower() if match else ""
    if first not in _READ_ONLY_KEYWORDS:
        raise ReadOnlyQueryError(
            "Only read-only queries are allowed (SELECT / WITH / DESCRIBE / SHOW ...) "
            "- the database is shared.")
    with db_cursor() as cur:
        _assert_single_read_statement(cur, sql)
        result = cur.execute(sql, list(params)) if params else cur.execute(sql)
        return result.df()


def run_sql(sql: str) -> tuple[pd.DataFrame | None, str | None]:
    """UI-friendly wrapper: returns (dataframe, None) or (None, error_message)."""
    try:
        return run_readonly_query(sql), None
    except Exception as exc:
        return None, str(exc)


def execute_sql(sql: str, params: list | tuple | None = None) -> None:
    """Run a single write/DDL statement (e.g. UPDATE) on a private cursor; no result."""
    with db_cursor() as cur:
        cur.execute(sql, list(params)) if params else cur.execute(sql)


def run_sql_file(path: str) -> None:
    """
    Re-run one .sql script (following `.read` includes) against the shared database,
    e.g. to refresh a single view without rebuilding everything. Relative paths and
    `.read` targets resolve from the working directory, like `sqls/all.sql`.
    """
    with db_cursor() as cur:
        _run_sql_script(cur, path)


# ═══════════════════════════════════════════════════════════════════════════════
# Scored ticker matrix  (one_ticker_gates, ticker deep-dive)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_gate_scores() -> pd.DataFrame:
    """Raw `gate_score` view, one row per ticker (uncached; callers cache if they want)."""
    return run_readonly_query("SELECT * FROM gate_score ORDER BY ticker")


def load_ticker_scores() -> tuple[pd.DataFrame, object, str]:
    """
    One-stop loader: `gate_score` view -> scored matrix (Quality / Valuation / Timing /
    Technical, plus every gate flag).

    Returns (df_matrix, conn, env_status).

    `conn` is a cursor on the shared all.sql database and `env_status` is always
    "REAL"; both are kept only so existing 3-tuple unpacking keeps working.
    """
    conn = get_connection()
    df_raw = fetch_gate_scores()
    if df_raw.empty:
        return pd.DataFrame(), conn, "REAL"
    return compute_scores(df_raw), conn, "REAL"