"""
schema_dump.py  –  Dump a DuckDB database as an LLM-ready schema reference.

For every table and view it writes:
  * the CREATE statement and row count
  * per column: type, distinct count, min/max (numeric & date/time), NULL count
      - fewer than MAX_DISTINCT (48) distinct values -> ENUM(...) with ALL of them
      - MAX_DISTINCT or more                         -> only 3 sample values
  * 3 sample rows for the table

The database is initialised by the SQL pipeline in page_sql_lab.py (all scripts
live in the sqls/ folder), so views such as abinash_screen are included.

The output is valid SQL (everything except the DDL is a comment), so it can be
saved as sqls/schema.sql and pasted straight into a prompt for SQL generation.

Usage (run from the project root; writes <BASE>/sqls/schema.sql)
    python schema_dump.py
    python schema_dump.py -o sqls/schema.sql --lab ui/pages/page_sql_lab.py

In code
    from schema_dump import dump_schema
    dump_schema(con, "sqls/schema.sql")
"""
from __future__ import annotations

import argparse
import datetime
import decimal
import importlib.util
import os
import re
import sys

import duckdb

MAX_DISTINCT = 48     # list every value when a column has fewer than this many
SAMPLE_ROWS  = 3      # sample rows per table
SAMPLE_VALUES = 3     # sample values per high-cardinality column
MAX_STR      = 40     # truncate long strings in samples / value lists

_NUMERIC  = re.compile(r"^(U?(TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT)|FLOAT|DOUBLE|REAL|DECIMAL)", re.I)
_TEMPORAL = re.compile(r"^(DATE|TIME|TIMESTAMP)", re.I)
_NESTED   = re.compile(r"(\[|^STRUCT|^MAP|^UNION|^BLOB)", re.I)

HEADER = """\
-- DuckDB schema reference for SQL generation (dialect: DuckDB).
-- * ENUM(...) lists are COMPLETE: the column has fewer than {n} distinct non-NULL values,
--   so use these exact spellings/casing in filters.
-- * "samples:" are just {v} example values of a column with {n}+ distinct values; NOT exhaustive.
-- * min/max are over non-NULL values. nulls=N appears only when the column has NULLs.
-- * Table sample rows are the first {s} rows; illustrative only.
"""


def _q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def _fmt(v, num_digits: int = 10) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return f"{v:.{num_digits}g}"
    if isinstance(v, decimal.Decimal):
        return str(v)
    if isinstance(v, (datetime.datetime, datetime.date, datetime.time)):
        return v.isoformat()
    if isinstance(v, str):
        v = " ".join(v.split())                       # no newlines inside comment lines
        if len(v) > MAX_STR:
            v = v[: MAX_STR - 1] + "…"
        return "'" + v.replace("'", "''") + "'"
    return " ".join(str(v).split())[: MAX_STR * 2]


def _dump_object(con, kind, schema, name, ddl, max_distinct, sample_rows) -> list[str]:
    qname = _q(name) if schema == "main" else f"{_q(schema)}.{_q(name)}"
    cols = con.execute(
        "SELECT column_name, data_type FROM duckdb_columns() "
        "WHERE database_name = current_database() AND schema_name = ? AND table_name = ? "
        "ORDER BY column_index", [schema, name]).fetchall()

    out = []
    ddl = (ddl or "").rstrip().rstrip(";") + ";"

    # ── one scan: row count + per-column distinct / nulls / min / max ─────────
    exprs, slots = ["COUNT(*)"], {}                    # slots: col -> index positions
    for col, typ in cols:
        if _NESTED.search(typ):
            continue
        c = _q(col)
        slots[col] = {"distinct": len(exprs)}
        exprs.append(f"COUNT(DISTINCT {c})")
        slots[col]["nulls"] = len(exprs)
        exprs.append(f"COUNT(*) FILTER (WHERE {c} IS NULL)")
        if _NUMERIC.match(typ) or _TEMPORAL.match(typ):
            slots[col]["min"] = len(exprs); exprs.append(f"MIN({c})")
            slots[col]["max"] = len(exprs); exprs.append(f"MAX({c})")
    try:
        stats = con.execute(f"SELECT {', '.join(exprs)} FROM {qname}").fetchone()
        n_rows, err = stats[0], None
    except Exception as e:                             # e.g. a view that fails at runtime
        stats, n_rows, err = None, None, str(e).splitlines()[0][:150]

    out.append(f"-- ===== {kind.upper()} {name}" + (f" — {n_rows:,} rows" if n_rows is not None else "") + " =====")
    out.append(ddl)
    if err:
        out.append(f"-- (stats unavailable: {err})")
        out.append("")
        return out

    # ── column details ────────────────────────────────────────────────────────
    out.append("-- Columns:")
    for col, typ in cols:
        s = slots.get(col)
        parts = [f"{col} {typ}"]
        if s:
            n = stats[s["distinct"]]
            parts.append("no rows" if not n_rows else "all NULL" if n == 0 else f"{n:,} distinct")
            if "min" in s and n:
                parts.append(f"min={_fmt(stats[s['min']])}, max={_fmt(stats[s['max']])}")
            if stats[s["nulls"]]:
                parts.append(f"nulls={stats[s['nulls']]:,}")
            if 0 < n < max_distinct:                      # small domain -> dump all as ENUM
                vals = con.execute(
                    f"SELECT DISTINCT {_q(col)} FROM {qname} WHERE {_q(col)} IS NOT NULL ORDER BY 1"
                ).fetchall()
                parts.append("ENUM(" + ", ".join(_fmt(v[0]) for v in vals) + ")")
            elif n >= max_distinct:                       # large domain -> 3 samples only
                vals = con.execute(
                    f"SELECT DISTINCT {_q(col)} FROM {qname} WHERE {_q(col)} IS NOT NULL LIMIT {SAMPLE_VALUES}"
                ).fetchall()
                parts.append("samples: " + ", ".join(_fmt(v[0]) for v in vals))
        out.append("--   " + " | ".join(parts))

    # ── sample rows ───────────────────────────────────────────────────────────
    if sample_rows:
        cur = con.execute(f"SELECT * FROM {qname} LIMIT {int(sample_rows)}")
        names = [d[0] for d in cur.description]
        rows = cur.fetchall()
        if rows:
            out.append(f"-- Sample rows ({len(rows)}):")
            for r in rows:
                out.append("--   {" + ", ".join(f"{n}: {_fmt(v)}" for n, v in zip(names, r)) + "}")
    out.append("")
    return out


def dump_schema(con, out_path: str | None = None,
                max_distinct: int = MAX_DISTINCT, sample_rows: int = SAMPLE_ROWS) -> str:
    """Return the schema text (and write it to out_path when given)."""
    objs = con.execute("""
        SELECT 'table' AS kind, schema_name, table_name AS name, sql FROM duckdb_tables()
         WHERE database_name = current_database() AND NOT internal AND NOT temporary
        UNION ALL
        SELECT 'view', schema_name, view_name, sql FROM duckdb_views()
         WHERE database_name = current_database() AND NOT internal AND NOT temporary
        ORDER BY 1, 2, 3
    """).fetchall()

    lines = [HEADER.format(n=max_distinct, s=sample_rows, v=SAMPLE_VALUES)]
    for kind, schema, name, ddl in objs:
        lines += _dump_object(con, kind, schema, name, ddl, max_distinct, sample_rows)

    text = "\n".join(lines)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def _load_lab(path: str):
    """Import page_sql_lab.py by file path (no package layout needed)."""
    spec = importlib.util.spec_from_file_location("page_sql_lab", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_lab(explicit: str | None) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for p in ([explicit] if explicit else []) + [
        os.path.join(os.getcwd(), "ui", "pages", "page_sql_lab.py"),
        os.path.join(here, "ui", "pages", "page_sql_lab.py"),
        os.path.join(here, "page_sql_lab.py"),
    ]:
        if p and os.path.exists(p):
            return p
    sys.exit("page_sql_lab.py not found - pass --lab <path>")


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump the SQL Lab DuckDB schema for LLM prompts")
    ap.add_argument("--lab", help="path to page_sql_lab.py (default: ui/pages/page_sql_lab.py)")
    ap.add_argument("-o", "--out", help="output file (default: <BASE>/sqls/schema.sql)")
    ap.add_argument("--max-distinct", type=int, default=MAX_DISTINCT)
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())                       # so `from bll import agent` resolves
    lab = _load_lab(_find_lab(args.lab))

    con, log = lab._lab_db()                              # runs init_raw -> ... -> buy_candidates from sqls/
    for e in log:
        print(f"  {e['status']:<8} {e['script']}" + (f"  !! {e['error']}" if e["error"] else ""))
    if any(e["error"] for e in log):
        sys.exit("DB build failed - schema not written.")

    out = args.out or os.path.join(lab._BASE, "sqls", "schema.sql")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    dump_schema(con, out, max_distinct=args.max_distinct)
    print(f"wrote {out}")
    con.close()


if __name__ == "__main__":
    main()
