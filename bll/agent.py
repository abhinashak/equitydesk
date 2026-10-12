import os
import sys
import json
import re
import time
import hashlib
from pathlib import Path

import duckdb
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.duckdb import DuckDbTools

# ======================= config =======================
BASE = Path(__file__).resolve().parent.parent                 # the equitydesk folder
SQL_DIR = BASE / "sqls"
SCRIPTS = ["init_raw.sql", "init.sql", "compact_clean.sql" , "dcf.sql" ]    # order matters
DB = BASE / "equity.duckdb"                            # must be a FILE
CACHE = BASE / "column_values.json"

MODEL_ID = "gemma-offline:latest"
OLLAMA_HOST = "http://localhost:11434"
NUM_CTX = 16384
HISTORY_RUNS = 3                     # previous Q&A turns the agent remembers in the loop (0 = none)

ALLOWED_TABLES = None                # e.g. ["general_info", "gate_score"]; None = all
EXCLUDE_SUFFIXES = ("_raw",)
MAX_DISTINCT = 512
NUMERIC_MAX_DISTINCT = 40
PROMPT_BUDGET_CHARS = 14000
MAX_VALUE_LEN = 60
SKIP_COLS = {"about"}
AUTO_UNION_BY_NAME = True
DEBUG = "--debug" in sys.argv

TEXT_TYPES = ("VARCHAR", "CHAR", "TEXT", "STRING")
NUM_TYPES = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UBIGINT", "UINTEGER",
             "USMALLINT", "UTINYINT", "DOUBLE", "FLOAT", "REAL", "DECIMAL", "BOOLEAN")
DATE_TYPES = ("DATE", "TIMESTAMP")

STATE = {"profile": None, "schema": None}              # filled once in main(), used by the lookup tool
WARNINGS = []


def warn(msg):
    WARNINGS.append(msg)
    if DEBUG:
        print("warning:", msg)


def _is(dtype, prefixes):
    return dtype.startswith(prefixes)


# ======================= STEP 1: build the database (once) =======================
UNION_RE = re.compile(r"read_parquet\s*\(([^()]*)\)", re.IGNORECASE)


def add_union_by_name(sql):
    count = 0

    def repl(m):
        nonlocal count
        inner = m.group(1)
        if "union_by_name" in inner.lower():
            return m.group(0)
        count += 1
        return "read_parquet(" + inner.rstrip() + ", union_by_name=true)"

    return UNION_RE.sub(repl, sql), count


def build_db():
    for p in (DB, Path(str(DB) + ".wal")):
        if p.exists():
            p.unlink()
    con = duckdb.connect(str(DB))
    try:
        for name in SCRIPTS:
            path = SQL_DIR / name
            if not path.exists():
                raise SystemExit(f"Missing SQL file: {path}")
            sql = path.read_text()
            if AUTO_UNION_BY_NAME:
                sql, n = add_union_by_name(sql)
                if n:
                    print(f"  (added union_by_name=true to {n} read_parquet call(s))")
            print("running:", path.relative_to(BASE))
            try:
                con.execute(sql)
            except Exception as e:
                raise SystemExit(f"\n{name} failed:\n{e}")
            print("done:   ", name)
    finally:
        con.close()                                    # release the write lock before the agent connects


# ======================= STEP 2: profile columns (once, cached) =======================
def approx_counts(con, t, cand):
    try:
        exprs = ", ".join(f'approx_count_distinct("{c}")' for c, _ in cand)
        return list(con.execute(f'SELECT {exprs} FROM "{t}"').fetchone())
    except Exception as e:
        warn(f"{t}: one-scan distinct count failed ({str(e)[:100]}); trying per column")
    result = []
    for c, _ in cand:
        try:
            result.append(con.execute(f'SELECT approx_count_distinct("{c}") FROM "{t}"').fetchone()[0])
        except Exception as e:
            warn(f"{t}.{c}: skipped ({str(e)[:100]})")
            result.append(None)
    return result


def profile_db():
    con = duckdb.connect(str(DB), read_only=True)
    out = {"schema": {}, "info": {}, "values": {}, "ranges": {}}
    try:
        objs = con.execute("""
            SELECT table_name, table_type FROM information_schema.tables
            WHERE table_schema = 'main' ORDER BY 1""").fetchall()
        for t, kind in objs:
            if ALLOWED_TABLES and t not in ALLOWED_TABLES:
                continue
            if t.lower().endswith(EXCLUDE_SUFFIXES):
                continue
            try:
                cols = [(c[0], c[1].upper()) for c in con.execute(f'DESCRIBE "{t}"').fetchall()]
            except Exception as e:
                out["info"][t] = f"{kind}, BROKEN: {str(e)[:120]}"
                warn(f"{t}: cannot describe ({str(e)[:100]})")
                continue
            out["schema"][t] = cols
            try:
                n_rows = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                out["info"][t] = f"{kind}, {n_rows:,} rows"
            except Exception as e:
                n_rows = None
                out["info"][t] = f"{kind}, row count failed"
                warn(f"{t}: row count failed ({str(e)[:100]})")

            for c, d in cols:
                if _is(d, DATE_TYPES):
                    try:
                        lo, hi = con.execute(f'SELECT MIN("{c}"), MAX("{c}") FROM "{t}"').fetchone()
                        out["ranges"][f"{t}.{c}"] = f"{lo} to {hi}"
                    except Exception as e:
                        warn(f"{t}.{c}: date range skipped ({str(e)[:100]})")

            cand = [(c, d) for c, d in cols
                    if c not in SKIP_COLS and (_is(d, TEXT_TYPES) or _is(d, NUM_TYPES))]
            if not cand or n_rows == 0:
                continue

            approx = approx_counts(con, t, cand)
            for (c, d), a in zip(cand, approx):
                limit = MAX_DISTINCT if _is(d, TEXT_TYPES) else NUMERIC_MAX_DISTINCT
                if a is None or a == 0 or a > limit * 1.2:
                    continue
                try:
                    rows = con.execute(
                        f'SELECT DISTINCT "{c}" FROM "{t}" WHERE "{c}" IS NOT NULL '
                        f'ORDER BY 1 LIMIT {limit + 1}').fetchall()
                except Exception as e:
                    warn(f"{t}.{c}: values skipped ({str(e)[:100]})")
                    continue
                if len(rows) > limit or not rows:
                    continue
                vals = [str(r[0])[:MAX_VALUE_LEN] for r in rows]
                out["values"][f"{t}.{c}"] = {"n": len(vals), "values": vals}
    finally:
        con.close()
    out["warnings"] = WARNINGS
    return out


def load_profile():
    settings = [MAX_DISTINCT, NUMERIC_MAX_DISTINCT, MAX_VALUE_LEN, sorted(SKIP_COLS),
                ALLOWED_TABLES, list(EXCLUDE_SUFFIXES)]
    mtime = DB.stat().st_mtime
    if CACHE.exists() and "--refresh-values" not in sys.argv:
        try:
            c = json.loads(CACHE.read_text())
            if c.get("db_mtime") == mtime and c.get("settings") == json.loads(json.dumps(settings)):
                print("using cached column values:", CACHE.name)
                return c
        except Exception:
            pass
    print("profiling columns (first run takes a little while)...")
    c = profile_db()
    c["db_mtime"] = mtime
    c["settings"] = settings
    CACHE.write_text(json.dumps(c))
    return c


# ======================= STEP 3: prompt text =======================
def build_values_prompt(prof):
    groups = {}
    for key, v in prof["values"].items():
        sig = hashlib.md5("\x1f".join(v["values"]).encode()).hexdigest()
        groups.setdefault(sig, []).append(key)

    entries = []
    for keys in groups.values():
        vals = prof["values"][keys[0]]["values"]
        label = " = ".join(keys[:6])
        if len(keys) > 6:
            label += f" (+{len(keys) - 6} more)"
        entries.append((len(vals), label, vals, keys))
    entries.sort(key=lambda e: (e[0], e[1]))

    lines, deferred, used = [], [], 0
    for n, label, vals, keys in entries:
        line = f"- {label} [{n}]: " + " | ".join(vals)
        if used + len(line) <= PROMPT_BUDGET_CHARS:
            lines.append(line)
            used += len(line)
        else:
            deferred.extend(keys)
    return "\n".join(lines), deferred


def build_instructions(prof):
    schema = prof["schema"]
    values_text, deferred = build_values_prompt(prof)
    ranges_text = "\n".join(f"- {k}: {v}" for k, v in prof["ranges"].items())
    schema_text = "\n".join(
        f"- {t}: " + ", ".join(f"{c[0]} ({c[1]})" for c in cols) for t, cols in schema.items())

    n_profiled = len(prof["values"])
    print(f"\ncolumn values: {n_profiled} columns profiled, "
          f"{n_profiled - len(deferred)} inline, {len(deferred)} via get_column_values tool, "
          f"prompt values ~{len(values_text):,} chars")

    ins = [
        f"Use ONLY these tables: {', '.join(schema)}. Do not use any other table.",
        "Never query tables whose names end in _raw.",
        "Schema (complete and authoritative):\n" + schema_text,
        "Distinct values. Use these EXACT spellings in WHERE clauses with = or IN. "
        "Entries joined by ' = ' share the same value list:\n" + values_text,
    ]
    if deferred:
        ins.append("Values for these columns are NOT listed above; call "
                   "get_column_values(table, column) to see them: " + ", ".join(deferred))
    if ranges_text:
        ins.append("Date ranges of the data (check freshness):\n" + ranges_text)
    ins += [
        "Some data is missing: columns can be NULL for some companies. Handle NULLs and say so if it affects the answer.",
        "If a value you need is not listed, call get_column_values(table, column, contains). Never guess a value.",
        "Do NOT call show_tables or describe_table. The schema above is complete.",
        "Only run SELECT queries. Never modify data. Always use LIMIT.",
        "Run one run_query call, then answer from its result.",
        "If a query returns an error, show the exact error text.",
        "Earlier questions in this conversation may be referred to (for example 'and for Construction?'); reuse their context.",
    ]
    return ins


# ======================= STEP 4: lookup tool =======================
def get_column_values(table: str, column: str, contains: str = "") -> str:
    """List the distinct values of a column so you can use the exact spelling in a WHERE clause.
    Optionally filter by a text fragment with `contains`. Use this before filtering on any
    value that was not shown in the prompt."""
    schema, profile = STATE["schema"], STATE["profile"]
    if table not in schema:
        return "Unknown table. Choose from: " + ", ".join(schema)
    names = [c[0] for c in schema[table]]
    if column not in names:
        return "Unknown column. Choose from: " + ", ".join(names)

    entry = profile["values"].get(f"{table}.{column}")
    if entry:
        vals = entry["values"]
        if contains:
            vals = [v for v in vals if contains.lower() in v.lower()]
        return f"{len(vals)} values: " + " | ".join(vals) if vals else "no match"

    if not contains:
        return (f"{table}.{column} has more than {MAX_DISTINCT} distinct values. "
                "Pass `contains` with a fragment to search.")
    try:
        con = duckdb.connect(str(DB), read_only=True)
        try:
            rows = con.execute(
                f'SELECT DISTINCT CAST("{column}" AS VARCHAR) FROM "{table}" '
                f'WHERE CAST("{column}" AS VARCHAR) ILIKE ? ORDER BY 1 LIMIT 30',
                [f"%{contains}%"]).fetchall()
        finally:
            con.close()
    except Exception as e:
        return f"lookup failed: {str(e)[:200]}"
    return " | ".join(r[0] for r in rows) if rows else "no match"


# ======================= STEP 5: create the agent (once) =======================
def create_agent(instructions):
    base = dict(
        name="SQL Analyst",
        model=Ollama(id=MODEL_ID, host=OLLAMA_HOST, options={"num_ctx": NUM_CTX}),
        tools=[DuckDbTools(db_path=str(DB), read_only=True), get_column_values],
        instructions=instructions,
        markdown=True,
        debug_mode=DEBUG,
    )
    if HISTORY_RUNS > 0:
        try:    # Agno 2.x argument names
            return Agent(**base, add_history_to_context=True, num_history_runs=HISTORY_RUNS)
        except TypeError:
            try:    # older Agno
                return Agent(**base, add_history_to_messages=True, num_history_responses=HISTORY_RUNS)
            except TypeError:
                print("note: this Agno version has no history option; each question is independent")
    return Agent(**base)


# ======================= STEP 6: the loop =======================
HELP = """commands:
  :tables               list tables with row counts
  :values table.column  show the distinct values profiled for a column
  :new                  start a fresh conversation (forget earlier questions)
  :help                 show this help
  :quit                 exit (or press Ctrl+D)
"""


def ask(agent, question):
    start = time.time()
    try:
        agent.print_response(question, stream=True)
    except Exception as e:                 # a failed question must not end the loop
        print(f"\n[error] {type(e).__name__}: {str(e)[:300]}")
    print(f"\n({time.time() - start:.1f}s)")


def handle_command(cmd, state):
    parts = cmd.split(maxsplit=1)
    name, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")
    prof = STATE["profile"]
    if name in (":quit", ":exit", ":q"):
        return "quit"
    if name == ":help":
        print(HELP)
    elif name == ":tables":
        for t, i in prof["info"].items():
            print(f"  {t}  [{i}]")
    elif name == ":values":
        entry = prof["values"].get(arg)
        print(f"{entry['n']} values: " + " | ".join(entry["values"]) if entry
              else f"no profiled values for '{arg}' (use table.column)")
    elif name == ":new":
        return "new"
    else:
        print("unknown command; type :help")
    return None


def run_loop(instructions, first_question=None):
    agent = create_agent(instructions)

    if first_question:                       # question given on the command line: answer it, then keep going
        print(f"\n> {first_question}")
        ask(agent, first_question)

    print("\nReady. Ask a question, or type :help. (:quit or Ctrl+D to exit)")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:                            # blank line: just prompt again
            continue
        if q.startswith(":"):
            result = handle_command(q, STATE)
            if result == "quit":
                break
            if result == "new":
                agent = create_agent(instructions)       # fresh agent = fresh history
                print("started a new conversation")
            continue
        ask(agent, q)                        # answer, then the loop shows the prompt again


# ======================= main =======================
def main():
    if DB.is_dir():
        raise SystemExit(f"DB path is a folder, need a file: {DB}")
    os.chdir(BASE)                                       # relative parquet paths in the SQL resolve from here

    # ---- once: build, profile, schema ----
    if "--rebuild" in sys.argv or not DB.exists():
        build_db()
    profile = load_profile()
    schema = profile["schema"]
    STATE["profile"], STATE["schema"] = profile, schema

    print(f"\n{len(profile['info'])} tables/views in {DB.name}:")
    for t, i in profile["info"].items():
        cols = ", ".join(c[0] for c in schema[t]) if t in schema else ""
        print(f"  {t}  [{i}]" + (f"  ->  {cols}" if cols else ""))

    saved = profile.get("warnings", [])
    if saved:
        print(f"\n{len(saved)} profiling warning(s) (some data may be missing or unreadable):")
        for w in saved[:10]:
            print("  -", w)
    if not schema:
        raise SystemExit("\nNo readable tables. Check the SQL scripts, then rerun with --rebuild.")

    instructions = build_instructions(profile)
    first = " ".join(a for a in sys.argv[1:] if not a.startswith("--")) or None
    run_loop(instructions, first_question=first)

if __name__ == "__main__":
    main()
