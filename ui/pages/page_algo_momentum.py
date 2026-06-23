"""
FundamentalSpreadSignal — Streamlit App
========================================
Sector-relative composite ranking built entirely from DuckDB views
defined in init.sql / compact_clean.sql.

Composite = weighted mean of z-scores (or percentile ranks) of:
  • ROE momentum   : delta ROE over last 4 quarters      → higher is better
  • Accruals ratio : (NI − CFO) / Total Assets           → lower is better (negated)
  • Rev CV         : coeff-of-variation of last 4Q sales → lower is better (negated)
  • D/E trend      : delta Debt/Equity over 4 annual pts → declining is better (negated)

Grouping   : broad_industry  (from general_info)
ROE mom    : quarterly_results  (quarterly cadence)
D/E trend  : balance_sheet      (annual cadence, more complete levels)
Accruals   : profit_loss (NI) + cash_flows (CFO) + balance_sheet (assets)  [annual]
Rev CV     : quarterly_results  (last 4 quarters of Sales)
"""

from __future__ import annotations
import warnings
import duckdb
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def render():
    # ── App config ──────────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Fundamental Spread Signal",
        page_icon="📊",
        layout="wide",
    )

    DB_PATH = ":memory:"
    INIT_SQL_PATH = "sqls/init.sql"

    # ── Sidebar controls ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("Signal Parameters")
    score_method = st.radio("Scoring method", ["zscore", "rank"], index=0)
    min_per_sector = st.slider("Min stocks per industry", 2, 20, 4)
    top_pct = st.slider("Long/Short top %", 0.10, 0.40, 0.25, step=0.05)
    weighting = st.radio("Book weighting", ["equal", "score"], index=0)

    # ── DuckDB connection ────────────────────────────────────────────────────────
    @st.cache_resource
    def get_conn(db_path: str, init_sql: str) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect(db_path)
        import os
        sql_dir = os.path.dirname(init_sql)
        # Execute init.sql (view definitions) then compact_clean.sql if present
        for sql_file in [init_sql, os.path.join(sql_dir, "compact_clean.sql")]:
            if not os.path.exists(sql_file):
                continue
            try:
                with open(sql_file) as f:
                    sql = f.read()
                for stmt in sql.split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        try:
                            conn.execute(stmt)
                        except Exception:
                            pass  # skip stmts whose parquet files don't exist yet
            except FileNotFoundError:
                st.warning(f"SQL file not found: '{sql_file}'. Assuming views already exist in DB.")
        return conn


    conn = get_conn(DB_PATH, INIT_SQL_PATH)

    # ── SQL helpers ──────────────────────────────────────────────────────────────
    def query(sql: str) -> pd.DataFrame:
        return conn.execute(sql).df()


    # ── Core data pulls ──────────────────────────────────────────────────────────
    @st.cache_data(ttl=600, show_spinner="Loading general info…")
    def load_general_info() -> pd.DataFrame:
        return query("""
            SELECT ticker, company_name, sector, broad_sector, broad_industry, industry,
                   roe, roce, stock_p_e, market_cap, current_price
            FROM general_info
            QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY ticker) = 1
        """)


    @st.cache_data(ttl=600, show_spinner="Computing ROE momentum (quarterly)…")
    def load_roe_momentum() -> pd.DataFrame:
        """Delta ROE over last 5 quarters using quarterly net_profit / equity_capital.

        Equity = Reserves + Equity Capital from balance_sheet (annual, repeated forward-fill).
        Net Profit from quarterly_results.
        """
        # Net Profit quarterly
        np_q = query("""
            WITH ranked AS (
                SELECT ticker, dt, val,
                       DENSE_RANK() OVER (PARTITION BY ticker ORDER BY dt DESC) AS rk
                FROM quarterly_results
                WHERE metric = 'Net Profit' AND val IS NOT NULL
            )
            SELECT ticker, dt, val AS net_profit, rk
            FROM ranked WHERE rk <= 5
        """)

        # Equity from balance_sheet annual (reserves + equity_capital)
        eq = query("""
            WITH bs AS (
                SELECT ticker, dt,
                       MAX(CASE WHEN metric IN ('Equity Capital') THEN val END) AS eq_cap,
                       MAX(CASE WHEN metric = 'Reserves'         THEN val END) AS reserves
                FROM balance_sheet
                GROUP BY ticker, dt
            )
            SELECT ticker, dt,
                   COALESCE(eq_cap, 0) + COALESCE(reserves, 0) AS equity
            FROM bs
            WHERE equity > 0
        """)

        if np_q.empty or eq.empty:
            return pd.DataFrame(columns=["ticker", "roe_momentum"])

        # For each ticker/quarter, find the most recent annual equity on or before that quarter
        np_q["dt"] = pd.to_datetime(np_q["dt"])
        eq["dt"] = pd.to_datetime(eq["dt"])

        # merge_asof requires the join key (dt) to be sorted globally — not by (ticker, dt)
        np_q = np_q.sort_values("dt").reset_index(drop=True)
        eq_sorted = eq.sort_values("dt").reset_index(drop=True)

        merged = pd.merge_asof(
            np_q, eq_sorted.rename(columns={"dt": "eq_dt"}),
            left_on="dt", right_on="eq_dt",
            by="ticker", direction="backward"
        )
        merged["roe_q"] = merged["net_profit"] / merged["equity"].replace(0, np.nan)

        # roe_momentum = latest roe_q minus roe_q 4 quarters ago
        result = []
        for tkr, grp in merged.groupby("ticker"):
            grp = grp.sort_values("dt")
            if len(grp) >= 2:
                roe_momentum = grp["roe_q"].iloc[-1] - grp["roe_q"].iloc[0]
                result.append({"ticker": tkr, "roe_momentum": roe_momentum})
        return pd.DataFrame(result)


    @st.cache_data(ttl=600, show_spinner="Computing accruals (annual)…")
    def load_accruals() -> pd.DataFrame:
        """Accruals = (Net Income − CFO) / Total Assets.  Annual cadence."""
        ni = query("""
            WITH yr AS (
                SELECT ticker, dt, DENSE_RANK() OVER (PARTITION BY ticker ORDER BY dt DESC) AS rk
                FROM (SELECT DISTINCT ticker, dt FROM profit_loss WHERE dt < CURRENT_DATE)
            )
            SELECT p.ticker, p.dt,
                   MAX(CASE WHEN metric = 'Net Profit' THEN val END) AS net_income
            FROM profit_loss p JOIN yr ON p.ticker = yr.ticker AND p.dt = yr.dt
            WHERE yr.rk = 1
            GROUP BY p.ticker, p.dt
        """)

        cfo = query("""
            SELECT ticker,
                   MAX(CASE WHEN metric = 'Cash from Operating Activity' THEN val END) AS cfo
            FROM cash_flows
            WHERE dt = (SELECT MAX(dt) FROM cash_flows c2 WHERE c2.ticker = cash_flows.ticker)
            GROUP BY ticker
        """)

        assets = query("""
            SELECT ticker,
                   MAX(CASE WHEN metric = 'Total Assets' THEN val END) AS total_assets
            FROM balance_sheet
            WHERE dt = (SELECT MAX(dt) FROM balance_sheet b2 WHERE b2.ticker = balance_sheet.ticker)
            GROUP BY ticker
        """)

        if ni.empty or cfo.empty or assets.empty:
            return pd.DataFrame(columns=["ticker", "accruals"])

        df = ni.merge(cfo, on="ticker", how="inner").merge(assets, on="ticker", how="inner")
        df["accruals"] = (df["net_income"] - df["cfo"]) / df["total_assets"].replace(0, np.nan)
        return df[["ticker", "accruals"]].dropna()


    @st.cache_data(ttl=600, show_spinner="Computing revenue CV (quarterly)…")
    def load_rev_cv() -> pd.DataFrame:
        """Coefficient of variation of sales over last 4 quarters."""
        sales = query("""
            WITH ranked AS (
                SELECT ticker, dt, val,
                       DENSE_RANK() OVER (PARTITION BY ticker ORDER BY dt DESC) AS rk
                FROM quarterly_results
                WHERE metric IN ('Sales', 'Revenue') AND val IS NOT NULL
            )
            SELECT ticker, val FROM ranked WHERE rk <= 4
        """)
        if sales.empty:
            return pd.DataFrame(columns=["ticker", "rev_cv"])

        def _cv(s):
            mu = s.mean()
            return s.std() / abs(mu) if mu != 0 else np.nan

        cv = sales.groupby("ticker")["val"].apply(_cv).reset_index()
        cv.columns = ["ticker", "rev_cv"]
        return cv.dropna()


    @st.cache_data(ttl=600, show_spinner="Computing D/E trend (annual)…")
    def load_de_trend() -> pd.DataFrame:
        """Delta Debt/Equity over last 4 annual balance-sheet periods."""
        de_raw = query("""
            WITH bs AS (
                SELECT ticker, dt,
                       MAX(CASE WHEN metric IN ('Borrowing','Borrowings') THEN val END) AS debt,
                       MAX(CASE WHEN metric = 'Reserves'                  THEN val END) AS reserves,
                       MAX(CASE WHEN metric = 'Equity Capital'            THEN val END) AS eq_cap
                FROM balance_sheet
                GROUP BY ticker, dt
            ),
            ranked AS (
                SELECT *,
                       DENSE_RANK() OVER (PARTITION BY ticker ORDER BY dt DESC) AS rk
                FROM bs
                WHERE dt < CURRENT_DATE
            )
            SELECT ticker, dt, rk,
                   debt / NULLIF(COALESCE(eq_cap,0) + COALESCE(reserves,0), 0) AS de_ratio
            FROM ranked WHERE rk <= 4 AND de_ratio IS NOT NULL
        """)
        if de_raw.empty:
            return pd.DataFrame(columns=["ticker", "de_trend"])

        result = []
        for tkr, grp in de_raw.groupby("ticker"):
            grp = grp.sort_values("dt")
            if len(grp) >= 2:
                de_trend = grp["de_ratio"].iloc[-1] - grp["de_ratio"].iloc[0]
                result.append({"ticker": tkr, "de_trend": de_trend})
        return pd.DataFrame(result)


    # ── Scoring functions ────────────────────────────────────────────────────────
    def _zscore(s: pd.Series) -> pd.Series:
        std = s.std()
        return pd.Series(0.0, index=s.index) if (std == 0 or pd.isna(std)) else (s - s.mean()) / std


    def _percentile_rank(s: pd.Series) -> pd.Series:
        if len(s) < 2:
            return pd.Series(0.0, index=s.index)
        return s.rank(pct=True) - 0.5


    def score_universe(
            metrics: pd.DataFrame,
            sectors: pd.Series,
            min_per_sector: int = 4,
            method: str = "zscore",
    ) -> pd.DataFrame:
        if metrics.empty:
            return pd.DataFrame()

        transform = _zscore if method == "zscore" else _percentile_rank
        cols = ["roe_momentum", "accruals", "rev_cv", "de_trend"]
        df = metrics.join(sectors.rename("industry"), how="inner").dropna(subset=cols)

        pieces = []
        for ind, group in df.groupby("industry"):
            if len(group) < min_per_sector:
                continue
            z = pd.DataFrame({
                "z_roe":  transform(group["roe_momentum"]),
                "z_accr": -transform(group["accruals"]),
                "z_rev":  -transform(group["rev_cv"]),
                "z_de":   -transform(group["de_trend"]),
            })
            out = group.copy()
            out["z_roe"] = z["z_roe"]
            out["z_accr"] = z["z_accr"]
            out["z_rev"] = z["z_rev"]
            out["z_de"] = z["z_de"]
            out["composite"] = z.mean(axis=1)
            pieces.append(out)

        return pd.concat(pieces) if pieces else pd.DataFrame()


    def build_book(scored: pd.DataFrame, top_pct: float, weighting: str) -> pd.DataFrame:
        if scored.empty:
            return pd.DataFrame(columns=["ticker", "industry", "side", "weight", "composite"])

        rows = []
        for ind, group in scored.groupby("industry"):
            n = len(group)
            k = max(1, int(round(n * top_pct)))
            ranked = group.sort_values("composite", ascending=False)
            longs = ranked.head(k)
            shorts = ranked.tail(k)

            for side, bucket in [("L", longs), ("S", shorts)]:
                if bucket.empty:
                    continue
                if weighting == "score":
                    vals = bucket["composite"] if side == "L" else -bucket["composite"]
                    shifted = vals - vals.min() + 0.1 * (vals.max() - vals.min() + 1e-9)
                    w = shifted / shifted.sum()
                else:
                    w = pd.Series(1.0 / k, index=bucket.index)
                for tkr in bucket.index:
                    rows.append({
                        "ticker": tkr,
                        "industry": ind,
                        "side": side,
                        "weight": float(w[tkr]),
                        "composite": float(bucket.loc[tkr, "composite"]),
                    })
        return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


    # ── Main UI ──────────────────────────────────────────────────────────────────
    st.title("📊 Fundamental Spread Signal")
    st.caption("Sector-relative composite ranking · DuckDB-powered")

    with st.spinner("Loading data from DuckDB…"):
        gi = load_general_info()
        roe_df   = load_roe_momentum()
        accr_df  = load_accruals()
        revcv_df = load_rev_cv()
        de_df    = load_de_trend()

    if gi.empty:
        st.error("Could not load general_info view. Check DB path and init.sql path in the sidebar.")
        st.stop()

    # Build metrics table
    metrics_df = (
        roe_df
        .merge(accr_df,  on="ticker", how="outer")
        .merge(revcv_df, on="ticker", how="outer")
        .merge(de_df,    on="ticker", how="outer")
        .set_index("ticker")
    )

    sector_map = gi.drop_duplicates("ticker").set_index("ticker")["broad_industry"]
    metrics_df = metrics_df.join(sector_map, how="left")

    # ── Coverage stats ───────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Universe", len(gi["ticker"].unique()))
    col2.metric("ROE momentum", roe_df["ticker"].nunique())
    col3.metric("Accruals", accr_df["ticker"].nunique())
    col4.metric("Rev CV", revcv_df["ticker"].nunique())
    col5.metric("D/E trend", de_df["ticker"].nunique())

    st.markdown("---")

    # ── Score ────────────────────────────────────────────────────────────────────
    scored = score_universe(
        metrics_df.drop(columns=["broad_industry"], errors="ignore"),
        metrics_df["broad_industry"],
        min_per_sector=min_per_sector,
        method=score_method,
    )

    if scored.empty:
        st.warning("Not enough data to score. Try lowering 'Min stocks per industry'.")
        st.stop()

    scored = scored.join(
        gi.drop_duplicates("ticker").set_index("ticker")[
            ["company_name", "sector", "broad_sector", "market_cap", "current_price", "stock_p_e", "roe", "roce"]
        ],
        how="left"
    )

    book = build_book(scored, top_pct=top_pct, weighting=weighting)

    # ── Tabs ─────────────────────────────────────────────────────────────────────
    tab_universe, tab_book, tab_charts = st.tabs(["📋 Scored Universe", "📒 Long / Short Book", "📈 Charts"])

    # ── Tab 1: Full Universe ─────────────────────────────────────────────────────
    with tab_universe:
        st.subheader("Full Scored Universe")

        industry_filter = st.multiselect(
            "Filter by broad_industry",
            options=sorted(scored["industry"].dropna().unique()),
            default=[],
        )
        display = scored if not industry_filter else scored[scored["industry"].isin(industry_filter)]

        show_cols = [
            "company_name", "industry", "sector", "composite",
            "z_roe", "z_accr", "z_rev", "z_de",
            "roe_momentum", "accruals", "rev_cv", "de_trend",
            "market_cap", "current_price", "stock_p_e", "roe", "roce",
        ]
        show_cols = [c for c in show_cols if c in display.columns]

        styled = display[show_cols].sort_values("composite", ascending=False)

        def _color_composite(val):
            if pd.isna(val):
                return ""
            if val > 0.5:
                return "background-color: #c6efce; color: #276221"
            if val > 0:
                return "background-color: #ebf5eb"
            if val > -0.5:
                return "background-color: #fce4d6"
            return "background-color: #f4b8b0; color: #8b0000"

        fmt = {c: "{:.3f}" for c in ["composite", "z_roe", "z_accr", "z_rev", "z_de",
                                     "roe_momentum", "accruals", "rev_cv", "de_trend"]}
        fmt.update({"market_cap": "{:.0f}", "current_price": "{:.1f}", "stock_p_e": "{:.1f}",
                    "roe": "{:.1f}", "roce": "{:.1f}"})
        fmt = {k: v for k, v in fmt.items() if k in styled.columns}

        st.dataframe(
            styled.style
            .map(_color_composite, subset=["composite"])
            .format(fmt, na_rep="—"),
            use_container_width=True,
            height=600,
            )

        st.caption(f"Showing {len(display)} stocks across {display['industry'].nunique()} industries.")

    # ── Tab 2: Long / Short Book ─────────────────────────────────────────────────
    with tab_book:
        st.subheader("Long / Short Book")
        if book.empty:
            st.info("No book to display.")
        else:
            book_display = book.join(
                gi.drop_duplicates("ticker").set_index("ticker")[["company_name", "sector", "market_cap", "current_price"]],
                how="left"
            ).reset_index()

            col_l, col_s = st.columns(2)
            with col_l:
                st.markdown("### 🟢 Longs")
                longs = book_display[book_display["side"] == "L"].sort_values("composite", ascending=False)
                st.dataframe(
                    longs[["ticker", "company_name", "industry", "sector", "composite", "weight", "market_cap", "current_price"]]
                    .style.format({"composite": "{:.3f}", "weight": "{:.3f}", "market_cap": "{:.0f}", "current_price": "{:.1f}"}, na_rep="—"),
                    use_container_width=True,
                    height=500,
                    )
                st.caption(f"{len(longs)} long positions across {longs['industry'].nunique()} industries")

            with col_s:
                st.markdown("### 🔴 Shorts")
                shorts = book_display[book_display["side"] == "S"].sort_values("composite", ascending=True)
                st.dataframe(
                    shorts[["ticker", "company_name", "industry", "sector", "composite", "weight", "market_cap", "current_price"]]
                    .style.format({"composite": "{:.3f}", "weight": "{:.3f}", "market_cap": "{:.0f}", "current_price": "{:.1f}"}, na_rep="—"),
                    use_container_width=True,
                    height=500,
                    )
                st.caption(f"{len(shorts)} short positions across {shorts['industry'].nunique()} industries")

            # Summary by industry
            st.markdown("### Industry Exposure")
            exp = (
                book_display.groupby(["industry", "side"])["weight"]
                .sum().unstack(fill_value=0).reset_index()
            )
            exp.columns.name = None
            for col in ["L", "S"]:
                if col not in exp.columns:
                    exp[col] = 0.0
            exp["net"] = exp.get("L", 0) - exp.get("S", 0)
            st.dataframe(exp.style.format({"L": "{:.3f}", "S": "{:.3f}", "net": "{:.3f}"}, na_rep="—"), use_container_width=True)

    # ── Tab 3: Charts ────────────────────────────────────────────────────────────
    with tab_charts:
        st.subheader("Composite Score Distribution")

        fig1 = px.histogram(
            scored.reset_index(), x="composite", color="industry",
            nbins=40, opacity=0.7,
            title="Composite score distribution by industry",
            labels={"composite": "Composite Score", "industry": "Industry"},
        )
        fig1.update_layout(barmode="overlay", height=400)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Z-Score Components — Top & Bottom 30 Stocks")
        top30 = scored.sort_values("composite", ascending=False).head(15)
        bot30 = scored.sort_values("composite", ascending=True).head(15)
        sel = pd.concat([top30, bot30]).reset_index()
        sel["label"] = sel["ticker"] + " (" + sel["industry"].fillna("") + ")"

        fig2 = go.Figure()
        colors = {"z_roe": "#2196F3", "z_accr": "#4CAF50", "z_rev": "#FF9800", "z_de": "#9C27B0"}
        labels = {"z_roe": "ROE Momentum", "z_accr": "Accruals (inv)", "z_rev": "Rev CV (inv)", "z_de": "D/E Trend (inv)"}
        for col, color in colors.items():
            if col in sel.columns:
                fig2.add_trace(go.Bar(
                    name=labels[col], x=sel["label"], y=sel[col],
                    marker_color=color, opacity=0.8,
                ))
        fig2.update_layout(
            barmode="stack", height=500,
            xaxis_tickangle=-45,
            title="Stacked Z-scores: Top 15 Longs + Top 15 Shorts",
            yaxis_title="Z-Score",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Composite vs ROE")
        scatter_df = scored.reset_index().merge(
            gi.drop_duplicates("ticker")[["ticker", "roe", "market_cap", "company_name"]]
            .rename(columns={"roe": "roe_gi", "market_cap": "mktcap_gi", "company_name": "cname_gi"}),
            on="ticker", how="left"
        )
        fig3 = px.scatter(
            scatter_df,
            x="roe_gi", y="composite",
            color="industry", size="mktcap_gi",
            hover_name="cname_gi", hover_data=["ticker"],
            title="Composite score vs ROE (size = market cap)",
            labels={"roe_gi": "ROE (%)", "composite": "Composite Score", "mktcap_gi": "Market Cap"},
            opacity=0.75,
        )
        fig3.update_layout(height=500)
        st.plotly_chart(fig3, use_container_width=True)
