"""HTML/CSS helpers for the portfolio dashboard."""
from html import escape
import pandas as pd
import streamlit as st

def render_trend_banner(ticker: str, last: float, sma50: float,
                         label: str = "Market Trend", panic: bool = False) -> None:
    """Compact SMA50 trend banner (sized to sit in one of three columns).
    green = above SMA50, red = below. panic=True uses 'EXIT PANIC' wording."""
    gap = (last - sma50) / sma50 * 100
    if last < sma50:
        kicker   = "Market Risk Alert" if panic else f"{label} Alert"
        headline = f"{ticker} is below its SMA50"
        tail     = "— EXIT PANIC" if panic else "— trend weakening"
        icon, icon_sz, pad = "🚨", "2rem", "16px 18px"
        bg, border = "linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%)", "#ef4444"
        kick_c = "#fca5a5"
    else:
        kicker   = label
        headline = f"{ticker} is above its SMA50"
        tail     = "— trend intact"
        icon, icon_sz, pad = "✅", "1.8rem", "16px 18px"
        bg, border = "linear-gradient(135deg, #14532d 0%, #166534 100%)", "#22c55e"
        kick_c = "#86efac"

    def _chip(lbl, val, color="#fff"):
        return (f'<span style="white-space:nowrap;">{lbl}: '
                f'<strong style="color:{color}">{val}</strong></span>')

    html = (
        f'<div style="background:{bg};border:2px solid {border};border-radius:12px;'
        f'padding:{pad};margin-bottom:16px;min-height:150px;'
        f'display:flex;align-items:center;gap:14px;">'
        f'<span style="font-size:{icon_sz};">{icon}</span><div>'
        f'<div style="color:{kick_c};font-size:0.74rem;font-weight:700;'
        f'letter-spacing:0.12em;text-transform:uppercase;">{kicker}</div>'
        f'<div style="color:#fff;font-size:1.05rem;font-weight:800;margin:2px 0 0;">{headline}</div>'
        f'<div style="color:#fff;font-size:1.05rem;font-weight:800;margin:0 0 4px;">{tail}</div>'
        f'<div style="color:{kick_c};font-size:0.82rem;display:flex;flex-wrap:wrap;gap:2px 10px;">'
        f'{_chip(ticker, f"₹{last:,.2f}")}'
        f'{_chip("SMA50", f"₹{sma50:,.2f}")}'
        f'{_chip("Gap", f"{gap:+.2f}%", kick_c)}'
        f'</div></div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def inject_portfolio_css() -> None:
    """Install the portfolio-specific CSS classes once per Streamlit rerun."""
    st.markdown("""
    <style>
    .winner-row  { background-color: rgba(22,163,74,0.10) !important; }
    .loser-row   { background-color: rgba(220,38,38,0.10) !important; }
    .sector-pill { display:inline-block; padding:2px 10px; border-radius:9999px; font-size:0.72rem; font-weight:600; }
    .period-chip { display:inline-block; padding:3px 10px; border-radius:6px; font-size:0.78rem; font-weight:600; margin:2px; }
    .portfolio-data-card { border:1px solid rgba(128,128,128,.35); border-radius:12px; padding:12px 14px; margin:6px 0; }
    .portfolio-data-card .card-label { opacity:.72; font-size:.76rem; }
    .portfolio-data-card .card-value { font-size:1.05rem; font-weight:700; }
    </style>
    """, unsafe_allow_html=True)

def render_dataframe_cards(df: pd.DataFrame, title_column: str, value_columns: list[str],
                           *, subtitle_column: str | None = None) -> None:
    """Render each DataFrame row as a compact HTML card. Missing columns are ignored."""
    if df is None or df.empty:
        st.caption("No data to display.")
        return
    cols = [c for c in value_columns if c in df.columns]
    if title_column not in df.columns:
        raise ValueError(f"title_column {title_column!r} is not present in DataFrame")
    for _, row in df.iterrows():
        title = escape(str(row[title_column]))
        subtitle = (f'<div class="card-label">{escape(str(row[subtitle_column]))}</div>'
                    if subtitle_column and subtitle_column in df.columns else "")
        values = "".join(
            f'<div><span class="card-label">{escape(str(col))}</span> '
            f'<span class="card-value">{escape(str(row[col]))}</span></div>'
            for col in cols
        )
        st.markdown(
            f'<div class="portfolio-data-card"><strong>{title}</strong>{subtitle}'
            f'<div style="display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:8px">{values}</div></div>',
            unsafe_allow_html=True,
        )
