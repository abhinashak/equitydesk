"""fasta_report.py - renders the weekly-rotation report as one self-contained HTML page.

Input is the data the backtest already has (DataFrames / dicts); nothing is parsed from fasta.txt.

    from fasta_report import render_html_report
    render_html_report(trades_df, picks_df, "outputs/fasta/fasta.html",
                       nifty_weekly_ret=..., nifty_q_summary=..., nifty_y_summary=...,
                       weekly_capital=WEEKLY_POSITION_CAPITAL, picks_title="NEXT-WEEK PICKS ...")

trades_df : one row per trade (ticker, buy_date, sell_date, ..., net_profit_rupees, position_capital, ...)
picks_df  : ticker, entry_date, price, confidence (any extra columns are shown as they are)
Weekly / quarterly / yearly PnL and the footer metrics are derived from trades_df with the same formulas as
fasta.py, unless you pass them in (weekly_pnl, quarterly_pnl, yearly_pnl as Series, metrics as [(label, value)]).
"""
import html as _html
import numpy as np
import pandas as pd
from pathlib import Path

_CSS = """
:root{--bg:#f6f7f9;--fg:#1c2330;--muted:#667085;--card:#fff;--border:#e3e6eb;--accent:#2563eb;--pos:#0f8a4b;--neg:#c92a2a;--posbg:#e6f6ec;--negbg:#fdeaea;--zebra:#f9fafb}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0f141b;--fg:#e6e9ef;--muted:#98a2b3;--card:#171e28;--border:#2a3441;--accent:#6ea8fe;--pos:#4cc38a;--neg:#ff7b7b;--posbg:#12301f;--negbg:#3a1a1a;--zebra:#1b2430}}
:root[data-theme="dark"]{--bg:#0f141b;--fg:#e6e9ef;--muted:#98a2b3;--card:#171e28;--border:#2a3441;--accent:#6ea8fe;--pos:#4cc38a;--neg:#ff7b7b;--posbg:#12301f;--negbg:#3a1a1a;--zebra:#1b2430}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:20px 16px 60px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:34px 0 10px}
.sub{color:var(--muted);margin:0 0 14px}
nav{position:sticky;top:0;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--border);display:flex;gap:16px;flex-wrap:wrap;z-index:5;align-items:center}
nav a{color:var(--accent);text-decoration:none;font-weight:600}nav .sp{flex:1}
button{background:var(--card);color:var(--fg);border:1px solid var(--border);border-radius:8px;padding:6px 11px;font:inherit;cursor:pointer}
button:hover{border-color:var(--accent)}
input[type=search]{background:var(--card);color:var(--fg);border:1px solid var(--border);border-radius:8px;padding:6px 10px;font:inherit;min-width:230px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:11px 13px}
.card .k{color:var(--muted);font-size:12px}.card .v{font-weight:700;font-variant-numeric:tabular-nums}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.box{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;text-align:left;white-space:nowrap}
th{font-size:12px;color:var(--muted);border-bottom:1px solid var(--border)}
tbody tr:nth-child(even){background:var(--zebra)}
td.num,th.num{text-align:right}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600}
.b-pos{background:var(--posbg);color:var(--pos)}.b-neg{background:var(--negbg);color:var(--neg)}.b-neu{background:var(--zebra);color:var(--muted);border:1px solid var(--border)}
details{background:var(--card);border:1px solid var(--border);border-radius:10px;margin:8px 0}
summary{cursor:pointer;padding:10px 13px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;list-style:none}
summary::-webkit-details-marker{display:none}
summary::before{content:"\\25B8";color:var(--muted)}details[open]>summary::before{content:"\\25BE"}
summary .wk{font-weight:700}summary .meta{color:var(--muted);margin-left:auto}
details .inner{padding:0 13px 12px}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}
.chip{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:3px 10px;font-weight:600}
svg{width:100%;height:auto;display:block}.lg{display:flex;gap:16px;color:var(--muted);margin-top:6px;font-size:12px}
.hide{display:none}.note{color:var(--muted);font-size:12px}
"""

_JS = """
function allDetails(open){document.querySelectorAll('#weeks details').forEach(function(d){if(!d.classList.contains('hide'))d.open=open})}
function filterWeeks(q){q=q.trim().toLowerCase();var shown=0;document.querySelectorAll('#weeks details').forEach(function(d){var ok=!q||d.textContent.toLowerCase().indexOf(q)>-1;d.classList.toggle('hide',!ok);if(ok)shown++;if(q&&ok)d.open=true});document.getElementById('shown').textContent=shown}
function copyTickers(){var e=document.getElementById('tickers');if(navigator.clipboard){navigator.clipboard.writeText(e.value)}else{e.select();document.execCommand('copy')}var b=document.getElementById('copybtn');b.textContent='Copied';setTimeout(function(){b.textContent='Copy tickers'},1500)}
function toggleTheme(){var r=document.documentElement;var dark=r.getAttribute('data-theme')==='dark'||(!r.getAttribute('data-theme')&&matchMedia('(prefers-color-scheme: dark)').matches);r.setAttribute('data-theme',dark?'light':'dark')}
"""

_HELPER_COLS = ['sort_date', 'week_period', 'quarter_period', 'year_period']
_esc = _html.escape


# ---------- formatting helpers ----------
def _signed(v, suffix="%"):
    cls = 'pos' if v > 0 else ('neg' if v < 0 else '')
    return f'<span class="{cls}">{v:+.2f}{suffix}</span>'


def _cell(col, v):
    if isinstance(v, (bool, np.bool_)):
        return f'<td>{"&#10003;" if v else "&ndash;"}</td>'
    if col == 'exit_reason':
        s = str(v)
        return f'<td><span class="badge {"b-neg" if "STOP" in s else "b-neu"}">{_esc(s.replace("STOCK_", "").replace("_", " "))}</span></td>'
    if col == 'action':
        return f'<td><span class="badge {"b-pos" if str(v) == "BUY" else "b-neu"}">{_esc(str(v))}</span></td>'
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        if pd.isna(v):
            return '<td class="num">&ndash;</td>'
        text = f"{v:,.2f}" if isinstance(v, (float, np.floating)) else f"{int(v):,d}"
        cls = ''
        if any(k in col for k in ('pnl', 'profit')):
            cls = ' pos' if v > 0 else (' neg' if v < 0 else '')
        return f'<td class="num{cls}">{text}</td>'
    return f'<td>{_esc(str(v))}</td>'


def _table(df):
    if df is None or df.empty:
        return ''
    head = ''.join(f'<th class="{"num" if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) else ""}">{_esc(str(c))}</th>'
                   for c in df.columns)
    cols = list(df.columns)
    body = ''.join('<tr>' + ''.join(_cell(c, v) for c, v in zip(cols, row)) + '</tr>' for row in df.itertuples(index=False, name=None))
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _perf_table(rows, label):
    body = ''.join(f'<tr><td>{_esc(str(k))}</td><td class="num">{_signed(p)}</td><td class="num">{_signed(b)}</td>'
                   f'<td class="num">{_signed(p - b, " pts")}</td></tr>' for k, p, b in rows)
    return (f'<div class="box scroll"><table><thead><tr><th>{label}</th><th class="num">Portfolio</th>'
            f'<th class="num">Nifty 50</th><th class="num">Difference</th></tr></thead><tbody>{body}</tbody></table></div>')


def _chart(points):
    """points: [(portfolio_weekly_pct, nifty_weekly_pct)] -> cumulative SUM of weekly % (as the quarterly/yearly tables)."""
    if len(points) < 2:
        return ''
    port, nift = [0.0], [0.0]
    for p, b in points:
        port.append(port[-1] + p)
        nift.append(nift[-1] + b)
    W, Ht, L, R, T, B = 900, 270, 62, 12, 12, 26
    lo, hi = min(min(port), min(nift), 0.0), max(max(port), max(nift), 0.0)
    if hi - lo < 1e-9:
        hi = lo + 1

    def xy(series):
        return ' '.join(f'{L + (W - L - R) * k / (len(series) - 1):.1f},{T + (Ht - T - B) * (1 - (v - lo) / (hi - lo)):.1f}'
                        for k, v in enumerate(series))
    ticks = ''
    for k in range(5):
        frac = k / 4
        y = T + (Ht - T - B) * (1 - frac)
        ticks += (f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--border)"/>'
                  f'<text x="{L - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="var(--muted)">{lo + (hi - lo) * frac:,.0f}%</text>')
    return (f'<div class="box"><svg viewBox="0 0 {W} {Ht}" role="img" aria-label="Cumulative weekly PnL, portfolio vs Nifty 50">{ticks}'
            f'<polyline fill="none" stroke="var(--muted)" stroke-width="1.6" stroke-dasharray="5 4" points="{xy(nift)}"/>'
            f'<polyline fill="none" stroke="var(--accent)" stroke-width="2" points="{xy(port)}"/></svg>'
            f'<div class="lg"><span><b style="color:var(--accent)">&#9473;</b> Portfolio</span><span><b>&#9476;</b> Nifty 50</span>'
            f'<span>Cumulative sum of weekly returns (%)</span></div></div>')


def _metric_card(label, value):
    cls = ''
    if value[:1] in '+-' and value[1:2].isdigit():
        cls = 'pos' if value[0] == '+' else 'neg'
    elif 'GAIN' in label.upper():
        cls = 'pos'
    elif 'LOSS' in label.upper():
        cls = 'neg'
    size = '15px' if len(value) > 22 else '20px'
    return f'<div class="card"><div class="k">{_esc(label)}</div><div class="v {cls}" style="font-size:{size}">{_esc(value)}</div></div>'


# ---------- derived numbers (same formulas as fasta.py) ----------
def _derive(trades, weekly_capital):
    """Returns (display_trades, week_groups, weekly_pnl, quarterly_pnl, yearly_pnl, metrics)."""
    t = trades.copy()
    if 'week_period' not in t.columns:
        t['week_period'] = pd.to_datetime(t['buy_date']).dt.to_period('W')
    t = t.sort_values(['week_period'], kind='stable')   # one block per week, in time order

    if 'net_profit_rupees' in t.columns:
        weekly_pnl = t.groupby('week_period')['net_profit_rupees'].sum() / weekly_capital * 100.0
    else:
        weekly_pnl = t.groupby('week_period')['pnl_pct'].mean()
    quarterly_pnl = weekly_pnl.groupby(weekly_pnl.index.asfreq('Q')).sum()
    yearly_pnl = weekly_pnl.groupby(weekly_pnl.index.asfreq('Y')).sum()

    metrics = []
    total_trades = len(t)
    overall = weekly_pnl.sum()
    metrics.append(("OVERALL CUMULATIVE REALISED PnL ACROSS ALL WEEKS", f"{overall:+.2f}%"))
    metrics.append(("AVERAGE PnL PER COMPLETED TRADE", f"{(overall / total_trades if total_trades else 0.0):+.2f}%"))
    metrics.append(("TOTAL COMPLETED TRADES EXECUTED", f"{total_trades}"))
    if {'total_transaction_cost', 'gross_profit_rupees', 'net_profit_rupees', 'position_capital'}.issubset(t.columns):
        net_total = t['net_profit_rupees'].sum()
        deployed = t.groupby('week_period')['position_capital'].sum().sum()
        metrics += [
            ("TOTAL TRANSACTION COSTS (₹)", f"{t['total_transaction_cost'].sum():,.2f}"),
            ("TOTAL GROSS ABSOLUTE PnL (₹)", f"{t['gross_profit_rupees'].sum():+,.2f}"),
            ("TOTAL NET ABSOLUTE PnL (₹)", f"{net_total:+,.2f}"),
            ("AVERAGE NET ABSOLUTE PnL / TRADE (₹)", f"{t['net_profit_rupees'].mean():+,.2f}"),
            ("TOTAL CAPITAL DEPLOYED ACROSS TRADES (₹)", f"{deployed:,.2f}"),
            ("NET RETURN ON TOTAL CAPITAL TRADED", f"{(net_total / deployed * 100.0 if deployed else 0.0):+.2f}%"),
        ]
    if 'net_pnl_pct' in t.columns:
        metrics.append(("AVERAGE NET PnL PER TRADE", f"{t['net_pnl_pct'].mean():+.2f}%"))
    if len(weekly_pnl) and 'net_profit_rupees' in t.columns:
        rupees = t.groupby('week_period')['net_profit_rupees'].sum()
        gw, lw = weekly_pnl.idxmax(), weekly_pnl.idxmin()
        metrics.append(("MAXIMUM GAIN WEEK", f"{gw} | {weekly_pnl[gw]:+.2f}% | ₹{rupees[gw]:+,.2f}"))
        metrics.append(("MAXIMUM LOSS WEEK", f"{lw} | {weekly_pnl[lw]:+.2f}% | ₹{rupees[lw]:+,.2f}"))

    display = t.drop(columns=[c for c in _HELPER_COLS if c in t.columns])
    groups = [(wk, display.loc[g.index]) for wk, g in t.groupby('week_period', sort=True)]
    return display, groups, weekly_pnl, quarterly_pnl, yearly_pnl, metrics


def render_html_report(trades_df, picks_df, output_path, nifty_weekly_ret=None, nifty_q_summary=None,
                       nifty_y_summary=None, weekly_capital=100_000.0, title="Weekly Rotation Report",
                       picks_title=None, weekly_pnl=None, quarterly_pnl=None, yearly_pnl=None, metrics=None):
    nifty_weekly_ret, nifty_q_summary, nifty_y_summary = nifty_weekly_ret or {}, nifty_q_summary or {}, nifty_y_summary or {}
    has_trades = trades_df is not None and not trades_df.empty
    groups, chart_points, rows_q, rows_y = [], [], [], []
    if has_trades:
        _, groups, w_pnl, q_pnl, y_pnl, m_list = _derive(trades_df, weekly_capital)
        w_pnl = w_pnl if weekly_pnl is None else weekly_pnl
        q_pnl = q_pnl if quarterly_pnl is None else quarterly_pnl
        y_pnl = y_pnl if yearly_pnl is None else yearly_pnl
        metrics = m_list if metrics is None else metrics
        rows_q = [(k, float(v), float(nifty_q_summary.get(k, 0.0))) for k, v in q_pnl.items()]
        rows_y = [(k, float(v), float(nifty_y_summary.get(k, 0.0))) for k, v in y_pnl.items()]
        chart_points = [(float(w_pnl.get(wk, 0.0)), float(nifty_weekly_ret.get(wk, 0.0))) for wk, _ in groups]
    metrics = metrics or []

    out = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>{_esc(title)}</title><style>{_CSS}</style></head><body><div class="wrap">',
           f'<h1>{_esc(title)}</h1><p class="sub">Generated from the backtest DataFrames</p>',
           '<nav><a href="#picks">Picks</a><a href="#summary">Summary</a><a href="#periods">Quarterly &amp; yearly</a><a href="#weeks">Weeks</a>'
           '<span class="sp"></span><button onclick="toggleTheme()">Light / dark</button></nav>',
           f'<h2 id="picks">{_esc(picks_title or "Picks")}</h2>']

    if picks_df is not None and not picks_df.empty:
        p = picks_df.sort_values('confidence', ascending=False).reset_index(drop=True) if 'confidence' in picks_df.columns else picks_df
        out.append('<div class="box">' + _table(p) + '</div>')
        tickers = p['ticker'].astype(str).tolist() if 'ticker' in p.columns else []
        if tickers:
            out.append('<div class="chips">' + ''.join(f'<span class="chip">{_esc(t)}</span>' for t in tickers) + '</div>')
            out.append(f'<textarea id="tickers" class="hide">{_esc(chr(10).join(tickers))}</textarea>'
                       '<button id="copybtn" onclick="copyTickers()">Copy tickers</button>')
    else:
        out.append('<div class="box">No active buy picks for the current week matching criteria.</div>')

    out.append('<h2 id="summary">Summary</h2>')
    if metrics:
        out.append('<div class="cards">' + ''.join(_metric_card(k, v) for k, v in metrics) + '</div>')
    else:
        out.append('<div class="box">No completed transactions generated with current rules.</div>')
    out.append(_chart(chart_points))

    out.append('<h2 id="periods">Quarterly &amp; yearly performance</h2>')
    if rows_y:
        out.append(_perf_table(rows_y, 'Year'))
    if rows_q:
        out.append('<div style="height:12px"></div>' + _perf_table(rows_q, 'Quarter'))

    out.append(f'<h2>Weekly trades</h2><div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:8px">'
               f'<input type="search" placeholder="Filter by ticker or week..." oninput="filterWeeks(this.value)">'
               f'<button onclick="allDetails(true)">Expand all</button><button onclick="allDetails(false)">Collapse all</button>'
               f'<span class="note"><span id="shown">{len(groups)}</span> of {len(groups)} weeks shown</span></div><div id="weeks">')
    pnl_by_week = {wk: float(w_pnl.get(wk, 0.0)) for wk, _ in groups} if has_trades else {}
    for k, (wk, g) in enumerate(reversed(groups)):        # newest first, latest three open
        pnl = pnl_by_week[wk]
        nif = nifty_weekly_ret.get(wk, 0.0)
        badge = 'b-pos' if pnl > 0 else ('b-neg' if pnl < 0 else 'b-neu')
        out.append(f'<details{" open" if k < 3 else ""}><summary><span class="wk">{_esc(str(wk))}</span>'
                   f'<span class="badge {badge}">{pnl:+.2f}%</span>'
                   f'<span class="meta">{len(g)} trades &middot; Nifty {nif:+.2f}%</span></summary>'
                   f'<div class="inner">{_table(g)}</div></details>')
    out.append('</div></div><script>' + _JS + '</script></body></html>')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(''.join(out), encoding='utf-8')
    return output_path
