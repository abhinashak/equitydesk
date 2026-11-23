"""Yahoo Finance RSS headlines for a set of picked stocks, rendered as Streamlit cards.

Usage (from the FASTA page):
    from utils.yahoo_rss_news import show_news
    show_news(picks_df)          # picks_df needs a 'ticker' column (e.g. COFORGE.NS)
"""
from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import streamlit as st

__all__ = ["show_news", "fetch_pick_news"]


def _bare(ticker):
    """Ticker without the exchange suffix (RELIANCE.NS -> RELIANCE)."""
    return re.sub(r"\.(NS|BO)$", "", ticker, flags=re.IGNORECASE)


YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=IN&lang=en-IN"
_IST = timezone(timedelta(hours=5, minutes=30))
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _yahoo_symbol(ticker):
    t = str(ticker).strip().upper()
    return t if "." in t else f"{t}.NS"


def _fetch_rss_items(symbol, timeout=10):
    """Headlines for ONE symbol (one request per stock, so every story is tied to the stock it came from)."""
    req = urllib.request.Request(YAHOO_RSS_URL.format(sym=urllib.parse.quote(symbol)),
                                 headers={"User-Agent": "Mozilla/5.0 (compatible; fasta-news/1.0)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        root = ET.fromstring(resp.read())
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        if not title or not link:
            continue
        desc = html.unescape(re.sub(r"<[^>]+>", " ", it.findtext("description") or ""))
        try:
            when = parsedate_to_datetime(it.findtext("pubDate")).astimezone(_IST)
        except Exception:
            when = None
        items.append({"title": html.unescape(title), "link": link,
                      "summary": re.sub(r"\s+", " ", desc).strip(), "when": when})
    items.sort(key=lambda x: x["when"] or _EPOCH, reverse=True)
    return items


@st.cache_data(ttl=600, show_spinner=False)
def fetch_pick_news(tickers: tuple):
    """{ticker: (items, error)} fetched in parallel; cached for 10 minutes."""
    def one(t):
        try:
            return t, _fetch_rss_items(_yahoo_symbol(t)), None
        except Exception as exc:
            return t, [], f"{type(exc).__name__}: {exc}"
    with ThreadPoolExecutor(max_workers=6) as ex:
        return {t: (items, err) for t, items, err in ex.map(one, tickers)}


def _ago(when):
    if when is None:
        return ""
    secs = (datetime.now(_IST) - when).total_seconds()
    if secs < 3600:
        return f"{max(int(secs // 60), 1)}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    return f"{int(secs // 86400)}d ago"


def _md_safe(text):
    """Keep headlines from being read as markdown / LaTeX."""
    return (text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
            .replace("$", "\\$").replace("*", "\\*").replace("_", "\\_"))


def _news_card(it, tickers):
    with st.container(border=True):
        st.markdown(f"**[{_md_safe(it['title'])}]({it['link'].replace(')', '%29')})**")
        meta = " · ".join(x for x in (
            " ".join(f"`{_bare(t)}`" for t in tickers),
            _ago(it["when"]),
            it["when"].strftime("%a %d %b, %H:%M IST") if it["when"] else "") if x)
        st.caption(meta)
        if it["summary"]:
            s = it["summary"]
            st.caption(_md_safe(s if len(s) <= 240 else s[:237].rstrip() + "…"))


def show_news(picks):
    st.subheader("📰 News on this week's picks")
    tickers = tuple(dict.fromkeys(picks['ticker'].astype(str)))
    c1, c2, c3 = st.columns([2, 2, 1])
    per = c1.slider("Headlines per stock", 1, 15, 5, key="fasta_news_per")
    max_age = c2.selectbox("Not older than", [0, 1, 3, 7, 30], index=3, key="fasta_news_age",
                           format_func=lambda d: "Any age" if d == 0 else f"{d} day{'s' if d > 1 else ''}")
    c3.write("")
    if c3.button("↻ Refresh", key="fasta_news_refresh", help="Re-fetch the feeds (otherwise cached for 10 minutes)."):
        fetch_pick_news.clear()

    with st.spinner("Fetching Yahoo Finance headlines…"):
        raw = fetch_pick_news(tickers)

    floor = datetime.now(_IST) - timedelta(days=max_age) if max_age else None
    per_stock, errors = {}, {}
    for t in tickers:
        items, err = raw.get(t, ([], "no response"))
        if err:
            errors[t] = err
        items = [i for i in items if floor is None or i["when"] is None or i["when"] >= floor]
        per_stock[t] = items[:per]

    merged = {}  # one card per story, even if Yahoo lists it under several picks
    for t, items in per_stock.items():
        for i in items:
            merged.setdefault(i["link"], {"item": i, "tickers": []})["tickers"].append(t)
    latest = sorted(merged.values(), key=lambda m: m["item"]["when"] or _EPOCH, reverse=True)

    st.caption(f"{len(latest)} stories across {sum(1 for v in per_stock.values() if v)} of {len(tickers)} picks "
               f"· source: Yahoo Finance RSS · refreshed every 10 min")
    t_all, t_by = st.tabs([f"Latest ({len(latest)})", "By stock"])
    with t_all:
        if not latest:
            st.info("No headlines in this window. Try a longer 'Not older than' range.")
        for m in latest:
            _news_card(m["item"], m["tickers"])
    with t_by:
        for t in tickers:
            items = per_stock[t]
            with st.expander(f"{_bare(t)}  ·  {len(items)} headline{'s' if len(items) != 1 else ''}",
                             expanded=bool(items) and t == tickers[0]):
                if t in errors:
                    st.warning(f"Could not load the feed for {t}: {errors[t]}")
                elif not items:
                    st.caption("No headlines in this window.")
                for i in items:
                    _news_card(i, [])
    if errors:
        st.caption(f"Feed errors for {len(errors)} stock(s): " + ", ".join(_bare(t) for t in errors)
                   + ". Use ↻ Refresh to retry (errors are cached for 10 minutes otherwise).")