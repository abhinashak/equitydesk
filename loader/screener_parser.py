"""
screener_parser.py
------------------
Parses a downloaded Screener.in company HTML page and writes one
Parquet file per information section.  Ticker is always the first
column so all per-ticker files can later be unioned with DuckDB.

Usage
-----
    python screener_parser.py path/to/TICKER.html [--out-dir ./parquet]

Output files (one file per section, appended if it already exists)
-------------------------------------------------------------------
    general_info.parquet          – company-level scalar KPIs
    pros.parquet                  – list of pros (one row per item)
    cons.parquet                  – list of cons (one row per item)
    quarterly_results.parquet     – quarterly P&L table   (long format)
    profit_loss.parquet           – annual  P&L table     (long format)
    balance_sheet.parquet         – annual  BS table      (long format)
    cash_flows.parquet            – annual  CF table      (long format)
    ratios.parquet                – annual  ratios table  (long format)
    compounded_sales_growth.parquet
    compounded_profit_growth.parquet
    stock_price_cagr.parquet
    return_on_equity.parquet
    shareholding_quarterly.parquet
    shareholding_yearly.parquet

Long-format schema for financial tables
----------------------------------------
    ticker | period | metric | value

Long-format schema for CAGR / RoE tables
-----------------------------------------
    ticker | horizon | value

Shareholding schema
--------------------
    ticker | period | category | percentage
"""

import argparse
import os
import re
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from bs4 import BeautifulSoup, Tag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Strip whitespace and normalise."""
    return re.sub(r"\s+", " ", text or "").strip()


def _parse_number(text: str) -> str:
    """Remove currency symbols / commas but keep the value as a string
    so we don't lose '%' or other suffixes; caller can cast if needed."""
    return _clean(text).replace(",", "")


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write *df* to *path*, always overwriting.
    The staging directory is fresh per-ticker-per-run; merging and
    deduplication is the DAL's responsibility, not the parser's.
    """
    if df.empty:
        print(f"  [skip] {path.name} – no data extracted")
        return
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(tbl, path)
    print(f"  [ok]   {path.name}  ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Ticker extraction
# ---------------------------------------------------------------------------

def extract_ticker(soup: BeautifulSoup) -> str:
    """
    Try multiple signals to get the NSE / screener ticker symbol.
    Priority: NSE link → screener company URL → page <title>.
    """
    # 1. NSE link  href="https://www.nseindia.com/...?symbol=M%26M"
    #    URL-decode first so M%26M → M&M before the regex runs.
    nse_a = soup.find("a", href=re.compile(r"nseindia\.com.*symbol="))
    if nse_a:
        href_decoded = unquote(nse_a["href"])
        m = re.search(r"symbol=([A-Z0-9&]+)", href_decoded)
        if m:
            return m.group(1)

    # 2. Screener canonical URL embedded in hidden <input name="next">
    inp = soup.find("input", {"name": "next"})
    if inp:
        m = re.search(r"/company/([^/]+)/", inp.get("value", ""))
        if m:
            return m.group(1)

    # 3. data-url on the AI button  /company/chat/3810/ — not useful, skip

    # 4. Fallback: title tag  "... | Screener"
    title = _clean(soup.title.string) if soup.title else ""
    # e.g. "Zydus Wellness Ltd share price | About Zydus Wellness | ..."
    # Just use the first meaningful word-chunk
    parts = [p.strip() for p in title.split("|")]
    if parts:
        return re.sub(r"[^A-Z0-9&-]", "", parts[0].upper())[:20]

    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------

def parse_general_info(soup: BeautifulSoup, ticker: str) -> pd.DataFrame:
    """
    Scalar KPIs from the top-of-page summary box.
    Returns a single-row DataFrame.
    """
    row: dict = {"ticker": ticker}

    # Always present columns – filled below if found, else stay empty string
    row["bse_code"] = ""
    row["nse_symbol"] = ""

    # Company name
    h1 = soup.find("h1", class_=re.compile(r"(margin-0|h2)"))
    if h1:
        row["company_name"] = _clean(h1.get_text())

    # About paragraph
    about_div = soup.find("div", class_="about")
    if about_div:
        row["about"] = re.sub(r"\s*\[\d+\]", "", _clean(about_div.get_text())).strip()

    # BSE / NSE codes
    bse_a = soup.find("a", href=re.compile(r"bseindia\.com"))
    if bse_a:
        bse_text = _clean(bse_a.get_text())
        m = re.search(r"(\d{6})", bse_text)
        if m:
            row["bse_code"] = m.group(1)

    nse_a = soup.find("a", href=re.compile(r"nseindia\.com"))
    if nse_a:
        href_decoded = unquote(nse_a["href"])
        m = re.search(r"symbol=([A-Z0-9&]+)", href_decoded)
        if m:
            row["nse_symbol"] = m.group(1)
        else:
            # Fallback: read visible text  "NSE: M&M"
            row["nse_symbol"] = (_clean(nse_a.get_text())
                                 .replace("NSE:", "").strip())

    # Sector / Industry from peer-comparison breadcrumb
    peer_section = soup.find("section", id="peers")
    if peer_section:
        breadcrumb_links = peer_section.find_all("a", href=re.compile(r"/market/"))
        labels = [_clean(a.get_text()) for a in breadcrumb_links]
        if len(labels) >= 1:
            row["broad_sector"] = labels[0]
        if len(labels) >= 2:
            row["sector"] = labels[1]
        if len(labels) >= 3:
            row["broad_industry"] = labels[2]
        if len(labels) >= 4:
            row["industry"] = labels[3]

    # Top KPI ratios  (#top-ratios ul li)
    ratios_ul = soup.find("ul", id="top-ratios")
    if ratios_ul:
        for li in ratios_ul.find_all("li"):
            name_span = li.find("span", class_="name")
            num_span = li.find("span", class_="number")
            if name_span and num_span:
                key = re.sub(r"[^a-z0-9_]", "_", _clean(name_span.get_text()).lower()).strip("_")
                row[key] = _parse_number(num_span.get_text())

    # Current price (shown near the top, outside the ratios list)
    price_span = soup.find("span", class_="number", string=re.compile(r"^\d[\d,\.]+$"))
    if price_span and "current_price" not in row:
        row["current_price"] = _parse_number(price_span.get_text())

    return pd.DataFrame([row])


def parse_pros_cons(soup: BeautifulSoup, ticker: str):
    """Returns (pros_df, cons_df)."""

    def _extract(cls: str) -> pd.DataFrame:
        div = soup.find("div", class_=cls)
        if not div:
            return pd.DataFrame()
        items = [_clean(li.get_text()) for li in div.find_all("li")]
        if not items:
            return pd.DataFrame()
        return pd.DataFrame({"ticker": ticker, "item": items})

    return _extract("pros"), _extract("cons")


def _parse_financial_table(section: Tag, ticker: str) -> pd.DataFrame:
    """
    Generic parser for the wide financial tables used by
    Quarterly Results, P&L, Balance Sheet, Cash Flows, Ratios.

    Returns a long-format DataFrame: ticker | period | metric | value
    """
    table = section.find("table", class_="data-table")
    if not table:
        return pd.DataFrame()

    # Header → periods
    header_row = table.find("thead").find("tr")
    periods = []
    for th in header_row.find_all("th"):
        text = _clean(th.get_text())
        if text:
            periods.append(text)

    # Body → rows
    records = []
    for tr in table.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue

        # First cell is the metric name (may contain a button label)
        metric = _clean(tds[0].get_text())
        # Strip trailing "+  " added by the expand button
        metric = re.sub(r"\s*\+\s*$", "", metric).strip()

        # Skip non-data rows (e.g. Raw PDF links)
        if not metric or metric.lower() in ("raw pdf",):
            continue

        values = [_parse_number(td.get_text()) for td in tds[1:]]

        for period, value in zip(periods, values):
            records.append({"ticker": ticker, "period": period,
                            "metric": metric, "value": value})

    return pd.DataFrame(records)


def parse_quarterly_results(soup: BeautifulSoup, ticker: str) -> pd.DataFrame:
    section = soup.find("section", id="quarters")
    return _parse_financial_table(section, ticker) if section else pd.DataFrame()


def parse_profit_loss(soup: BeautifulSoup, ticker: str) -> pd.DataFrame:
    section = soup.find("section", id="profit-loss")
    return _parse_financial_table(section, ticker) if section else pd.DataFrame()


def parse_balance_sheet(soup: BeautifulSoup, ticker: str) -> pd.DataFrame:
    section = soup.find("section", id="balance-sheet")
    return _parse_financial_table(section, ticker) if section else pd.DataFrame()


def parse_cash_flows(soup: BeautifulSoup, ticker: str) -> pd.DataFrame:
    section = soup.find("section", id="cash-flow")
    return _parse_financial_table(section, ticker) if section else pd.DataFrame()


def parse_ratios(soup: BeautifulSoup, ticker: str) -> pd.DataFrame:
    section = soup.find("section", id="ratios")
    return _parse_financial_table(section, ticker) if section else pd.DataFrame()


def parse_ranges_tables(soup: BeautifulSoup, ticker: str):
    """
    The four compact "ranges" tables that appear after P&L:
      Compounded Sales Growth | Compounded Profit Growth |
      Stock Price CAGR        | Return on Equity

    Returns a dict of { table_title → DataFrame(ticker, horizon, value) }
    """
    results = {}

    for tbl in soup.find_all("table", class_="ranges-table"):
        th = tbl.find("th")
        if not th:
            continue
        title = _clean(th.get_text())

        records = []
        for tr in tbl.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) == 2:
                horizon = _clean(tds[0].get_text()).rstrip(":")
                value = _parse_number(tds[1].get_text())
                records.append({"ticker": ticker, "horizon": horizon, "value": value})

        if records:
            results[title] = pd.DataFrame(records)

    return results


def _parse_shareholding_table(table: Tag, ticker: str) -> pd.DataFrame:
    """Parse one shareholding table (quarterly or yearly)."""
    if not table:
        return pd.DataFrame()

    # Header
    header_tr = table.find("thead").find("tr")
    periods = [_clean(th.get_text()) for th in header_tr.find_all("th") if _clean(th.get_text())]

    records = []
    for tr in table.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        category = _clean(tds[0].get_text())
        # Strip trailing "+"
        category = re.sub(r"\s*\+\s*$", "", category).strip()

        if not category:
            continue

        values = [_clean(td.get_text()) for td in tds[1:]]
        for period, value in zip(periods, values):
            records.append({"ticker": ticker, "period": period,
                            "category": category, "value": value})

    return pd.DataFrame(records)


def parse_shareholding(soup: BeautifulSoup, ticker: str):
    """Returns (quarterly_df, yearly_df)."""
    section = soup.find("section", id="shareholding")
    if not section:
        return pd.DataFrame(), pd.DataFrame()

    # Quarterly table
    q_div = section.find("div", id="quarterly-shp")
    quarterly = _parse_shareholding_table(
        q_div.find("table", class_="data-table") if q_div else None, ticker
    )

    # Yearly table
    y_div = section.find("div", id="yearly-shp")
    yearly = _parse_shareholding_table(
        y_div.find("table", class_="data-table") if y_div else None, ticker
    )

    return quarterly, yearly


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

RANGES_TABLE_FILE_MAP = {
    "Compounded Sales Growth":   "compounded_sales_growth.parquet",
    "Compounded Profit Growth":  "compounded_profit_growth.parquet",
    "Stock Price CAGR":          "stock_price_cagr.parquet",
    "Return on Equity":          "return_on_equity.parquet",
}


def process_file(html_path: str, out_dir: str = ".") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {html_path}")

    with open(html_path, "r", encoding="utf-8", errors="replace") as fh:
        soup = BeautifulSoup(fh, "lxml")

    ticker = extract_ticker(soup)
    print(f"Ticker     : {ticker}")
    print(f"Output dir : {out.resolve()}")
    print("-" * 60)

    # 1. General info
    _write_parquet(parse_general_info(soup, ticker),
                   out / "general_info.parquet")

    # 2. Pros & Cons
    pros_df, cons_df = parse_pros_cons(soup, ticker)
    _write_parquet(pros_df, out / "pros.parquet")
    _write_parquet(cons_df, out / "cons.parquet")

    # 3. Financial tables
    _write_parquet(parse_quarterly_results(soup, ticker),
                   out / "quarterly_results.parquet")
    _write_parquet(parse_profit_loss(soup, ticker),
                   out / "profit_loss.parquet")
    _write_parquet(parse_balance_sheet(soup, ticker),
                   out / "balance_sheet.parquet")
    _write_parquet(parse_cash_flows(soup, ticker),
                   out / "cash_flows.parquet")
    _write_parquet(parse_ratios(soup, ticker),
                   out / "ratios.parquet")

    # 4. CAGR / RoE compact tables
    ranges = parse_ranges_tables(soup, ticker)
    for title, filename in RANGES_TABLE_FILE_MAP.items():
        df = ranges.get(title, pd.DataFrame())
        _write_parquet(df, out / filename)

    # 5. Shareholding
    shp_q, shp_y = parse_shareholding(soup, ticker)
    _write_parquet(shp_q, out / "shareholding_quarterly.parquet")
    _write_parquet(shp_y, out / "shareholding_yearly.parquet")

    print(f"\nDone for {ticker}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse Screener.in HTML pages → Parquet files"
    )
    parser.add_argument(
        "html_files",
        nargs="+",
        help="One or more downloaded HTML files (e.g. ZYDUSWELL.html INFY.html)",
    )
    parser.add_argument(
        "--out-dir",
        default="./parquet_output",
        help="Directory where Parquet files are written (default: ./parquet_output)",
    )
    args = parser.parse_args()

    for html_file in args.html_files:
        process_file(html_file, args.out_dir)

    print("\nAll files processed.")
    print(f"Query example (DuckDB):")
    print(f"  SELECT * FROM '{args.out_dir}/profit_loss.parquet'")
    print(f"  WHERE ticker = 'ZYDUSWELL' AND metric = 'Sales';")


if __name__ == "__main__":
    main()