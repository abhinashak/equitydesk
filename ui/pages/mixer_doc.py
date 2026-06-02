# mixer_doc.py

GATE_LOGIC_MD = """
**The core philosophy:** A quality stock is already priced. The opportunity is when it falls.
The engine therefore separates *what the business is* from *what it currently costs*.

---
### Gate 1 — Business Quality (Is this a genuinely good business?)
| Gate | Positive (PASS) | Negative (FAIL) |
|---|---|---|
| 1a ROCE | ≥ 18% | < 8% |
| 1b Free Cash Flow | FCF > 0 | FCF < 0 |
| 1c Profit CAGR | 3Y CAGR ≥ 15% | 3Y CAGR < 0% |
| 1d CFO vs Net Profit | CFO > Net Profit (real earnings) | CFO < 50% of Net Profit |
| 1e Debt/Equity | D/E < 0.5 | D/E > 1.5 |

---
### Gate 2 — Valuation (Is it cheap right now?)
| Gate | Positive (PASS) | Negative (FAIL) |
|---|---|---|
| 2a P/E vs Sector | Stock P/E ≥ 15% below sector median | Stock P/E > 30% above sector median |
| 2b PEG Ratio | PEG < 1.0 (growing faster than priced) | PEG > 2.5 |
| 2c Price/Book | P/B < 3.0 | P/B > 8.0 |

**PE Discount %** = (Sector median P/E − Stock P/E) ÷ Sector median P/E × 100.
A positive number means the stock is cheaper than its sector peers.

---
### Gate 3 — Timing (Is the business still accelerating?)
| Gate | Positive | Negative |
|---|---|---|
| 3a Sales Momentum | Q0 > Q1 > Q2 (3 consecutive quarters up) | Q0 < Q1 |
| 3b Profit Acceleration | Latest profit > prior quarter | Profit declining |
| 3c OPM Expansion | Latest OPM > prior quarter | OPM squeezed > 3pp |
| 3d Promoter Conviction | Holding stable or rising | Fell > 2pp |
| 3e Institutional Accumulation | FII or DII increasing | Both FII and DII declining |

---
### Gate 4 — Technical (Where is price in its range?)
| Gate | Standard Mode | Dip Opportunity Mode |
|---|---|---|
| 4a BB Squeeze | Width ≤ 115% of 63-day min | Same |
| 4b MA50 Zone | 0% – +12% above 50 DMA | Same |
| 4c Price Position | Within 5% of 52W high | **10–40% below 52W high** (buy window) |
| 4d Volume | 5-day vol ≥ 1.1x 50-day vol | Same |

---
### Composite Score
`Composite = (Quality × Wq + Valuation × Wv + Timing × Wt + Technical × Wtech) ÷ total_weight`

In **Dip Opportunity Mode**, Quality and Valuation weights are multiplied by 1.5×, Timing and Technical by 0.5×.
This surfaces high-quality businesses at a discount even if their price momentum is negative — exactly the setup for buying a good company on a fall.
"""
