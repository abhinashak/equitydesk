"""
utils/fees.py
─────────────
India equity CNC fee calculator (NSE).
Used by both BLL and CLI layers.
"""

from dataclasses import dataclass


@dataclass
class FeeBreakdown:
    stt: float
    etc: float
    sebi: float
    gst: float
    dp: float
    total: float

    def __str__(self) -> str:
        return (
            f"STT={self.stt:.2f}  ETC={self.etc:.2f}  "
            f"SEBI={self.sebi:.2f}  GST={self.gst:.2f}  "
            f"DP={self.dp:.2f}  → TOTAL={self.total:.2f}"
        )


def calc_fees(
    value: float,
    stt_rate: float = 0.001,
    etc_rate: float = 0.0000325,
    sebi_rate: float = 0.000001,
    gst_rate: float = 0.18,
    dp_base: float = 15.34,
    side: str = "buy",   # "buy" | "sell"
) -> FeeBreakdown:
    """Compute CNC transaction fees for a given trade value (₹)."""
    # STT only on sell side for CNC equity
    stt  = value * stt_rate if side == "sell" else 0.0
    etc  = value * etc_rate
    sebi = value * sebi_rate
    gst  = (etc + sebi) * gst_rate
    dp   = dp_base if side == "sell" else 0.0
    total = stt + etc + sebi + gst + dp
    return FeeBreakdown(stt=stt, etc=etc, sebi=sebi, gst=gst, dp=dp, total=total)
