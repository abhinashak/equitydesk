PERIODS: list[tuple[str, int, str]] = [
    ("3D",   3,    "3 days"),
    ("7D",   7,    "7 days"),
    ("30D",  30,   "1 month"),
    ("45D",  45,   "45 days"),
    ("2M",   60,   "2 months"),
    ("3M",   90,   "3 months"),
    ("4M",   120,  "4 months"),
    ("5M",   150,  "5 months"),
    ("6M",   180,  "6 months"),
    ("1Y",   365,  "1 year"),
    ("2Y",   730,  "2 years"),
    ("3Y",   1095, "3 years"),
    ("4Y",   1460, "4 years"),
    ("5Y",   1825, "5 years"),
    ("10Y",  3650, "10 years"),
]

PERIOD_DAYS: dict[str, int] = {label: days for label, days, _ in PERIODS}
PERIOD_INTERVALS: dict[str, str] = {label: interval for label, _, interval in PERIODS}
PERIOD_LABELS: list[str] = [label for label, _, _ in PERIODS]


def get_period_days(label: str) -> int:
    """Return lookback days for a period label, or raise KeyError if invalid."""
    if label not in PERIOD_DAYS:
        raise ValueError(f"Unknown period label: {label!r}. Valid options: {PERIOD_LABELS}")
    return PERIOD_DAYS[label]


def get_period_interval(label: str) -> str:
    """Return DuckDB interval string for a period label, or raise KeyError if invalid."""
    if label not in PERIOD_INTERVALS:
        raise ValueError(f"Unknown period label: {label!r}. Valid options: {PERIOD_LABELS}")
    return PERIOD_INTERVALS[label]
