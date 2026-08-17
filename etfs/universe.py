"""The ETF universe, as data.

Three groups: the 11 GICS sector SPDRs, a cross-asset sleeve, and single-country
funds. Carried over from earlier research, with the sector substitutes replaced
by the real GICS sector SPDRs and a mislabelled ticker corrected (EWS, iShares
Singapore, had been tagged as Italy -- that's EWI).
"""

import polars as pl

# (ticker, description)
SECTOR = [
    ("XLE", "energy"),
    ("XLB", "materials"),
    ("XLI", "industrials"),
    ("XLY", "discretionary"),
    ("XLP", "staples"),
    ("XLV", "healthcare"),
    ("XLF", "financials"),
    ("XLK", "technology"),
    ("XLC", "communication services"),
    ("XLU", "utilities"),
    ("XLRE", "real estate"),
]

ASSET_CLASS = [
    ("IWB", "us large"),
    ("IWM", "us small"),
    ("IEFA", "eafe large"),
    ("IEMG", "emerging equity"),
    ("SHY", "us sovereign 1-3yr"),
    ("IEI", "us sovereign 3-7yr"),
    ("IEF", "us sovereign 7-10yr"),
    ("TLH", "us sovereign 10-20yr"),
    ("TLT", "us sovereign 20+yr"),
    ("VTEB", "us muni"),
    ("LQD", "us investment grade"),
    ("HYG", "us high yield"),
    ("MBB", "us mortgage backed"),
    # HYXU (eafe high yield) was in the old research set but is delisted --
    # reference data 404s it and only a single stale bar is served.
    ("EMB", "emerging debt"),
    ("VNQ", "us real estate"),
    ("GNR", "global natural resources"),
    ("GLD", "gold"),
]

COUNTRY = [
    ("EWJ", "japan"),
    ("INDA", "india"),
    ("EWT", "taiwan"),
    ("MCHI", "china-all"),
    ("EWY", "south korea"),
    ("FXI", "china-large"),
    ("EWZ", "brazil"),
    ("EWU", "uk"),
    ("EWC", "canada"),
    ("EWW", "mexico"),
    ("EWA", "australia"),
    ("EWL", "switzerland"),
    ("EWP", "spain"),
    ("SMIN", "india-small"),
    ("INDY", "india-large"),
    ("EWG", "germany"),
    ("EWQ", "france"),
    ("KSA", "saudi arabia"),
    ("EWH", "hong kong"),
    ("EWI", "italy"),
    ("ECH", "chile"),
    ("EIDO", "indonesia"),
    ("EWM", "malaysia"),
    ("EWD", "sweden"),
    ("EZA", "south africa"),
    ("EPOL", "poland"),
    ("EWN", "netherlands"),
    ("CNYA", "china-a"),
    ("EDEN", "denmark"),
    ("THD", "thailand"),
    ("TUR", "turkey"),
    ("JPXN", "japan-large"),
    ("EIS", "israel"),
    ("EWZS", "brazil-small"),
    ("EIRL", "ireland"),
    ("EPHE", "philippines"),
    ("SCJ", "japan-small"),
    ("ENZL", "new zealand"),
    ("EPU", "peru"),
    ("EWUS", "uk-small"),
    ("EWO", "austria"),
    ("ECNS", "china-small"),
    ("ENOR", "norway"),
    ("EFNL", "finland"),
    ("EWK", "belgium"),
    ("EWS", "singapore"),
    ("QAT", "qatar"),
    ("UAE", "uae"),
    # Excluded deliberately, not by oversight: KWT (kuwait, listed 2020-09) and
    # TCHI (china-tech, listed 2022-02) are both live, but as the two latest
    # listings they bound the balanced panel at 2022-02-01. Dropping them moves
    # the common start back to 2018-06-19 and grows it by 75%; XLC is kept
    # despite also listing late, because dropping a GICS sector would leave the
    # sector group structurally incomplete.
]

GROUPS = {"sector": SECTOR, "asset class": ASSET_CLASS, "country": COUNTRY}


def universe(groups: list[str] | None = None) -> pl.DataFrame:
    """Return the universe as a DataFrame of ticker / desc / group.

    Args:
        groups: subset of GROUPS keys; None means all of them.
    """
    groups = list(GROUPS) if groups is None else groups
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown groups: {sorted(unknown)}; expected {sorted(GROUPS)}")

    rows = [
        (ticker, desc, group)
        for group in groups
        for ticker, desc in GROUPS[group]
    ]
    return pl.DataFrame(rows, schema=["ticker", "desc", "group"], orient="row")


def tickers(groups: list[str] | None = None) -> list[str]:
    return universe(groups)["ticker"].to_list()
