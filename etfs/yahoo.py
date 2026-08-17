"""Daily OHLCV from Yahoo Finance's chart endpoint.

No API key, no account, no per-day request budget -- bars run back to each
fund's inception. The OHLC is already split-adjusted, so no split table is
needed; `adj_close` additionally nets out dividends.

Two caveats worth knowing. The endpoint is undocumented and unsupported, so it
can change without notice. And it is rate-limited by IP: sustained bursts get
HTTP 429, which this client backs off against and finally reports as
`QuotaExhausted` so `Store.sync` can stop cleanly and resume later.

`Store` takes any object exposing `daily(ticker, full)`, so swapping in another
source is a drop-in.
"""

import datetime as dt
import time
from dataclasses import dataclass, field

import polars as pl
import requests

from etfs.errors import BadSymbol, QuotaExhausted

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

# Yahoo rejects the default urllib/requests agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

SCHEMA = {
    "dt": pl.Datetime,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "adj_close": pl.Float64,
}


@dataclass
class Client:
    """Rate-limited Yahoo chart client."""

    start: dt.date = dt.date(2018, 1, 1)
    min_interval: float = 0.5
    max_retries: int = 4
    timeout: float = 30.0

    _last_request: float = field(default=0.0, init=False, repr=False)

    def _pace(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def daily(self, ticker: str, full: bool = True) -> pl.DataFrame:
        """Daily OHLCV for one ticker.

        Args:
            ticker: the symbol to fetch.
            full: True for history back to `self.start`, False for the last
                ~6 months (cheaper payload for routine updates).
        """
        start = self.start if full else dt.date.today() - dt.timedelta(days=180)
        params = {
            "period1": int(dt.datetime.combine(start, dt.time()).timestamp()),
            "period2": int(time.time()) + 86_400,
            "interval": "1d",
        }

        payload, throttled = None, False
        for attempt in range(self.max_retries):
            self._pace()
            try:
                r = requests.get(
                    CHART_URL.format(ticker=ticker), params=params,
                    headers=HEADERS, timeout=self.timeout,
                )
                if r.status_code == 429:
                    throttled = True
                    time.sleep(5 * 2 ** attempt)
                    continue
                throttled = False
                r.raise_for_status()
                payload = r.json()
                break
            except (requests.RequestException, ValueError):
                time.sleep(2 ** attempt)
        if payload is None:
            if throttled:
                # Rate limited past our patience: let sync() bank its progress.
                raise QuotaExhausted(
                    f"rate limited fetching {ticker}; retry later"
                )
            raise RuntimeError(f"request failed for {ticker}")

        chart = payload.get("chart") or {}
        if chart.get("error"):
            raise BadSymbol(f"{ticker}: {chart['error'].get('description')}")
        results = chart.get("result")
        if not results:
            raise BadSymbol(f"{ticker}: empty response")

        result = results[0]
        stamps = result.get("timestamp")
        if not stamps:
            raise BadSymbol(f"{ticker}: no bars returned")

        quote = result["indicators"]["quote"][0]
        adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
        adj = adj if adj is not None else [None] * len(stamps)

        # Bars are stamped at the open in exchange-local time; shift by the
        # exchange's UTC offset so the calendar date is the trading date.
        offset = result["meta"].get("gmtoffset", 0)
        rows = [
            [
                dt.datetime.fromtimestamp(t + offset, dt.timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0, tzinfo=None
                ),
                quote["open"][i], quote["high"][i], quote["low"][i],
                quote["close"][i],
                None if quote["volume"][i] is None else float(quote["volume"][i]),
                adj[i],
            ]
            for i, t in enumerate(stamps)
        ]

        return (
            pl.DataFrame(rows, schema=SCHEMA, orient="row")
            .drop_nulls(subset=["open", "high", "low", "close"])
            .unique(subset=["dt"], keep="last")
            .sort("dt")
            .with_columns(pl.lit(ticker).alias("ticker"))
        )
