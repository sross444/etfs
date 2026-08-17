"""A resumable, on-disk cache of daily bars -- one parquet per ticker.

Source-agnostic: anything exposing `daily(ticker, full) -> pl.DataFrame` works,
so a different data source is a drop-in. `sync()` fetches what it can and
records progress in a manifest; run it again to pick up the rest. `load()`
returns the panel.
"""

import datetime as dt
import json
from pathlib import Path

import polars as pl

from etfs.errors import BadSymbol, QuotaExhausted
from etfs.universe import universe


def latest_session(today: dt.date | None = None) -> dt.date:
    """Most recent weekday on or before today. Ignores market holidays, which
    only costs an occasional redundant refresh."""
    day = today or dt.date.today()
    while day.weekday() >= 5:
        day -= dt.timedelta(days=1)
    return day


def make_client():
    from etfs import yahoo

    return yahoo.Client()


class Store:
    def __init__(
        self,
        dir: str | Path = "data",
        groups: list[str] | None = None,
        client=None,
    ):
        self.dir = Path(dir)
        self.bars = self.dir / "daily"
        self.bars.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dir / "manifest.json"
        self.universe = universe(groups)
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = make_client()
        return self._client

    def path(self, ticker: str) -> Path:
        return self.bars / f"{ticker}.parquet"

    def _read_manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {}

    def _write_manifest(self, manifest: dict) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def sync(self, force: bool = False, limit: int | None = None) -> dict:
        """Bring every ticker up to the latest session.

        Skips tickers already checked today and those previously found to be bad
        symbols. If the source starts refusing requests, stops cleanly --
        everything fetched so far is already on disk, so a later run resumes.

        Args:
            force: re-fetch full history even for tickers that look current.
            limit: stop after this many network fetches.

        Returns:
            Per-ticker status: fetched / current / skipped / bad / quota.
        """
        manifest = self._read_manifest()
        today, session = dt.date.today(), latest_session()
        report, fetches = {}, 0

        for ticker in self.universe["ticker"]:
            entry = manifest.get(ticker, {})

            if not force:
                if entry.get("status") == "bad_symbol":
                    report[ticker] = "bad"
                    continue
                if entry.get("checked") == today.isoformat():
                    report[ticker] = "skipped"
                    continue
                if entry.get("last_date", "") >= session.isoformat():
                    manifest[ticker] = {**entry, "checked": today.isoformat()}
                    report[ticker] = "current"
                    continue

            if limit is not None and fetches >= limit:
                report[ticker] = "quota"
                continue

            existing = (
                pl.read_parquet(self.path(ticker))
                if self.path(ticker).exists() and not force
                else None
            )
            try:
                fresh = self.client.daily(ticker, full=existing is None)
                fetches += 1
            except BadSymbol:
                manifest[ticker] = {"status": "bad_symbol",
                                    "checked": today.isoformat()}
                report[ticker] = "bad"
                self._write_manifest(manifest)
                continue
            except QuotaExhausted:
                report[ticker] = "quota"
                self._write_manifest(manifest)
                break

            if existing is not None:
                fresh = (
                    pl.concat([existing, fresh], how="diagonal")
                    .unique(subset=["ticker", "dt"], keep="last")
                    .sort("dt")
                )
            fresh.write_parquet(self.path(ticker))

            manifest[ticker] = {
                "status": "ok",
                "checked": today.isoformat(),
                "first_date": fresh["dt"].min().date().isoformat(),
                "last_date": fresh["dt"].max().date().isoformat(),
                "rows": fresh.height,
            }
            report[ticker] = "fetched"
            self._write_manifest(manifest)

        self._write_manifest(manifest)
        return report

    def load(
        self,
        start: dt.date | None = None,
        end: dt.date | None = None,
        common_start: bool = False,
        settled_only: bool = False,
        fill_gaps: bool = False,
    ) -> pl.DataFrame:
        """Load the cached panel, joined to the universe metadata.

        Args:
            start: drop bars before this date.
            end: drop bars after this date.
            common_start: truncate to the latest inception across tickers, so
                every ticker spans the same window (a balanced panel). Note that
                one late-listing ticker costs the whole panel -- TCHI (2022)
                alone halves it. Check `coverage()` first.
            settled_only: drop the current session. A sync run during market
                hours stores a live, incomplete bar for today; use this when a
                partial last bar would corrupt a calculation.
            fill_gaps: pad sessions a ticker is missing while the rest of the
                panel traded, giving a rectangular panel. See
                `fill_missing_sessions`; adds an `imputed` flag column.

        Note that the cached bars are already split-adjusted at source, so no
        split back-adjustment happens here.
        """
        files = sorted(self.bars.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no cached data in {self.bars}; run sync() first")

        data = pl.concat([pl.read_parquet(f) for f in files], how="diagonal")
        data = data.join(self.universe, on="ticker", how="inner")

        if start is not None:
            data = data.filter(pl.col("dt") >= pl.lit(start).cast(pl.Datetime))
        if end is not None:
            data = data.filter(pl.col("dt") <= pl.lit(end).cast(pl.Datetime))
        if settled_only:
            today = pl.lit(dt.date.today()).cast(pl.Datetime)
            data = data.filter(pl.col("dt") < today)
        if common_start:
            first = data.group_by("ticker").agg(pl.col("dt").min())["dt"].max()
            data = data.filter(pl.col("dt") >= first)
        if fill_gaps:
            data = fill_missing_sessions(data)

        return data.sort(["ticker", "dt"])

    def coverage(self) -> pl.DataFrame:
        """What's on disk, per ticker -- the quick way to see sync progress."""
        manifest = self._read_manifest()
        rows = [
            (t, m.get("status"), m.get("first_date"), m.get("last_date"),
             m.get("rows"))
            for t, m in sorted(manifest.items())
        ]
        return pl.DataFrame(
            rows,
            schema=["ticker", "status", "first_date", "last_date", "rows"],
            orient="row",
        )


def fill_missing_sessions(data: pl.DataFrame) -> pl.DataFrame:
    """Pad sessions a ticker is missing while the rest of the panel traded.

    A missing bar becomes open = high = low = close = the prior close, with
    volume 0 -- a zero-return, zero-volume day, which is the honest way to say
    "this fund did not trade and no price was established". Nothing is
    interpolated and no future information is used.

    Only *interior* gaps are filled: the calendar for each ticker runs from its
    own first bar to its own last, so a fund is never back-filled to before it
    listed. Runs of consecutive gaps carry the same last real close.

    Adds an `imputed` boolean so a synthetic bar is always distinguishable from
    a real one -- filter it out with `df.filter(~pl.col("imputed"))`.
    """
    sessions = data.select("dt").unique()
    bounds = data.group_by("ticker").agg(
        pl.col("dt").min().alias("_first"), pl.col("dt").max().alias("_last")
    )
    grid = (
        bounds.join(sessions, how="cross")
        .filter(
            (pl.col("dt") >= pl.col("_first")) & (pl.col("dt") <= pl.col("_last"))
        )
        .select("ticker", "dt")
    )

    merged = (
        grid.join(data, on=["ticker", "dt"], how="left")
        .sort(["ticker", "dt"])
        .with_columns(pl.col("close").is_null().alias("imputed"))
    )

    # Carry the last real close forward; for a run of gaps this is the last
    # close before the run, which is what "no trading happened" implies.
    carried = ["close"] + (["adj_close"] if "adj_close" in data.columns else [])
    merged = merged.with_columns(
        [pl.col(c).forward_fill().over("ticker").alias(c) for c in carried]
    )

    labels = [c for c in ("desc", "group") if c in data.columns]
    return merged.with_columns(
        *[
            pl.when(pl.col("imputed")).then(pl.col("close")).otherwise(pl.col(c)).alias(c)
            for c in ("open", "high", "low")
        ],
        pl.when(pl.col("imputed")).then(0.0).otherwise(pl.col("volume")).alias("volume"),
        *[
            pl.col(c).forward_fill().backward_fill().over("ticker").alias(c)
            for c in labels
        ],
    )
