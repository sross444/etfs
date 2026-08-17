import datetime as dt
import json

import polars as pl
import pytest

from etfs.errors import BadSymbol, QuotaExhausted
from etfs.store import Store, fill_missing_sessions, latest_session
from etfs.universe import GROUPS, universe


# --------------------------------------------------------------------------
# universe
# --------------------------------------------------------------------------

def test_universe_is_unique_and_complete():
    u = universe()
    assert u.height == sum(len(v) for v in GROUPS.values())
    assert u["ticker"].n_unique() == u.height
    assert u.null_count().sum_horizontal().item() == 0


def test_universe_subset():
    u = universe(["sector"])
    assert u.height == 11
    assert set(u["group"]) == {"sector"}


def test_universe_rejects_unknown_group():
    with pytest.raises(ValueError, match="unknown groups"):
        universe(["nope"])


# --------------------------------------------------------------------------
# calendar
# --------------------------------------------------------------------------

def test_latest_session_rolls_back_over_the_weekend():
    assert latest_session(dt.date(2026, 8, 15)) == dt.date(2026, 8, 14)  # Sat
    assert latest_session(dt.date(2026, 8, 16)) == dt.date(2026, 8, 14)  # Sun
    assert latest_session(dt.date(2026, 8, 17)) == dt.date(2026, 8, 17)  # Mon


# --------------------------------------------------------------------------
# store: resumability
# --------------------------------------------------------------------------

class StubClient:
    """Serves N tickers then behaves as if the daily quota is gone."""

    def __init__(self, budget=99, bad=()):
        self.budget, self.bad, self.seen = budget, set(bad), []

    def daily(self, ticker, full=True):
        if ticker in self.bad:
            raise BadSymbol("Invalid API call.")
        if len(self.seen) >= self.budget:
            raise QuotaExhausted("out")
        self.seen.append((ticker, full))
        return bars_payload_frame(ticker)


def bars_payload_frame(ticker, days=3):
    return pl.DataFrame({
        "dt": [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(days)],
        "open": [1.0] * days, "high": [1.0] * days, "low": [1.0] * days,
        "close": [1.0] * days, "volume": [1.0] * days,
        "ticker": [ticker] * days,
    })


def test_sync_stops_on_quota_and_keeps_what_it_got(tmp_path):
    client = StubClient(budget=4)
    store = Store(dir=tmp_path, groups=["sector"], client=client)
    report = store.sync()

    assert sum(v == "fetched" for v in report.values()) == 4
    assert len(list((tmp_path / "daily").glob("*.parquet"))) == 4
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest) == 4


def test_sync_resumes_where_it_stopped(tmp_path):
    Store(dir=tmp_path, groups=["sector"], client=StubClient(budget=4)).sync()

    resumed = StubClient(budget=99)
    Store(dir=tmp_path, groups=["sector"], client=resumed).sync()
    # the 4 already on disk are stale (bars end 2024) so they refresh with
    # compact; the remaining 7 are first-time full pulls.
    assert sum(1 for _, full in resumed.seen if full) == 7
    assert len(list((tmp_path / "daily").glob("*.parquet"))) == 11


def test_sync_skips_tickers_already_checked_today(tmp_path):
    Store(dir=tmp_path, groups=["sector"], client=StubClient()).sync()
    second = StubClient()
    report = Store(dir=tmp_path, groups=["sector"], client=second).sync()
    assert second.seen == []
    assert all(v == "skipped" for v in report.values())


def test_sync_records_bad_symbols_and_does_not_retry_them(tmp_path):
    client = StubClient(bad=["XLE"])
    store = Store(dir=tmp_path, groups=["sector"], client=client)
    report = store.sync()
    assert report["XLE"] == "bad"

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["XLE"]["status"] == "bad_symbol"
    assert not (tmp_path / "daily" / "XLE.parquet").exists()


def test_sync_merges_without_duplicating_rows(tmp_path):
    store = Store(dir=tmp_path, groups=["sector"], client=StubClient())
    store.sync()
    store.sync(force=True)  # same bars again
    df = pl.read_parquet(tmp_path / "daily" / "XLE.parquet")
    assert df.height == 3
    assert df.unique(subset=["ticker", "dt"]).height == 3


def test_load_joins_universe_metadata(tmp_path):
    store = Store(dir=tmp_path, groups=["sector"], client=StubClient())
    store.sync()
    df = store.load()
    assert set(df.columns) == {
        "dt", "open", "high", "low", "close", "volume", "ticker", "desc", "group",
    }
    assert df["ticker"].n_unique() == 11
    assert df.filter(pl.col("ticker") == "XLE")["desc"].unique().item() == "energy"


def test_load_common_start_balances_the_panel(tmp_path):
    store = Store(dir=tmp_path, groups=["sector"], client=StubClient())
    store.sync()
    # give one ticker a later inception
    late = bars_payload_frame("XLE").filter(pl.col("dt") >= dt.datetime(2024, 1, 2))
    late.write_parquet(tmp_path / "daily" / "XLE.parquet")
    assert store.load().filter(pl.col("ticker") == "XLE").height == 2

    balanced = store.load(common_start=True)
    counts = balanced.group_by("ticker").len()["len"].unique().to_list()
    assert counts == [2]  # every ticker trimmed to XLE's later inception
    assert balanced["dt"].min() == dt.datetime(2024, 1, 2)


def test_load_without_cache_is_a_clear_error(tmp_path):
    store = Store(dir=tmp_path, groups=["sector"], client=StubClient())
    with pytest.raises(FileNotFoundError, match="run sync"):
        store.load()


# --------------------------------------------------------------------------
# yahoo provider
# --------------------------------------------------------------------------

from etfs import yahoo  # noqa: E402


def yahoo_payload(days=3, offset=-14400, error=None, stamps=True):
    if error is not None:
        return {"chart": {"result": None, "error": {"description": error}}}
    base = int(dt.datetime(2024, 1, 2, 14, 30).timestamp())
    ts = [base + i * 86_400 for i in range(days)] if stamps else None
    quote = {
        "open": [10.0 + i for i in range(days)],
        "high": [11.0 + i for i in range(days)],
        "low": [9.0 + i for i in range(days)],
        "close": [10.5 + i for i in range(days)],
        "volume": [1000 + i for i in range(days)],
    }
    return {"chart": {"result": [{
        "meta": {"gmtoffset": offset},
        "timestamp": ts,
        "indicators": {
            "quote": [quote],
            "adjclose": [{"adjclose": [9.0 + i for i in range(days)]}],
        },
    }], "error": None}}


def fake_yahoo(monkeypatch, payloads, status=200):
    calls = []

    class R:
        status_code = status

        def __init__(self, p):
            self._p = p

        def raise_for_status(self):
            pass

        def json(self):
            return self._p

    def fake_get(url, params, headers, timeout):
        calls.append((url, params))
        return R(payloads.pop(0))

    monkeypatch.setattr("etfs.yahoo.requests.get", fake_get)
    monkeypatch.setattr("etfs.yahoo.time.sleep", lambda s: None)
    return yahoo.Client(min_interval=0.0), calls


def test_yahoo_parses_bars(monkeypatch):
    client, _ = fake_yahoo(monkeypatch, [yahoo_payload(3)])
    df = client.daily("XLE")
    assert df.columns == ["dt", "open", "high", "low", "close", "volume",
                          "adj_close", "ticker"]
    assert df.height == 3
    assert df["dt"].is_sorted()
    assert df["volume"].dtype == pl.Float64


def test_yahoo_normalises_stamps_to_the_trading_date(monkeypatch):
    """Bars are stamped 09:30 exchange-local; the date must not drift."""
    client, _ = fake_yahoo(monkeypatch, [yahoo_payload(1)])
    df = client.daily("XLE")
    bar = df["dt"].item()
    assert (bar.hour, bar.minute) == (0, 0)
    assert bar.date() == dt.date(2024, 1, 2)


def test_yahoo_delisted_symbol_raises_bad_symbol(monkeypatch):
    client, _ = fake_yahoo(
        monkeypatch, [yahoo_payload(error="No data found, symbol may be delisted")]
    )
    with pytest.raises(BadSymbol, match="delisted"):
        client.daily("NOPE")


def test_yahoo_empty_series_raises_bad_symbol(monkeypatch):
    client, _ = fake_yahoo(monkeypatch, [yahoo_payload(stamps=True, days=0)])
    with pytest.raises(BadSymbol):
        client.daily("NOPE")


def test_yahoo_full_flag_widens_the_window(monkeypatch):
    client, calls = fake_yahoo(monkeypatch, [yahoo_payload(1), yahoo_payload(1)])
    client.daily("XLE", full=True)
    client.daily("XLE", full=False)
    assert calls[0][1]["period1"] < calls[1][1]["period1"]


def test_yahoo_drops_null_price_bars(monkeypatch):
    payload = yahoo_payload(3)
    payload["chart"]["result"][0]["indicators"]["quote"][0]["close"][1] = None
    client, _ = fake_yahoo(monkeypatch, [payload])
    assert client.daily("XLE").height == 2


def test_store_builds_a_default_client_lazily(tmp_path):
    """No client passed -> one is constructed on first use, not at __init__."""
    from etfs.store import make_client

    assert isinstance(make_client(), yahoo.Client)
    store = Store(dir=tmp_path, groups=["sector"])
    assert store._client is None
    assert isinstance(store.client, yahoo.Client)


def test_sustained_rate_limiting_raises_quota_exhausted(monkeypatch):
    """429 past our patience must surface as QuotaExhausted so sync() banks
    its progress rather than dying mid-run."""
    class R:
        status_code = 429

        def raise_for_status(self):
            raise AssertionError("should not be reached")

        def json(self):
            raise AssertionError("should not be reached")

    monkeypatch.setattr("etfs.yahoo.requests.get",
                        lambda url, params, headers, timeout: R())
    monkeypatch.setattr("etfs.yahoo.time.sleep", lambda s: None)
    with pytest.raises(QuotaExhausted, match="rate limited"):
        yahoo.Client(min_interval=0.0).daily("XLE")


def test_sync_resumes_after_rate_limiting(tmp_path):
    """The whole point of the manifest: a throttled run is not a lost run."""
    store = Store(dir=tmp_path, groups=["sector"], client=StubClient(budget=4))
    report = store.sync()
    assert sum(v == "fetched" for v in report.values()) == 4
    assert len(list((tmp_path / "daily").glob("*.parquet"))) == 4

    resumed = Store(dir=tmp_path, groups=["sector"], client=StubClient())
    resumed.sync()
    assert len(list((tmp_path / "daily").glob("*.parquet"))) == 11


# --------------------------------------------------------------------------
# cli -- guards against import drift, which a refactor already broke once
# --------------------------------------------------------------------------

def test_cli_imports_and_runs(tmp_path, monkeypatch, capsys):
    from etfs import cli

    monkeypatch.setattr(
        "etfs.cli.Store",
        lambda dir, groups: Store(dir=dir, groups=groups, client=StubClient()),
    )
    assert cli.main(["--dir", str(tmp_path), "--groups", "sector"]) == 0
    assert "fetched=11" in capsys.readouterr().out


def test_cli_reports_pending_tickers_on_quota(tmp_path, monkeypatch, capsys):
    from etfs import cli

    monkeypatch.setattr(
        "etfs.cli.Store",
        lambda dir, groups: Store(
            dir=dir, groups=groups, client=StubClient(budget=3)
        ),
    )
    cli.main(["--dir", str(tmp_path), "--groups", "sector"])
    assert "rerun later" in capsys.readouterr().out



def test_load_settled_only_drops_the_live_session(tmp_path):
    store = Store(dir=tmp_path, groups=["sector"], client=StubClient())
    store.sync()
    today = pl.DataFrame({
        "dt": [dt.datetime.combine(dt.date.today(), dt.time())],
        "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
        "volume": [1.0], "ticker": ["XLE"],
    })
    pl.concat([pl.read_parquet(tmp_path / "daily" / "XLE.parquet"), today],
              how="diagonal").write_parquet(tmp_path / "daily" / "XLE.parquet")

    assert store.load()["dt"].max().date() == dt.date.today()
    assert store.load(settled_only=True)["dt"].max().date() < dt.date.today()


def test_load_end_bounds_the_window(tmp_path):
    store = Store(dir=tmp_path, groups=["sector"], client=StubClient())
    store.sync()
    out = store.load(end=dt.date(2024, 1, 2))
    assert out["dt"].max() == dt.datetime(2024, 1, 2)


# --------------------------------------------------------------------------
# gap filling
# --------------------------------------------------------------------------

def panel(rows):
    """rows: (ticker, day-of-month, close) -> a minimal OHLCV frame."""
    return pl.DataFrame(
        [
            [t, dt.datetime(2024, 1, d), c, c + 1, c - 1, c, 100.0]
            for t, d, c in rows
        ],
        schema=["ticker", "dt", "open", "high", "low", "close", "volume"],
        orient="row",
    )


def test_fill_uses_prior_close_with_zero_volume():
    data = panel([("A", 1, 10.0), ("A", 2, 11.0), ("A", 3, 12.0),
                  ("B", 1, 20.0),                 ("B", 3, 22.0)])
    out = fill_missing_sessions(data).sort(["ticker", "dt"])

    gap = out.filter((pl.col("ticker") == "B") & (pl.col("dt").dt.day() == 2))
    assert gap.height == 1
    row = gap.row(0, named=True)
    assert row["open"] == row["high"] == row["low"] == row["close"] == 20.0
    assert row["volume"] == 0.0
    assert row["imputed"] is True


def test_fill_makes_the_panel_rectangular():
    data = panel([("A", 1, 10.0), ("A", 2, 11.0), ("A", 3, 12.0),
                  ("B", 1, 20.0),                 ("B", 3, 22.0)])
    out = fill_missing_sessions(data)
    assert out.height == 6
    counts = out.group_by("dt").len()["len"].unique().to_list()
    assert counts == [2]


def test_fill_leaves_real_bars_untouched():
    data = panel([("A", 1, 10.0), ("A", 2, 11.0),
                  ("B", 1, 20.0),               ("B", 3, 22.0), ("A", 3, 12.0)])
    out = fill_missing_sessions(data)
    real = out.filter(~pl.col("imputed")).drop("imputed").sort(["ticker", "dt"])
    assert real.equals(data.sort(["ticker", "dt"]).select(real.columns))


def test_fill_does_not_backfill_before_inception():
    """A late lister must not be invented into existence early."""
    data = panel([("A", 1, 10.0), ("A", 2, 11.0), ("A", 3, 12.0),
                                                  ("B", 3, 22.0)])
    out = fill_missing_sessions(data)
    b = out.filter(pl.col("ticker") == "B")
    assert b.height == 1
    assert b["dt"].min() == dt.datetime(2024, 1, 3)


def test_fill_does_not_extend_past_the_last_bar():
    data = panel([("A", 1, 10.0), ("A", 2, 11.0), ("A", 3, 12.0),
                  ("B", 1, 20.0)])
    out = fill_missing_sessions(data)
    assert out.filter(pl.col("ticker") == "B").height == 1


def test_fill_carries_one_close_across_a_run_of_gaps():
    data = panel([("A", 1, 10.0), ("A", 2, 11.0), ("A", 3, 12.0), ("A", 4, 13.0),
                  ("B", 1, 20.0),                                 ("B", 4, 23.0)])
    out = fill_missing_sessions(data).sort(["ticker", "dt"])
    gaps = out.filter((pl.col("ticker") == "B") & pl.col("imputed"))
    assert gaps.height == 2
    assert gaps["close"].to_list() == [20.0, 20.0]  # both carry the last real close


def test_fill_carries_adj_close_and_labels():
    data = panel([("A", 1, 10.0), ("A", 2, 11.0),
                  ("B", 1, 20.0)]).with_columns(
        pl.col("close").alias("adj_close"),
        pl.lit("x").alias("desc"),
        pl.lit("g").alias("group"),
    )
    data = pl.concat([data, panel([("B", 3, 22.0)]).with_columns(
        pl.col("close").alias("adj_close"),
        pl.lit("x").alias("desc"), pl.lit("g").alias("group"))], how="diagonal")
    out = fill_missing_sessions(data)
    gap = out.filter(pl.col("imputed")).row(0, named=True)
    assert gap["adj_close"] == 20.0
    assert (gap["desc"], gap["group"]) == ("x", "g")
    assert out["desc"].null_count() == 0


def test_fill_is_a_no_op_on_a_complete_panel():
    data = panel([("A", 1, 10.0), ("A", 2, 11.0), ("B", 1, 20.0), ("B", 2, 21.0)])
    out = fill_missing_sessions(data)
    assert out.height == data.height
    assert not out["imputed"].any()


def test_load_fill_gaps_flag(tmp_path):
    store = Store(dir=tmp_path, groups=["sector"], client=StubClient())
    store.sync()
    short = pl.read_parquet(tmp_path / "daily" / "XLE.parquet").filter(
        pl.col("dt") != dt.datetime(2024, 1, 2)
    )
    short.write_parquet(tmp_path / "daily" / "XLE.parquet")

    assert store.load().height == 11 * 3 - 1
    filled = store.load(fill_gaps=True)
    assert filled.height == 11 * 3
    assert filled["imputed"].sum() == 1
    assert "imputed" not in store.load().columns
