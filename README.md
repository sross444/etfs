# etfs

Daily OHLCV for a fixed ETF universe, cached to parquet.

Condensed from the `etfs_old` research repo: everything here exists to get
`dt, open, high, low, close, volume` for 76 ETFs from 2018 onward, and nothing
else.

```python
from etfs import Store

store = Store()              # data/ , all groups, Yahoo provider
store.sync()                 # download or update every ticker
df = store.load()            # one tidy polars frame
```

```
shape: (164_562, 10)
dt | open | high | low | close | volume | adj_close | ticker | desc | group
```

Or from the shell:

```bash
pip install -e .
etfs-sync                    # everything
etfs-sync --groups sector    # just the 11 sector SPDRs
```

## The universe

`etfs/universe.py`, three groups, 76 tickers:

| group | n | what |
|---|---|---|
| `sector` | 11 | the GICS sector SPDRs (XLE … XLRE) |
| `asset class` | 17 | equity / rates / credit / real assets sleeves |
| `country` | 48 | single-country funds |

Ported from the old `lots_of_etfs` notebook with three corrections, each
verified against a second source rather than assumed:

- The notebook used `SMH`, `XTL` and `IYR` as sector proxies. Replaced with the
  actual sector SPDRs `XLK`, `XLC` and `XLRE`.
- The notebook labelled `EWS` as Italy. `EWS` is iShares MSCI **Singapore**;
  Italy is `EWI`. Both are now present, correctly labelled.
- `HYXU` and `MUG` are delisted — reference data 404s `HYXU`, and only a single
  stale bar is served for it, none for `MUG`. Dropped.

### Start date

**Every ticker is trading from 2018-06-19** — that's `XLC`'s first session, and
coverage steps cleanly from 75 to 76 on that date with no ragged ramp. So:

```python
df = store.load(common_start=True, settled_only=True)
# 2,050 sessions x 76 tickers, 2018-06-19 -> present
```

`XLC` is the binding constraint and is kept deliberately: it lists later than
everything else, but dropping a GICS sector would leave the sector group
structurally incomplete. `KWT` (Sep 2020) and `TCHI` (Feb 2022) listed later
still and were removed from the universe entirely — they alone had pinned the
balanced panel at 2022-02-01, costing 75% of it.

A sync during market hours stores a live, partial bar for the current session.
`load(settled_only=True)` drops it.

### Gaps

14 tickers have no bar on **2026-08-11** (`CNYA EDEN EFNL ENOR EWUS IEFA IEMG
INDA KSA QAT SMIN UAE XLC XLRE`) while the rest of the universe traded. This is
upstream — a fresh Yahoo fetch reproduces it — not a sync artifact.

`load(fill_gaps=True)` pads them:

```python
df = store.load(common_start=True, settled_only=True, fill_gaps=True)
# 155,800 rows = 2,050 sessions x 76 tickers, exactly rectangular
```

A padded bar is `open = high = low = close = the prior close`, with
`volume = 0` — a zero-return, zero-volume day, which is what "no trading
occurred and no price was established" actually means. Nothing is interpolated
and no future information is used, so the fill cannot leak lookahead into a
backtest.

Two guarantees worth relying on:

- **Only interior gaps are filled.** Each ticker's calendar runs from its own
  first bar to its own last, so `XLC` is never invented back to January 2018.
- **Real bars are never altered** — the fill only inserts.

Every synthetic bar carries `imputed = True`, so it stays auditable:

```python
df.filter(pl.col("imputed"))            # just the padding
df.filter(~pl.col("imputed"))           # back to real trades only
```

Weight by `volume` and these days drop out on their own.

## Data source

Bars come from Yahoo Finance's chart endpoint. **No API key, no account, no
per-day request budget** — history runs back to each fund's inception, and the
OHLC arrives already split-adjusted, so there is no split table to maintain.
`adj_close` additionally nets out dividends.

Two caveats. The endpoint is undocumented and unsupported, so it can change
without notice. And it is rate-limited by IP: sustained bursts return HTTP 429,
which the client paces against, backs off from, and finally reports as
`QuotaExhausted` so a sync stops cleanly instead of dying mid-run.

`Store` accepts any object exposing `daily(ticker, full) -> pl.DataFrame`, so
pointing it at a different source is a drop-in:

```python
Store(client=MySource())
```

### History depth

The default fetch window starts at 2018-01-01, which is *not* each fund's
inception — most of the universe goes back much further (`XLE` and `XLK` to
1998, `EWJ` to 1996). Widen it when you want the deeper history:

```python
from etfs import yahoo
Store(client=yahoo.Client(start=dt.date(1996, 1, 1))).sync(force=True)
```

That does not extend a *balanced* panel, which is pinned at 2018-06-19 by
`XLC`'s genuine inception, but it does reach the 2000 and 2008 drawdowns that
are otherwise outside the data entirely.

## Resumability

`sync()` writes one parquet per ticker plus a `data/manifest.json`, and is safe
to interrupt. It skips tickers already checked today, skips those already
current, records unrecognised symbols so they aren't retried forever, and banks
its progress if the source starts rate-limiting — so a throttled run is never a
lost run. Just run it again.

```bash
etfs-sync
# fetched=78
```

## Indicators

`etfs/technicals.py` computes 10 indicators per window, per ticker, over the
whole panel at once:

```python
from etfs.technicals import metrics, metric_columns

df = store.load(common_start=True, settled_only=True, fill_gaps=True)
df = metrics(df)                       # +50 columns, ~1s for 155k rows
```

`ts` trend strength · `er` signed efficiency ratio · `rsi` · `so` stochastic
oscillator · `rvwq` close vs volume-weighted quantiles — each with a `v` suffix
variant weighting the day's move by its volume. Default windows are
`[3, 7, 14, 28, 56]`; `metric_columns()` lists the names.

### Imputed bars

`metrics(on_imputed=...)` decides how the padded no-trade bars are treated:

| mode | behaviour |
|---|---|
| `"skip"` (default) | drop them, compute on the real series, carry the last real value onto them |
| `"include"` | treat as ordinary bars — they are zero-return and zero-volume, so they drag indicators toward neutral |
| `"null"` | compute on the real series, leave the indicator null on imputed rows |

This matters more than it looks. Including a single imputed bar changed
`ts_28` on **56 subsequent bars** (the whole window) by up to 0.24, and flipped
its sign on 6 of them. A day the fund didn't trade shouldn't register as a day
of no momentum, so `skip` is the default.

### Elliott wave

```python
df = metrics(df, elliott=True)          # +10 ew_* columns
```

Elliott wave analysis is discretionary — analysts label the same chart
differently, and counts get revised in hindsight. `etfs/elliott.py` does not
resolve that. It extracts the parts that *are* mechanical and emits them as
numbers: ATR-scaled swing pivots, Elliott's three hard rules, and the Fibonacci
ratios between legs.

| column | meaning |
|---|---|
| `ew_dir` | +1 up-impulse, −1 down, 0 if the pivots don't form one |
| `ew_leg` | which leg of the current attempt is forming, 1–5 |
| `ew_rules` | fraction of the three hard rules satisfied, 0–1 |
| `ew_w2_retr`, `ew_w3_ext`, `ew_w4_retr` | leg ratios |
| `ew_fib` | closeness of those ratios to canonical Fibonacci levels, 0–1 |
| `ew_swing_pos` | close within the current swing; 1 at the extreme, negative if the swing failed |
| `ew_swing_age` | bars since the last pivot confirmed |
| `ew_pivot_lag` | bars that pivot took to confirm — the information lag |

**These features are causal, and that is the whole design constraint.** A swing
pivot is not knowable when it happens — only once price has retraced far enough
away from it. A naive zigzag marks the pivot at its own bar, back-dating
information by however many bars confirmation took. That is repainting, and it
leaks the future into any backtest built on it.

Here every pivot carries its confirmation bar, features at bar `t` use only
pivots confirmed by `t`, and the test suite asserts that computing on a
truncated series reproduces the full-series values exactly, across several
seeds. The price is lag: **median 5 sessions, up to 33**, reported per bar in
`ew_pivot_lag`. Nothing here calls a turn as it happens, and anything claiming
to would be repainting.

Two things worth knowing before using these:

- On the real panel only **6.4%** of scored bars satisfy all three hard rules;
  55% satisfy just one. Mechanical zigzag pivots rarely form textbook impulses,
  so `ew_rules` is better read as a continuous structure score than a filter.
- `on_imputed` is inherited from `metrics()`, so a padded no-trade day cannot
  manufacture a phantom swing pivot.

### Ported with corrections

The original `etfs_old/technicals.py` had five defects, all fixed here and all
covered by tests:

1. **No per-ticker grouping** — every `shift`/`rolling` bled across the boundary
   between one ticker and the next. It was only correct one ticker at a time.
2. **True range used the previous bar's *high*** instead of the previous
   *close*.
3. **Directional movement was divided by true range twice**, leaving 1/price
   units. The final ratio doesn't cancel it, because smoothing happens between
   the two divisions.
4. **`drop_nulls()` mid-pipeline** inside `efficiency_ratio` and `rsi` silently
   dropped the warm-up rows of whichever indicators had already been added, so
   results depended on call order.
5. **The decay ran backwards.** polars applies `weights[0]` to the *oldest* bar,
   and the original passed `decay ** arange(window)`, whose first entry is 1.0 —
   so `decay=0.9` weighted the stalest bar most.

Separately, `metrics()` could never have run in the old repo at all:
`np.quantile(weights=...)` requires numpy >= 2.0 and that environment had 1.26,
where it raises `TypeError`. The rolling weighted quantile is now vectorised —
**38x faster**, 46s to 1.2s for the full panel — and verified against
`np.quantile(method="inverted_cdf")` across 200 random trials.

## Layout

```
etfs/
  universe.py       the ticker list, as data
  store.py          resumable parquet cache + panel assembly
  yahoo.py          the data source
  technicals.py     indicators over the panel
  elliott.py        causal Elliott-wave structure features
  errors.py
  cli.py            etfs-sync
tests/              101 tests, no network
```

```bash
pytest
```

## License

MIT — see [LICENSE](LICENSE).
