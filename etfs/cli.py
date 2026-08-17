"""`etfs-sync` -- fetch daily bars into the local parquet cache, then report.

Safe to interrupt and safe to re-run: it skips what is already current and
resumes whatever a previous run did not finish.
"""

import argparse
import collections

from etfs.store import Store
from etfs.universe import GROUPS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="etfs-sync", description=__doc__)
    parser.add_argument("--dir", default="data", help="cache directory")
    parser.add_argument(
        "--groups", nargs="*", choices=sorted(GROUPS), default=None,
        help="universe subset (default: all)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="stop after this many network fetches",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-download full history, ignoring the cache",
    )
    args = parser.parse_args(argv)

    store = Store(dir=args.dir, groups=args.groups)
    report = store.sync(force=args.force, limit=args.limit)

    counts = collections.Counter(report.values())
    order = ["fetched", "current", "skipped", "bad", "quota"]
    print(" ".join(f"{k}={counts[k]}" for k in order if counts[k]))

    if bad := [t for t, s in report.items() if s == "bad"]:
        print(f"unrecognised symbols: {', '.join(sorted(bad))}")
    if pending := [t for t, s in report.items() if s == "quota"]:
        print(f"{len(pending)} ticker(s) left, rate limited -- rerun later: "
              f"{', '.join(sorted(pending)[:8])}"
              f"{' ...' if len(pending) > 8 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
