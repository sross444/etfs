"""Errors shared by the data sources."""


class BadSymbol(ValueError):
    """The source has no data for this symbol (unlisted, delisted, typo).

    Permanent: `Store.sync` records it and stops retrying.
    """


class QuotaExhausted(RuntimeError):
    """The source is refusing further requests for now.

    Transient: `Store.sync` stops cleanly, keeping whatever it already wrote,
    so a later run resumes.
    """
