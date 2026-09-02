"""Turn a stream of individual messages into conversational units.

One message is usually too short to embed into anything meaningful — lesson 6.5 opens on
exactly that failure. This module is the fix used on both halves of that notebook, showcase
and your-turn: merge consecutive messages from the same author into one unit, where
"consecutive" is decided by a gap threshold fitted to the data rather than guessed.

    from scripts.sessionize import fit_session_threshold, sessionize, merge_messages

    threshold = fit_session_threshold(df["timestamp"], df["author"])
    df["session_id"] = sessionize(df, "timestamp", "author", threshold)
    units = merge_messages(df, ["author", "session_id"])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from goad_toolkit.analytics import DistributionFitter, FitResult
from goad_toolkit.distributions import DistributionRegistry


def fit_session_threshold(
    timestamp: pd.Series, author: pd.Series, tail: float = 0.05
) -> float:
    """Fit an exponential to each author's within-burst gaps, and read off a threshold.

    Same technique lesson 4 used on gaps within an hour: fit `exponential`, and treat a gap
    longer than the burst regime would produce with probability `tail` as the start of a new
    session rather than a continuation. Zero-second gaps (two messages sharing a timestamp,
    a resolution artefact rather than an instant reply) and gaps over an hour (plainly a
    return, not a burst) are excluded from the fit, exactly as lesson 4 excluded them.

    Args:
        timestamp: one timestamp per message.
        author: one author per message, aligned with `timestamp`.
        tail: the burst-regime probability mass treated as "still the same session".

    Returns:
        A threshold in seconds: gaps longer than this start a new session.
    """
    frame = pd.DataFrame(
        {"timestamp": pd.to_datetime(timestamp), "author": author}
    ).sort_values(["author", "timestamp"])
    gaps = frame.groupby("author")["timestamp"].diff().dt.total_seconds()
    burst = gaps[(gaps > 0) & (gaps < 3600)]

    registry = DistributionRegistry()
    fit = DistributionFitter(registry, seed=42).fit_distribution("exponential", burst.to_numpy())
    if not isinstance(fit, FitResult):
        raise ValueError(f"exponential fit failed on the burst gaps: {fit}")

    # goad_toolkit types FitResult.params as a plain tuple; at runtime it's the
    # exponential's namedtuple, `loc` and `scale`.
    loc, scale = fit.params.loc, fit.params.scale  # ty: ignore[unresolved-attribute]
    return float(loc + scale * np.log(1 / tail))


def sessionize(
    data: pd.DataFrame, timestamp_col: str, author_col: str, threshold: float
) -> pd.Series:
    """Assign a session id to each row: a new id whenever the gap since that author's
    previous message exceeds `threshold`, or there is no previous message.

    Returns:
        One session id per row, aligned with `data`'s original index. Ids restart at 0
        for every author — pair with the author column to get a unique key.
    """
    order = data.sort_values([author_col, timestamp_col])
    gap = (
        pd.to_datetime(order[timestamp_col])
        .groupby(order[author_col])
        .diff()
        .dt.total_seconds()
    )
    new_session = gap.isna() | (gap > threshold)
    session_id = new_session.groupby(order[author_col]).cumsum() - 1
    return session_id.reindex(data.index)


def merge_messages(
    data: pd.DataFrame, group_cols: list[str], text_col: str = "message"
) -> pd.DataFrame:
    """Join every group's messages into one text unit, in their original order.

    Returns:
        One row per group: `group_cols`, the joined text, and `n_messages`.
    """
    ordered = data.sort_values(group_cols)
    return (
        ordered.groupby(group_cols)
        .agg(
            **{
                text_col: (text_col, lambda s: " ".join(s)),
                "n_messages": (text_col, "size"),
            }
        )
        .reset_index()
    )
