"""Day-by-hour tables for the time lessons, in a form notebooks can import.

Comparing a day's hourly *shape* against a baseline starts from the same reshape
every time: one row per day, one column per hour of that day, message counts in
the cells. That pivot is plumbing, not the point of lesson 3 — what the lesson
teaches is what to do with the table (normalise each day to its own shares, then
average), and that step stays in the notebook.

    from scripts.profiles import day_hour_table

    table = day_hour_table(messages)                      # one row per raw message
    table = day_hour_table(hourly, values="messages")     # already-counted hourly totals
"""

from __future__ import annotations

import pandas as pd


def day_hour_table(
    data: pd.DataFrame,
    date: str = "date",
    hour: str = "hour",
    values: str | None = None,
) -> pd.DataFrame:
    """One row per day, one column per hour (0-23), message counts in the cells.

    Parameters:
    -----------
    data : pd.DataFrame
        Either one row per message, or one row per (day, hour) with a count column.
    date, hour : str
        Column names for the day and the hour of day.
    values : str | None
        None counts rows; a column name sums that column instead, for frames
        that already carry hourly totals.

    Hours with no messages become 0, so every row spans the full 24 columns and
    profiles built from two tables line up hour for hour.
    """
    if values is None:
        counts = data.groupby([date, hour]).size()
    else:
        counts = data.groupby([date, hour])[values].sum()
    # A one-level unstack of a Series is always a DataFrame; the stubs return a union.
    table: pd.DataFrame = counts.unstack(
        hour, fill_value=0
    )  # ty: ignore[invalid-assignment]
    return table.reindex(columns=range(24), fill_value=0)
