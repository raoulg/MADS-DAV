"""Day-by-hour tables for the time lessons, as a pipeline step notebooks can import.

Comparing a day's hourly *shape* against a baseline starts from the same reshape
every time: one row per day, one column per hour of that day, message counts in
the cells. That pivot is plumbing, not the point of lesson 3 — what the lesson
teaches is what to do with the table (normalise each day to its own shares, then
average), and that step stays in the notebook, as a transform defined where it
can be defended.

    from scripts.profiles import DayHourTable

    pipeline = Pipeline().add(Filter, expr="is_release").add(DayHourTable)
    table = pipeline.apply(messages)                      # one row per raw message
    Pipeline().add(DayHourTable, values="messages")       # already-counted totals
"""

from __future__ import annotations

import pandas as pd
from goad_toolkit.datatransforms import TransformBase


class DayHourTable(TransformBase):
    """One row per day, one column per hour (0-23), message counts in the cells.

    Takes either one row per message, or one row per (day, hour) with a count
    column named by `values` — `values=None` counts rows, a column name sums
    that column instead.

    Hours with no messages become 0, so every row spans the full 24 columns and
    profiles built from two tables line up hour for hour.
    """

    def transform(
        self,
        data: pd.DataFrame,
        date: str = "date",
        hour: str = "hour",
        values: str | None = None,
    ) -> pd.DataFrame:
        if values is None:
            counts = data.groupby([date, hour]).size()
        else:
            counts = data.groupby([date, hour])[values].sum()
        # A one-level unstack of a Series is always a DataFrame; the stubs return a union.
        table: pd.DataFrame = counts.unstack(
            hour, fill_value=0
        )  # ty: ignore[invalid-assignment]
        return table.reindex(columns=range(24), fill_value=0)
