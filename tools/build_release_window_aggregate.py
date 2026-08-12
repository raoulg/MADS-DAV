#!/usr/bin/env python3
"""Build the release-vs-ordinary hourly aggregate lesson 3 needs for Finding 1.

`#ubuntu` and `#ubuntu-it` are control channels: lesson 3 only needs their
per-hour message counts, split by whether the day was an Ubuntu release day,
not the full per-message text. So — same reasoning as the `#ubuntu` control
file `build_irc_showcase.py` already vendors — only the aggregate ships.
`#ubuntu-uk` needs no separate file: its full text is already vendored, and
the notebook derives its own hourly split from that.

Every release in the window falls on a Thursday, so "ordinary" here means
ordinary *Thursdays* — the fair comparison, not all days including weekends.

    uv run scripts/build_release_window_aggregate.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "datasets/common-pile/ubuntu_irc@refs/convert/parquet/default/train"
START, END = "2013-01-01", "2017-12-31"
CHANNELS = {"#ubuntu", "#ubuntu-it"}
OUT = Path("data/showcase/ubuntu_irc_release_hourly.csv")

RELEASE_DATES = pd.to_datetime(
    [
        "2013-04-25", "2013-10-17",  # 13.04, 13.10
        "2014-04-17", "2014-10-23",  # 14.04, 14.10
        "2015-04-23", "2015-10-22",  # 15.04, 15.10
        "2016-04-21", "2016-10-13",  # 16.04, 16.10
        "2017-04-13", "2017-10-19",  # 17.04, 17.10
    ]
)


def _in_window(pf: pq.ParquetFile, group: int, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    rg = pf.metadata.row_group(group)
    for c in range(rg.num_columns):
        col = rg.column(c)
        if col.path_in_schema == "created":
            st = col.statistics
            if st is None:
                return True
            return pd.Timestamp(st.min) <= end and pd.Timestamp(st.max) >= start
    return True


def fetch_days(start: str, end: str, channels: set[str]) -> pd.DataFrame:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    fs = HfFileSystem()
    frames = []
    for name in sorted(f["name"] for f in fs.ls(REPO, detail=True)):
        with fs.open(name, "rb") as fh:
            pf = pq.ParquetFile(fh)
            keep = [g for g in range(pf.num_row_groups) if _in_window(pf, g, start_ts, end_ts)]
            if not keep:
                continue
            df = pf.read_row_groups(keep, columns=["created", "text", "metadata"]).to_pandas()
            df["channel"] = df["metadata"].map(lambda m: m["channel"])
            frames.append(
                df[df["channel"].isin(channels) & df["created"].between(start_ts, end_ts)][
                    ["created", "channel", "text"]
                ]
            )
    return pd.concat(frames, ignore_index=True).sort_values("created").reset_index(drop=True)


def hourly_lines(days: pd.DataFrame) -> pd.DataFrame:
    """One row per message line, with its date and hour, exploded from the day-blob text."""
    rows = []
    for created, text in zip(days["created"], days["text"]):
        for m in re.finditer(r"^\[(\d{2}):\d{2}\]", text, flags=re.MULTILINE):
            rows.append((created, int(m.group(1))))
    return pd.DataFrame(rows, columns=["date", "hour"])


def main() -> None:
    days = fetch_days(START, END, CHANNELS)
    print(f"fetched {len(days)} channel-days")

    # Per-day, per-hour counts — not pre-pooled into ordinary/release totals.
    # Pooling lets a handful of high-volume days dominate a 10-day release
    # sample; keeping the per-day grain lets the notebook choose (and show)
    # that methodology explicitly instead of baking it into the vendored file.
    records = []
    for channel in sorted(CHANNELS):
        lines = hourly_lines(days[days["channel"] == channel])
        lines["is_release"] = lines["date"].isin(RELEASE_DATES)
        is_thursday = lines["date"].dt.day_name() == "Thursday"
        keep = lines[is_thursday | lines["is_release"]]

        for date, day_lines in keep.groupby("date"):
            counts = day_lines["hour"].value_counts()
            for hour in range(24):
                records.append(
                    {
                        "channel": channel,
                        "date": date.date().isoformat(),
                        "is_release": bool(day_lines["is_release"].iloc[0]),
                        "hour": hour,
                        "messages": int(counts.get(hour, 0)),
                    }
                )

    out = pd.DataFrame.from_records(records)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {len(out)} rows -> {OUT}")
    print(out.groupby(["channel", "is_release"])["date"].nunique().to_string())


if __name__ == "__main__":
    main()
