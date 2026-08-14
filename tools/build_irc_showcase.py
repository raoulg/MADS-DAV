#!/usr/bin/env python3
"""Build the IRC showcase slice used by lessons 1-6.

The course needs a chat-shaped dataset that is not the student's own: multi-author,
timestamped, bursty, long-tailed. `common-pile/ubuntu_irc` is that, and it is Public Domain.

One source row is one channel on one day: the date is a column, the times sit inside the
text, one message per line. Turning that into (timestamp, author, message) rows is lesson 1's
exercise, so this script deliberately stops before parsing and ships the raw day documents.

    uv run scripts/build_irc_showcase.py                 # the default slice
    uv run scripts/build_irc_showcase.py --check         # ...and verify it

The corpus is ordered by DATE, not by channel, so a channel's rows are scattered through
every parquet row group. Selecting a date window prunes row groups; selecting a channel does
not. That is why the window comes first.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "datasets/common-pile/ubuntu_irc@refs/convert/parquet/default/train"

# #ubuntu-uk is the primary: English, ~200 messages/day, a sharp weekly cycle, and a
# release-day spike on every release in the window. #ubuntu-nl is deliberately thin — it is
# what a quiet chat looks like, and lessons 2 and 4 use it to show what small n costs.
DEFAULT_CHANNELS = ("#ubuntu-uk", "#ubuntu-nl")
DEFAULT_START, DEFAULT_END = "2013-01-01", "2017-12-31"

# #ubuntu is global and 24/7, which makes it the control for the daily-cycle contrast. Its
# raw text is ~400 MB, so only the hourly aggregate is vendored.
CONTROL_CHANNEL = "#ubuntu"


def fetch_days(start: str, end: str, channels: set[str]) -> pd.DataFrame:
    """Fetch raw channel-day documents, reading only row groups inside the date window."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if not isinstance(start_ts, pd.Timestamp) or not isinstance(end_ts, pd.Timestamp):
        raise ValueError(f"could not parse --start/--end as dates: {start!r}, {end!r}")
    fs = HfFileSystem()
    frames, read, total = [], 0, 0

    # detail=True makes fs.ls return dicts, not the plain filenames the stub assumes.
    listing = fs.ls(REPO, detail=True)
    for name in sorted(f["name"] for f in listing):  # ty: ignore[invalid-argument-type]
        with fs.open(name, "rb") as fh:
            pf = pq.ParquetFile(fh)
            total += pf.num_row_groups
            keep = [
                g
                for g in range(pf.num_row_groups)
                if _in_window(pf, g, start_ts, end_ts)
            ]
            if not keep:
                continue
            read += len(keep)
            df = pf.read_row_groups(
                keep, columns=["created", "text", "metadata"]
            ).to_pandas()
            df["channel"] = df["metadata"].map(lambda m: m["channel"])
            frames.append(
                df[
                    df["channel"].isin(channels)
                    & df["created"].between(start_ts, end_ts)
                ][["created", "channel", "text"]]
            )

    print(f"read {read}/{total} row groups")
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("created")
        .reset_index(drop=True)
    )


def _in_window(
    pf: pq.ParquetFile, group: int, start: pd.Timestamp, end: pd.Timestamp
) -> bool:
    """Whether a row group's `created` statistics overlap the window."""
    rg = pf.metadata.row_group(group)
    for c in range(rg.num_columns):
        col = rg.column(c)
        if col.path_in_schema == "created":
            st = col.statistics
            if st is None:
                return True
            return pd.Timestamp(st.min) <= end and pd.Timestamp(st.max) >= start
    return True


def hourly_profile(days: pd.DataFrame) -> pd.DataFrame:
    """Messages per hour-of-day, counted straight off the raw text.

    Only the leading [HH:MM] is needed, so this sidesteps the full parse on purpose — the
    control channel is used for one comparison and does not need to be shipped in full.
    """
    hours = days["text"].str.extractall(r"^\[(\d{2}):\d{2}\]", flags=8)[0]
    counts = hours.astype(int).value_counts().sort_index()
    return counts.rename_axis("hour").rename("messages").reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS))
    ap.add_argument("--out", type=Path, default=Path("data/showcase"))
    ap.add_argument(
        "--control",
        action="store_true",
        help=f"also build the {CONTROL_CHANNEL} hourly profile",
    )
    ap.add_argument(
        "--check", action="store_true", help="report volume and cycle strength"
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    wanted = set(args.channels) | ({CONTROL_CHANNEL} if args.control else set())
    days = fetch_days(args.start, args.end, wanted)

    slice_ = days[days["channel"].isin(args.channels)]
    target = args.out / "ubuntu_irc_days.parquet"
    slice_.to_parquet(target, index=False)
    size = target.stat().st_size / 1e6
    print(f"\n{len(slice_)} channel-days -> {target} ({size:.1f} MB)")
    print(slice_.groupby("channel").size().to_string())

    if args.control:
        control = days[days["channel"] == CONTROL_CHANNEL]
        profile = hourly_profile(control)
        path = args.out / "ubuntu_irc_control_hourly.csv"
        profile.to_csv(path, index=False)
        print(f"{CONTROL_CHANNEL} hourly profile -> {path}")

    if args.check:
        for ch, g in slice_.groupby("channel"):
            prof = hourly_profile(g)
            share = prof["messages"] / prof["messages"].sum()
            total = int(prof["messages"].sum())
            print(
                f"  {ch:12s} {total:>9,} messages  {total / len(g):6.0f}/day  "
                f"peak/trough hour {share.max() / share.min():6.1f}x"
            )


if __name__ == "__main__":
    main()
