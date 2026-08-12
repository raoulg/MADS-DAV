#!/usr/bin/env python3
"""Publish the IRC showcase slice to the Hugging Face hub.

The slice is built by `build_irc_showcase.py` and lives at
`data/showcase/ubuntu_irc_days.parquet`. This script writes its dataset card and
uploads both to a hub dataset repo, so a student who has not cloned the data — or
who wants it outside this repo — can fetch it by name.

    uv run scripts/publish_irc_showcase.py                       # print the card, upload nothing
    uv run scripts/publish_irc_showcase.py --push                # create the repo and upload

Uploading needs a token with write access to the target namespace, from
`huggingface-cli login` or HF_TOKEN. Nothing is uploaded without `--push`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "showcase" / "ubuntu_irc_days.parquet"
DEFAULT_REPO = "pttrn-io/ubuntu-irc-days"

CARD = """---
license: cc0-1.0
task_categories:
- text-classification
language:
- en
- nl
tags:
- chat
- irc
- education
- time-series
pretty_name: Ubuntu IRC channel-days
size_categories:
- 1K<n<10K
---

# Ubuntu IRC channel-days

Five years of two Ubuntu IRC channels, one row per channel-day, used as the
showcase corpus for the MADS Data Analysis and Visualisation course.

## What one row is

One row is **one channel on one day**. The date is a column; the individual
message times sit inside the text, one message per line:

| column | type | meaning |
|---|---|---|
| `created` | datetime64[ms] | the day |
| `channel` | string | `#ubuntu-uk` or `#ubuntu-nl` |
| `text` | string | every message that day, `[HH:MM] <nick> message` per line |

Turning that into `(timestamp, author, message)` rows is deliberately **not** done
here — it is the first exercise of the course, and shipping it pre-parsed would
remove the lesson. Expect to write a regex.

## The slice

{stats}

Selected from [`common-pile/ubuntu_irc`](https://huggingface.co/datasets/common-pile/ubuntu_irc)
by `scripts/build_irc_showcase.py` in the course repository, taking the two
channels above between {start} and {end}.

`#ubuntu-uk` is the primary channel: English, busy enough for daily and weekly
structure to be visible, with a release-day spike on every Ubuntu release in the
window. `#ubuntu-nl` is deliberately thin — it is what a quiet channel looks
like, and what small sample sizes cost you.

The global `#ubuntu` channel is the control for the daily-cycle contrast: it runs
24/7 across every timezone, so its peak-to-trough ratio is close to flat where a
regional channel's is dramatic. Its raw text is ~400 MB, so only an hourly
aggregate is distributed, in the course repository rather than here.

## Loading it

```python
import pandas as pd

days = pd.read_parquet(
    "hf://datasets/{repo}/ubuntu_irc_days.parquet"
)
```

## Provenance and licence

Derived from `common-pile/ubuntu_irc`, which is Public Domain; this slice is
released under CC0-1.0 to match. The messages are public IRC logs from support
channels, published by the Ubuntu project. They are real people's words: quote
them as you would any published archive, and do not use them to profile
individuals.
"""


def stats_table(days: pd.DataFrame) -> str:
    """Per-channel volume and cycle strength, measured off the shipped file."""
    rows = ["| channel | days | messages | per day | peak/trough hour |", "|---|---|---|---|---|"]
    for channel, group in days.groupby("channel"):
        hours = group["text"].str.extractall(r"^\[(\d{2}):\d{2}\]", flags=8)[0].astype(int)
        profile = hours.value_counts().sort_index()
        share = profile / profile.sum()
        total = int(profile.sum())
        rows.append(
            f"| `{channel}` | {len(group):,} | {total:,} | {total / len(group):.0f} | "
            f"{share.max() / share.min():.1f}× |"
        )
    return "\n".join(rows)


def build_card(days: pd.DataFrame, repo: str) -> str:
    return CARD.format(
        stats=stats_table(days),
        start=days["created"].min().date(),
        end=days["created"].max().date(),
        repo=repo,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="target hub dataset repo")
    parser.add_argument("--push", action="store_true", help="actually create and upload")
    args = parser.parse_args()

    if not PARQUET.exists():
        raise SystemExit(
            f"{PARQUET} is missing. Build it first:\n"
            f"    uv run scripts/build_irc_showcase.py"
        )

    days = pd.read_parquet(PARQUET)
    card = build_card(days, args.repo)

    if not args.push:
        print(card)
        print(f"\n--- dry run: nothing uploaded. Add --push to create {args.repo}.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=PARQUET,
        path_in_repo="ubuntu_irc_days.parquet",
        repo_id=args.repo,
        repo_type="dataset",
    )
    print(f"uploaded {PARQUET.name} ({PARQUET.stat().st_size / 1e6:.1f} MB) to {args.repo}")


if __name__ == "__main__":
    main()
