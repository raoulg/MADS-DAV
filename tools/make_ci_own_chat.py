"""Give CI a chat of its own, so the your-turn notebooks have data to run on.

`load_own_chat()` raises when there is no export — a your-turn notebook without data
has nothing to test. On a student's machine the data comes from their real export; in
CI it comes from here: a synthetic chat with the same schema the preprocessor writes,
plus enough structure (daily rhythm, weekend shift, a few event spikes, per-author
styles) that every lesson's technique has something to find.

    uv run python tools/make_ci_own_chat.py

Writes `data/processed/ci-fixture.parq` and, only when none exists yet, a
`config.toml` pointing `current` at it. An existing config.toml is never touched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CONFIG = ROOT / "config.toml"
FILENAME = "ci-fixture.parq"

AUTHORS = {
    # name: (weight, peak_hour, verbosity)
    "Alex": (1.0, 9, 60),
    "Sam": (0.9, 13, 35),
    "Robin": (0.7, 20, 90),
    "Kim": (0.6, 21, 25),
    "Charlie": (0.5, 11, 45),
    "Noor": (0.4, 17, 70),
    "Jesse": (0.3, 22, 30),
    "Fatima": (0.2, 8, 55),
}

OPENERS = [
    "morning!", "hey everyone", "goedemorgen", "quick question:", "ok so",
    "did anyone see this?", "update:", "haha", "wait what", "hmm",
]
BODIES = [
    "are we still on for tonight",
    "check this out https://example.com/article",
    "I completely forgot about the deadline",
    "kan iemand mij de notulen sturen",
    "that meeting could have been an email",
    "the train is delayed again, story of my life",
    "who is bringing the cake tomorrow",
    "just pushed the fix, can someone test it",
    "het regent alweer, typisch",
    "I found a better route via the park",
    "does thursday work for everyone or should we move it",
    "the photos from last weekend are online now https://example.com/album",
    "remember to bring your laptop",
    "lunch at the usual place",
    "this thread is getting out of hand",
]
CLOSERS = ["", " 😂", " 👍", " ❤️", "?", "!", " (sorry for the spam)", " 🎉"]


def build_fixture(seed: int = 20260901) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-09-01", "2026-05-31", freq="D")
    # a few spike days, the kind 03.3 asks students to go looking for
    events = {pd.Timestamp("2025-12-25"), pd.Timestamp("2026-01-01"),
              pd.Timestamp("2026-03-14")}

    rows = []
    for day in days:
        weekend = day.dayofweek >= 5
        for author, (weight, peak, verbosity) in AUTHORS.items():
            base = 6 * weight * (1.6 if weekend else 1.0)
            if day in events:
                base *= 4
            n = rng.poisson(base)
            hours = np.clip(rng.normal(peak, 3.0, n), 0, 23).astype(int)
            for hour in hours:
                minute = int(rng.integers(0, 60))
                second = int(rng.integers(0, 60))
                text = (
                    str(rng.choice(OPENERS)) + " " + str(rng.choice(BODIES))
                    + str(rng.choice(CLOSERS))
                )
                # verbosity: some authors pad their messages
                if rng.random() < verbosity / 200:
                    text = text + " " + str(rng.choice(BODIES))
                rows.append(
                    {
                        "timestamp": day + pd.Timedelta(
                            hours=int(hour), minutes=minute, seconds=second
                        ),
                        "author": author,
                        "message": text,
                    }
                )

    chat = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    chat["timestamp"] = chat["timestamp"].dt.tz_localize("UTC")

    # the derived columns the preprocessor writes
    chat["anon_author"] = chat["author"].map(
        {name: f"user-{i:02d}" for i, name in enumerate(AUTHORS)}
    )
    emoji = {"😂", "👍", "❤️", "🎉"}
    chat["has_emoji"] = chat["message"].map(
        lambda text: any(char in text for char in emoji)
    )
    top_authors = chat["author"].value_counts().head(3).index
    chat["is_topk"] = chat["author"].isin(top_authors)
    chat["message_length"] = chat["message"].str.len()
    chat["has_link"] = chat["message"].str.contains(r"https?://")
    chat["hour"] = chat["timestamp"].dt.hour
    chat["day_of_week"] = chat["timestamp"].dt.dayofweek
    chat["timestamp_category"] = pd.cut(
        chat["hour"],
        bins=[-1, 6, 9, 17, 22, 24],
        labels=["night", "morning", "worktimes", "evening", "late"],
        ordered=False,
    )
    return chat


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    chat = build_fixture()
    target = PROCESSED / FILENAME
    chat.to_parquet(target)
    print(f"wrote {len(chat):,} messages, {chat.author.nunique()} authors -> {target}")

    if CONFIG.exists():
        print(f"{CONFIG} already exists; left untouched")
    else:
        CONFIG.write_text(f'current = "{FILENAME}"\n')
        print(f"wrote {CONFIG} pointing current at {FILENAME}")


if __name__ == "__main__":
    main()
