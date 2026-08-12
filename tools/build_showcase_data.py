#!/usr/bin/env python3
"""Vendor the small showcase datasets into data/showcase/.

Lessons 2-6 demonstrate each technique on a dataset chosen so the pattern is unmissable,
before the student points the same code at their own chat. Those datasets are small and
famous, and every one of them is here because it shows exactly one thing clearly.

They are committed rather than downloaded on demand: `sns.load_dataset` fetches from GitHub
on first use, and a lecture should not depend on the room's wifi.

    uv run scripts/build_showcase_data.py
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
import requests
import seaborn as sns

# Each entry says what the dataset is FOR, because "why this one" is the part that gets lost.
SEABORN = {
    "penguins": "lesson 2 — clean categorical comparison: 3 species x 2 sexes x 3 islands",
    "titanic": "lesson 2 — the failure modes: high-cardinality bars, proportions across unequal groups",
    "flights": "lesson 3 — textbook trend plus multiplicative seasonality in 144 points",
    "taxis": "lesson 4 — a genuine long tail in the wild; mean and median disagree",
    "mpg": "lesson 5 — scatter with hue/size, and an obviously non-linear relation",
    "anscombe": "lesson 5 — identical summary statistics, four different pictures",
    "diamonds": "lesson 5 — correlation matrix, and heteroscedasticity in the scatter",
}

# The Palmer archive as published, which seaborn's `penguins` is a tidied subset of: it keeps
# the isotope measurements and the full species names. Lesson 5.2 and the dashboards correlate
# the isotopes against the body measurements, so they need this one rather than the tidy copy.
PALMER_RAW = (
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/"
    "penguins_raw.csv"
)

DATASAURUS_BASE = (
    "https://raw.githubusercontent.com/jumpingrivers/datasauRus/main/inst/extdata"
)
# From the same Autodesk research as the Datasaurus. The Simpsons set is the continuous
# counterpart to the Berkeley table below: a scatter whose overall slope reverses once the
# groups are separated.
FROM_DATASAURUS = {
    "datasaurus": ("DatasaurusDozen-Long.tsv",
                   "lesson 5 — 13 datasets, same statistics, one of them a dinosaur"),
    "simpsons_paradox": ("SimpsonsParadox-Long.tsv",
                         "lesson 2 — Simpson's paradox as a scatter, not a table"),
}

# Bickel, Hammel & O'Connell (1975), Science 187:398-404. The six largest departments of the
# 1973 UC Berkeley graduate admissions round -- the standard Simpson's paradox table.
BERKELEY = [
    # dept, gender, applied, admitted
    ("A", "men", 825, 512), ("A", "women", 108, 89),
    ("B", "men", 560, 353), ("B", "women", 25, 17),
    ("C", "men", 325, 120), ("C", "women", 593, 202),
    ("D", "men", 417, 138), ("D", "women", 375, 131),
    ("E", "men", 191, 53), ("E", "women", 393, 94),
    ("F", "men", 373, 22), ("F", "women", 341, 24),
]


def berkeley() -> pd.DataFrame:
    df = pd.DataFrame(BERKELEY, columns=["department", "gender", "applied", "admitted"])
    df["rate"] = (df["admitted"] / df["applied"]).round(4)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/showcase"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for name, why in SEABORN.items():
        df = sns.load_dataset(name)
        path = args.out / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"{name:12s} {len(df):>6,} rows  {path.stat().st_size / 1024:>7.0f} KB   {why}")

    resp = requests.get(PALMER_RAW, timeout=30)
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text))
    path = args.out / "penguins_raw.csv"
    raw.to_csv(path, index=False)
    print(f"{'penguins_raw':12s} {len(raw):>6,} rows  {path.stat().st_size / 1024:>7.0f} KB   "
          "lesson 5 — the Palmer archive, isotopes included")

    for name, (remote, why) in FROM_DATASAURUS.items():
        resp = requests.get(f"{DATASAURUS_BASE}/{remote}", timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), sep="\t")
        path = args.out / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"{name:12s} {len(df):>6,} rows  {path.stat().st_size / 1024:>7.0f} KB   {why}")

    b = berkeley()
    b.to_csv(args.out / "berkeley_admissions.csv", index=False)
    agg = b.groupby("gender")[["applied", "admitted"]].sum()
    agg["rate"] = agg["admitted"] / agg["applied"]
    print(f"{'berkeley':12s} {len(b):>6,} rows           "
          "lesson 2 — Simpson's paradox")
    print(f"{'':14s}aggregate: men {agg.loc['men', 'rate']:.1%}, women {agg.loc['women', 'rate']:.1%}")
    better = (b.pivot(index="department", columns="gender", values="rate")
              .assign(women_higher=lambda d: d["women"] > d["men"])["women_higher"])
    print(f"{'':14s}per department, women admitted at a higher rate in "
          f"{better.sum()}/{len(better)}")

    total = sum(p.stat().st_size for p in args.out.glob("*.csv"))
    print(f"\n{len(list(args.out.glob('*.csv')))} csv files, {total / 1e6:.1f} MB total")


if __name__ == "__main__":
    main()
