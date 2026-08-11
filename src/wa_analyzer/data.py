"""Where data comes from, so a notebook does not open with four cells of path handling.

Two sources, and every lesson uses both:

- `load_showcase(...)` — the curated datasets in `data/showcase/`, committed to the repo.
  A technique is demonstrated on one of these first, on data chosen so the pattern is
  certain to be there.
- `load_own_chat()` — your own exported chat, after you have run the preprocessor. This is
  where the technique gets used, and where a null result is a legitimate answer.

`load_own_chat()` returns None rather than raising when there is no chat to load, so a
notebook runs top to bottom on the showcase half whether or not you have an export yet.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

# src/wa_analyzer/data.py -> repo root
ROOT = Path(__file__).resolve().parents[2]
SHOWCASE = ROOT / "data" / "showcase"
PROCESSED = ROOT / "data" / "processed"
CONFIG = ROOT / "config.toml"

SHOWCASE_DATASETS = {
    "penguins": "three species, two sexes, three islands — a clean categorical comparison",
    "titanic": "the categorical failure modes: many levels, very unequal group sizes",
    "berkeley_admissions": "Simpson's paradox, as the published 1973 table",
    "simpsons_paradox": "Simpson's paradox, as a scatter",
    "flights": "trend plus multiplicative seasonality, in 144 points",
    "taxis": "a genuine long tail — mean and median disagree",
    "mpg": "a relation that is obviously not a straight line",
    "anscombe": "four datasets, identical statistics, four different pictures",
    "datasaurus": "thirteen of them, and one is a dinosaur",
    "diamonds": "a correlation matrix, and a scatter that fans out",
    "ubuntu_irc": "five years of two Ubuntu IRC channels, one row per channel-day",
    "ubuntu_irc_control_hourly": "#ubuntu hourly totals, as a comparison for the above",
    "ubuntu_irc_release_hourly": (
        "#ubuntu and #ubuntu-it hourly totals, ordinary Thursday vs release day"
    ),
}


def list_showcase() -> pd.DataFrame:
    """Show which showcase datasets exist and what each one is for."""
    rows = [
        {"dataset": name, "what it shows": why, "on disk": _showcase_path(name).exists()}
        for name, why in SHOWCASE_DATASETS.items()
    ]
    return pd.DataFrame(rows)


def _showcase_path(name: str) -> Path:
    """Resolve a showcase name to a file, preferring parquet where both exist."""
    parquet = SHOWCASE / f"{name}_days.parquet" if name == "ubuntu_irc" else SHOWCASE / f"{name}.parquet"
    return parquet if parquet.exists() else SHOWCASE / f"{name}.csv"


def load_showcase(name: str) -> pd.DataFrame:
    """Load one of the curated showcase datasets by name.

    Args:
        name: a key of SHOWCASE_DATASETS, e.g. "penguins" or "ubuntu_irc".

    Returns:
        The dataset as a DataFrame.

    Raises:
        FileNotFoundError: if the file is missing, with the command that rebuilds it.
    """
    path = _showcase_path(name)
    if not path.exists():
        known = ", ".join(sorted(SHOWCASE_DATASETS))
        raise FileNotFoundError(
            f"No showcase dataset '{name}' at {path}.\n"
            f"Known datasets: {known}\n"
            f"If the file is simply missing, rebuild it with:\n"
            f"    uv run scripts/build_showcase_data.py"
        )
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_own_chat(verbose: bool = True) -> Optional[pd.DataFrame]:
    """Load your own preprocessed chat, or return None if there is not one yet.

    Reads `config.toml` for the `current` key — the parquet file written by notebook 01.

    Args:
        verbose: log an explanation when there is nothing to load.

    Returns:
        Your chat as a DataFrame, or None if `config.toml` or the file it names is missing.
        Returning None rather than raising is deliberate: the showcase half of every
        notebook must run whether or not you have an export.
    """
    if not CONFIG.exists():
        if verbose:
            logger.info(
                "No config.toml yet, so there is no chat of your own to load. "
                "The showcase half of this notebook runs without it. "
                "To use your own data: copy config.example.toml to config.toml, "
                "export a chat, and run `analyzer --device ios` (or android)."
            )
        return None

    with CONFIG.open("rb") as f:
        config = tomllib.load(f)

    current = config.get("current", "")
    datafile = PROCESSED / current
    if not current or not datafile.exists():
        if verbose:
            logger.warning(
                f"config.toml points `current` at '{current}', which is not in "
                f"{PROCESSED}. Run notebook 01 to produce it, then set `current` to the "
                f"filename it writes."
            )
        return None

    data = pd.read_parquet(datafile) if datafile.suffix in {".parq", ".parquet"} else pd.read_csv(datafile)
    if verbose:
        logger.success(f"Loaded {len(data):,} of your own messages from {datafile.name}")
    return data
