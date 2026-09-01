"""Where data comes from, so a notebook does not open with four cells of path handling.

Two sources, and every lesson uses both:

- `load_showcase(...)` — the curated datasets in `data/showcase/`, committed to the repo.
  A technique is demonstrated on one of these first, on data chosen so the pattern is
  certain to be there.
- `load_own_chat()` — your own exported chat, after you have run the preprocessor. This is
  where the technique gets used, and where a null result is a legitimate answer.

`load_own_chat()` raises when there is no chat to load. A your-turn notebook without data
has nothing to test, and a notebook that quietly runs on nothing teaches nothing — the
error message says exactly which step of the setup is missing.
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
    "penguins_raw": "the same birds as published, with the isotope columns and full species names",
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


# Datasets too large to keep in the repository, with the hub copy to fall back on.
HUB_DATASETS = {
    "ubuntu_irc": ("pttrn-io/ubuntu-irc-days", "ubuntu_irc_days.parquet"),
}


def list_showcase() -> pd.DataFrame:
    """Show which showcase datasets exist and what each one is for."""
    rows = [
        {
            "dataset": name,
            "what it shows": why,
            "on disk": _showcase_path(name).exists(),
        }
        for name, why in SHOWCASE_DATASETS.items()
    ]
    return pd.DataFrame(rows)


def _showcase_path(name: str) -> Path:
    """Resolve a showcase name to a file, preferring parquet where both exist."""
    parquet = (
        SHOWCASE / f"{name}_days.parquet"
        if name == "ubuntu_irc"
        else SHOWCASE / f"{name}.parquet"
    )
    return parquet if parquet.exists() else SHOWCASE / f"{name}.csv"


def _download_from_hub(name: str) -> Path:
    """Fetch a showcase dataset that is published on the hub.

    Only reached when the committed copy is absent, so a lecture never depends on
    the room's wifi. The download is cached by `huggingface_hub`, so it happens
    once per machine.
    """
    repo, filename = HUB_DATASETS[name]
    from huggingface_hub import hf_hub_download

    logger.info(f"{name} is not in data/showcase, fetching it from {repo}")
    return Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset"))


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
    if not path.exists() and name in HUB_DATASETS:
        path = _download_from_hub(name)
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


def load_own_chat(
    filename: Optional[str] = None, verbose: bool = True
) -> pd.DataFrame:
    """Load your own preprocessed chat.

    Reads `config.toml` for the `current` key — the parquet file written by notebook 01.3 —
    unless `filename` is given, which loads that file from `data/processed/` directly and
    skips `config.toml` entirely.

    Args:
        filename: load this file from `data/processed/` instead of consulting
            `config.toml`'s `current` key. Use it to point at a specific export without
            editing the config, e.g. to compare two of your own chats side by side.
        verbose: log a success line when the chat loads.

    Returns:
        Your chat as a DataFrame.

    Raises:
        FileNotFoundError: when there is nothing to load yet, saying which setup step is
            missing. A your-turn notebook without data has nothing to test, so it stops
            here rather than running on nothing.
    """
    if filename is not None:
        datafile = PROCESSED / filename
        if not datafile.exists():
            raise FileNotFoundError(
                f"{datafile} does not exist. Run the preprocessor "
                "(`analyzer --device ios` or android) and notebook 01.3 first."
            )
    else:
        if not CONFIG.exists():
            raise FileNotFoundError(
                "No config.toml, so there is no chat of your own to load yet. "
                "See the README's 'Run the preprocessor' section: copy "
                "config.example.toml to config.toml, export a chat, run "
                "`analyzer --device ios` (or android), then notebook 01.3."
            )

        with CONFIG.open("rb") as f:
            config = tomllib.load(f)

        current = config.get("current", "")
        datafile = PROCESSED / current
        if not current or not datafile.exists():
            raise FileNotFoundError(
                f"config.toml points `current` at '{current}', which is not in "
                f"{PROCESSED}. Run notebook 01.3 to produce it, then set `current` to "
                f"the filename it writes."
            )

    data = (
        pd.read_parquet(datafile)
        if datafile.suffix in {".parq", ".parquet"}
        else pd.read_csv(datafile)
    )
    if verbose:
        logger.success(
            f"Loaded {len(data):,} of your own messages from {datafile.name}"
        )
    return data
