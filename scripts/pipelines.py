"""The IRC parsing pipeline lesson 1 derives, in a form later lessons can import.

Lesson 1 writes `ParseIRCLines` from scratch — that derivation is the exercise and
it stays in the notebook. This module is where the same code lives afterwards, so
lesson 2 onwards can reuse the parsed frame instead of re-deriving it with a copy
of the regex that then drifts.

    from scripts.pipelines import build_irc_pipeline

    enriched = build_irc_pipeline().apply(load_showcase("ubuntu_irc"))
"""

from __future__ import annotations

import re

import pandas as pd
from goad_toolkit.datatransforms import (
    Pipeline,
    RegexFeature,
    TimeFeatures,
    TransformBase,
)
from loguru import logger


class ParseIRCLines(TransformBase):
    """Turn one-row-per-day IRC logs into one row per message.

    Combines the `<nick>` pattern from 1.2 with the `/me` action-line fallback from 1.3, and
    replaces 1.3's manual coverage `print`s with a `loguru` log line.
    """

    LINE = re.compile(r"^\[(\d{2}):(\d{2})\]\s+<(\S+)>\s+(.*)$")
    ACTION = re.compile(r"^\[(\d{2}):(\d{2})\]\s+\*\s+(\S+)\s+(.*)$")

    def transform(self, data: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        rows, missing = [], 0
        for row in data.itertuples():
            for line in getattr(row, text_column).split("\n"):
                if not line.strip():
                    continue
                m = self.LINE.match(line) or self.ACTION.match(line)
                if m:
                    hh, mm, author, message = m.groups()
                    # itertuples() rows carry every column dynamically, so no stub can
                    # know `.created` and `.channel` exist ahead of time.
                    rows.append(
                        (
                            row.created,  # ty: ignore[unresolved-attribute]
                            row.channel,  # ty: ignore[unresolved-attribute]
                            int(hh),
                            int(mm),
                            author,
                            message,
                            bool(self.ACTION.match(line)),
                        )
                    )
                else:
                    missing += 1

        columns = pd.Index(
            ["date", "channel", "hh", "mm", "author", "message", "is_action"]
        )
        parsed = pd.DataFrame(rows, columns=columns)
        coverage = len(parsed) / (len(parsed) + missing)
        logger.info(
            f"{self.name}: parsed {len(parsed):,} messages, {coverage:.2%} of lines "
            f"({missing:,} unparsed)"
        )
        return parsed


def build_irc_pipeline() -> Pipeline:
    """Assemble lesson 1's full pipeline: parse, then enrich.

    A factory rather than a module-level instance. A shared `Pipeline` would let one
    notebook's `pipeline["mentions"] = {...}` change what every other notebook imports,
    and goad documents that dict-style override as the way to retune a step.

    Returns:
        `ParseIRCLines` → `TimeFeatures` → three `RegexFeature` steps, ready for
        `.apply(load_showcase("ubuntu_irc"))`.
    """
    pipeline = Pipeline()
    pipeline.add(ParseIRCLines)
    pipeline.add(TimeFeatures, column="date")
    pipeline.add(
        RegexFeature,
        name="urls",
        column="message",
        pattern=r"https?://\S+",
        feature="has_url",
        mode="has",
    )
    pipeline.add(
        RegexFeature,
        name="questions",
        column="message",
        pattern=r"\?",
        feature="n_question",
        mode="count",
    )
    pipeline.add(
        RegexFeature,
        name="mentions",
        column="message",
        pattern=r"^(\S+)[:,]\s",
        feature="addressed_to",
        mode="extract",
    )
    return pipeline
