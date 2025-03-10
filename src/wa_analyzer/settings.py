from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class BaseRegexes(BaseModel):
    timestamp: str
    author: str
    message: str


iosRegexes = BaseRegexes(
    timestamp=r"\[(.+?)]\s.+?:.+",
    author=r"\[.+?]\s(.+?):.+",
    message=r"\[.+?]\s.+?:(.+)",
)


androidRegexes = BaseRegexes(
    timestamp=r"(.+?)\s-\s.+?:.+",
    author=r".+?\s-\s(.+?):.+",
    message=r".+?\s-\s.+?:(.*)",
)

oldRegexes = BaseRegexes(
    timestamp=r"^\d{1,2}/\d{1,2}/\d{2}, \d{2}:\d{2}",
    author=r"(?<=\s-\s)(.*?)(?=:)",
    message=r"^\d{1,2}/\d{1,2}/\d{2}, \d{2}:\d{2}[-~a-zA-Z0-9\s]+:",
)

csvRegexes = BaseRegexes(
    timestamp=r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
    author=r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},([^,]+),",
    message=r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},[^,]+,(.+)",
)


@dataclass
class Folders:
    raw: Path
    processed: Path
    datafile: Path


@dataclass
class NetworkAnalysisConfig:
    """Configuration for network analysis."""
    response_window: int = 3600  # Default 1 hour in seconds
    time_window: int = 60 * 60 * 24 * 30 * 2  # Default 2 months in seconds
    time_overlap: int = 60 * 60 * 24 * 30  # Default 1 month in seconds
    edge_weight_multiplier: float = 1.0
    min_edge_weight: float = 0.5
