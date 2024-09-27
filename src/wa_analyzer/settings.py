from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


class BaseRegexes(BaseModel):
    timestamp: str
    author: str
    clear: str
    fmt: str


iosRegexes = BaseRegexes(
    timestamp=r"(?<=\[)\d{2}-\d{2}-\d{4},? \d{1,2}:\d{2}:\d{2}(?=\])",
    author=r"(?<=\]\s)(.*?)(?=:)",
    clear=r"\[.*].*:",
    fmt="%d-%m-%Y, %H:%M:%S",
)


androidRegexes = BaseRegexes(
    timestamp=r"^\d{2}-\d{2}-\d{4} \d{2}:\d{2}",
    author=r"(?<=\s-\s)(.*?)(?=:)",
    clear=r"^\d{2}-\d{2}-\d{4} \d{2}:\d{2}[-~a-zA-Z\s]+:",
    fmt="%d-%m-%Y %H:%M",
)

oldRegexes = BaseRegexes(
    timestamp=r"^\d{1,2}/\d{1,2}/\d{2}, \d{2}:\d{2}",
    author=r"(?<=\s-\s)(.*?)(?=:)",
    clear=r"^\d{1,2}/\d{1,2}/\d{2}, \d{2}:\d{2}[-~a-zA-Z0-9\s]+:",
    fmt="%m/%d/%y, %H:%M",
)


@dataclass
class Folders:
    raw: Path
    processed: Path
    datafile: Path
