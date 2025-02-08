from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class Folders:
    raw: Path
    processed: Path
    datafile: Path
