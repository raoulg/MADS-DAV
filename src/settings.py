from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class BaseRegexes(Protocol):
    timestamp: str
    author: str
    clear: str
    format: str


class iOS_Regexes(BaseRegexes):
    timestamp = r"(?<=\[)\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}(?=\])"
    author = r"(?<=\]\s)(.*?)(?=:)"
    clear = r"\[\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}\]\s[~a-zA-Z\s]+:"
    format = "%d-%m-%Y %H:%M:%S"


class Android_Regexes(BaseRegexes):
    timestamp = r"^\d{2}-\d{2}-\d{4} \d{2}:\d{2}"
    author = r"(?<=\s-\s)(.*?)(?=:)"
    clear = r"^\d{2}-\d{2}-\d{4} \d{2}:\d{2}[-~a-zA-Z\s]+:"
    format = "%d-%m-%Y %H:%M"


@dataclass
class Folders:
    raw: Path
    processed: Path
    datafile: Path
