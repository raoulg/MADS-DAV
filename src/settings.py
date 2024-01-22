from dataclasses import dataclass
from pathlib import Path

@dataclass
class Regexes:
    timestamp = r"(?<=\[)\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}(?=\])"
    author = r"(?<=\]\s)(.*?)(?=:)"
    clear = r"\[\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}\]\s[~a-zA-Z\s]+:"

@dataclass
class Folders:
    raw: Path
    processed: Path
    datafile: Path