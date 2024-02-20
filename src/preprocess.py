import re
import sys
from datetime import datetime
from pathlib import Path

# fix bug
import click
import pandas as pd
from loguru import logger

from src.settings import (BaseRegexes, Folders, androidRegexes, iosRegexes,
                          oldRegexes)

logger.remove()
logger.add("logs/logfile.log", rotation="1 week", level="DEBUG")
logger.add(sys.stderr, level="INFO")

logger.debug(f"Python path: {sys.path}")


class WhatsappPreprocessor:
    def __init__(self, folders: Folders, regexes: BaseRegexes):
        self.folders = folders
        self.regexes = regexes

    def __call__(self):
        records, _ = self.process()
        self.save(records)

    def save(self, records: list[tuple]) -> None:
        df = pd.DataFrame(records, columns=["timestamp", "author", "message"])
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        outfile = self.folders.processed / f"whatsapp-{now}.csv"
        logger.info(f"Writing to {outfile}")
        df.to_csv(outfile, index=False)
        logger.success("Done!")

    def process(self) -> tuple:
        records = []
        appended = []
        datafile = self.folders.raw / self.folders.datafile

        tsreg = self.regexes.timestamp
        clearreg = self.regexes.clear
        authorreg = self.regexes.author
        fmt = self.regexes.fmt

        with datafile.open(encoding="utf-8") as f:
            for line in f.readlines():
                ts = re.search(tsreg, line)
                if ts:
                    timestamp = datetime.strptime(ts.group(0), fmt)
                    msg = re.sub(clearreg, "", line)
                    author = re.search(authorreg, line)
                    if author:
                        name = author.group(0)
                    else:
                        name = "Unknown"
                    records.append((timestamp, name, msg))
                else:
                    appended.append(timestamp)
                    msg = msg + re.sub(clearreg, "", line)
                    records[-1] = (timestamp, name, msg)

        logger.info(f"Found {len(records)} records")
        logger.info(f"Appended {len(appended)} records")
        logger.debug(f"appended: {appended}")
        return records, appended


@click.command()
@click.option("--device", default="android", help="Device type: iOS or Android")
def main(device: str):
    if device.lower() == "ios":
        logger.info("Using iOS regexes")
        regexes: BaseRegexes = iosRegexes
    elif device.lower() == "old":
        logger.info("Using old version regexes")
        regexes: BaseRegexes = oldRegexes  # type: ignore
    else:
        logger.info("Using Android regexes")
        regexes: BaseRegexes = androidRegexes  # type: ignore

    folders = Folders(
        raw=Path("data/raw"),
        processed=Path("data/processed"),
        datafile=Path("_chat.txt"),
    )
    preprocessor = WhatsappPreprocessor(folders, regexes)
    preprocessor()


if __name__ == "__main__":
    main()
