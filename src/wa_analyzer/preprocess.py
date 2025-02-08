import re
import sys
import tomllib
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
from loguru import logger

from wa_analyzer.settings import (BaseRegexes, Folders, androidRegexes,
                                  iosRegexes, oldRegexes)

logger.remove()
logger.add("logs/logfile.log", rotation="1 week", level="DEBUG")
logger.add(sys.stderr, level="INFO")

logger.debug(f"Python path: {sys.path}")


class WhatsappPreprocessor:
    def __init__(self, folders: Folders, regexes: BaseRegexes, datetime_format: str):
        self.folders = folders
        self.regexes = regexes
        self.datetime_format = datetime_format

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
        messagereg = self.regexes.message
        authorreg = self.regexes.author

        with datafile.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f.readlines()):
                ts = re.match(tsreg, line)
                if ts:
                    try:
                        timestamp = datetime.strptime(ts.groups()[0], self.datetime_format)
                    except ValueError as e:
                        logger.error(f"Error while processing timestamp of line {line_number}: {e}")
                        continue
                    msg = re.search(messagereg, line)
                    author = re.search(authorreg, line)
                    if msg is None:
                        logger.error(f"Could not find a message for line {line_number}. Please check the data and / or the message regex")
                        continue
                    if author is None:
                        logger.error(f"Could not find an author for line {line_number}. Please check the data and / or the author regex")
                        continue
                    author = author.groups()[0].strip()
                    msg = msg.groups()[0].strip()
                    records.append((timestamp, author, msg))
                elif len(records) > 0:
                    appended.append(timestamp)
                    msg += " " + line.strip()
                    records[-1] = (timestamp, author, msg)

        logger.info(f"Found {len(records)} valid records")
        logger.info(f"Appended {len(appended)} records")
        return records, appended


@click.command()
@click.option("--device", default="android", help="Device type: iOS or Android")
def main(device: str):
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
        raw = Path(config["raw"])
        processed = Path(config["processed"])
        datafile = Path(config["input"])
        datetime_format = config["datetime_format"]

    if device.lower() == "ios":
        logger.info("Using iOS regexes")
        regexes: BaseRegexes = iosRegexes
    elif device.lower() == "old":
        logger.info("Using old version regexes")
        regexes: BaseRegexes = oldRegexes  # type: ignore
    else:
        logger.info("Using Android regexes")
        regexes: BaseRegexes = androidRegexes  # type: ignore

    if not (raw / datafile).exists():
        logger.error(f"File {raw / datafile} not found")
    else:
        logger.info(f"Reading from {raw / datafile}")

    folders = Folders(
        raw=raw,
        processed=processed,
        datafile=datafile,
    )
    preprocessor = WhatsappPreprocessor(folders, regexes, datetime_format)
    preprocessor()


if __name__ == "__main__":
    main()
