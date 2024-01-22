import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.settings import Folders, Regexes

logger.remove()
logger.add("logs/logfile.log", rotation="1 MB", level="DEBUG")
logger.add(sys.stderr, level="INFO")


if __name__ == "__main__":
    folders = Folders(
        raw=Path("data/raw"),
        processed=Path("data/processed"),
        datafile=Path("_chat.txt"),
    )
    regexes = Regexes()

    records = []
    appended = []
    datafile = folders.raw / folders.datafile
    with datafile.open() as f:
        for line in f.readlines():
            ts = re.search(regexes.timestamp, line)
            if ts:
                timestamp = datetime.strptime(ts.group(0), "%d-%m-%Y %H:%M:%S")
                msg = re.sub(regexes.clear, "", line)
                author = re.search(regexes.author, line)
                if author:
                    name = author.group(0)
                else:
                    name = "Unknown"
                records.append((timestamp, name, msg))
            else:
                appended.append(timestamp)
                msg = msg + re.sub(regexes.clear, "", line)
                records[-1] = (timestamp, name, msg)

    logger.info(f"Found {len(records)} records")
    logger.info(f"Appended {len(appended)} records")
    logger.debug(f"appended: {appended}")

    df = pd.DataFrame(records, columns=["timestamp", "author", "message"])
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    outfile = folders.processed / f"whatsapp-{now}.csv"
    logger.info(f"Writing to {outfile}")
    df.to_csv(outfile, index=False)
    logger.success("Done!")
