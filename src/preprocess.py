from datetime import datetime
from pathlib import Path
from loguru import logger
import re
import sys
import pandas as pd
from src.settings import Regexes, Folders

logger.remove()
logger.add("logs/logfile.log", rotation="1 MB", level="DEBUG")
logger.add(sys.stderr, level="INFO")


if __name__ == "__main__":

    folders = Folders(
        raw=Path("data/raw"),
        processed=Path("data/processed"),
        datafile=Path("_chat.txt")
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
                    author = author.group(0)
                records.append((timestamp, author, msg))
            else:
                appended.append(timestamp)
                msg = msg + re.sub(regexes.clear, "", line)
                records[-1] = (timestamp, author, msg)

    logger.info(f"Found {len(records)} records")
    logger.info(f"Appended {len(appended)} records")
    logger.debug(f"appended: {appended}")

    df = pd.DataFrame(records, columns=["timestamp", "author", "message"])
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    outfile = folders.processed / f"whatsapp-{now}.csv"
    logger.info(f"Writing to {outfile}")
    df.to_csv(outfile, index=False)
    logger.success("Done!")









