import os
import sys

from loguru import logger

logger.info("===== sys.path =====")
logger.info(f"Length of sys.path: {len(sys.path)}")
for i, p in enumerate(sys.path):
    logger.info(f"sys.path[{i}]: {p}")

logger.info("===== PYTHONPATH =====")
pythonpath = str(os.environ.get("PYTHONPATH"))

if not pythonpath:
    pythonpath = "None"
if not isinstance(pythonpath, str):
    logger.error("PYTHONPATH is not a string")

logger.info(f"Length of PYTHONPATH: {len(pythonpath.split(os.pathsep))}")

for i, p in enumerate(pythonpath.split(os.pathsep)):
    logger.info(f"PYTHONPATH[{i}]: {p}")

logger.info("===== PATH =====")
path = str(os.environ.get("PATH"))
if not path:
    path = "None"
if not isinstance(path, str):
    logger.error("PATH is not a string")

logger.info(f"Length of PATH: {len(path.split(os.pathsep))}")
for i, p in enumerate(path.split(os.pathsep)):
    logger.info(f"PATH[{i}]: {p}")
