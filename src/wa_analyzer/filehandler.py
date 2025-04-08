from pathlib import Path

import pandas as pd
from loguru import logger


class FileHandler:
    def __init__(self, config):
        self.config = config

    def load(self, filepath: Path) -> pd.DataFrame:
        """Load preprocessed WhatsApp data."""
        timecol = self.config.time_col
        data = pd.read_csv(filepath, parse_dates=[timecol])
        logger.success(f"Loade data from {filepath}")
        return data
