import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("snr_st_mix")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(output_dir / "train.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
