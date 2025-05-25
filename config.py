# config.py   : Project directory paths and default settings.
#
# Author       : Casper Bröcheler <casper.jxb@gmail.com>
# GitHub       : https://github.com/casperbroch
# Affiliation  : Maastricht University


from pathlib import Path

ROOT          = Path(__file__).resolve().parent   # Base project directory
DATA_DIR      = ROOT / "data"                     # Root for data storage
MODELS_DIR    = DATA_DIR / "models"               # Saved model directory
LOGS_DIR      = DATA_DIR / "logs"                 # Training logs directory
VIDEOS_DIR    = DATA_DIR / "videos"               # Generated videos directory

DEFAULT_DEVICE = "cpu"                            # Fallback compute device