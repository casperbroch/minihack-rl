from pathlib import Path
ROOT          = Path(__file__).resolve().parent
DATA_DIR      = ROOT / "data"
MODELS_DIR    = DATA_DIR / "models"
LOGS_DIR      = DATA_DIR / "logs"
VIDEOS_DIR    = DATA_DIR / "videos"

DEFAULT_DEVICE = "cpu"      
