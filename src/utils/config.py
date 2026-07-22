"""
Shared config loader. Every fetch/build/analysis script does:

    from src.utils.config import load_config, PROJECT_ROOT
    cfg = load_config()

so that paths and parameters live in one place (config/config.yaml).
"""
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
SECTOR_MAP_PATH = PROJECT_ROOT / "config" / "sector_naics_map.csv"

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "output" / "tables"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "output" / "figures"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def ensure_dirs(*dirs) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
