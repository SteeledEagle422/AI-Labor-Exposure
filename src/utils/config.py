"""
Shared config loader. Every fetch/build/analysis script does:

    from src.utils.config import load_config, PROJECT_ROOT
    cfg = load_config()

so that paths and parameters live in one place (config/config.yaml).
"""
from __future__ import annotations

from pathlib import Path
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
SECTOR_MAP_PATH = PROJECT_ROOT / "config" / "sector_naics_map.csv"

# Loads .env (BLS_API_KEY, etc.) into the environment if present; a real
# shell-exported var always takes precedence over the .env file.
load_dotenv(PROJECT_ROOT / ".env")

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


def exposure_primary_column(cfg: dict | None = None) -> str:
    """Map config.yaml's exposure.eloundou.primary_column (an Eloundou raw
    column like "dv_rating_beta") to the crosswalk output column name
    (build_exposure_crosswalk.py emits "exposure_eloundou_{alpha,beta,gamma}")
    so the analysis scripts' default exposure measure actually follows the
    config value instead of a hardcoded string."""
    cfg = cfg or load_config()
    raw_col = cfg["exposure"]["eloundou"]["primary_column"]
    suffix = raw_col.removeprefix("dv_rating_")
    return f"exposure_eloundou_{suffix}"
