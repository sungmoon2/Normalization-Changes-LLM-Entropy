"""Shared path configuration for all experiment scripts.

Set ENTROPY_DATA_DIR environment variable to override the default
PoT_Experiment data directory location. When unset, uses the default
sibling directory relative to this repository.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

POT_DIR = (
    Path(os.environ["ENTROPY_DATA_DIR"])
    if os.environ.get("ENTROPY_DATA_DIR")
    else PROJECT_ROOT / "PoT_Experiment_Entropy_Attention_Extraction_Experiment"
)
