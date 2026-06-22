"""
Gemini 3 Flash Preview configuration for the Molecular Representation Benchmark Suite.

Thin wrapper: imports shared constants from scripts/config.py, defines
Gemini-specific model config and results directory.
"""

import os
import sys
from pathlib import Path

# Add project root to path so we can import from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from config import (
    REPRESENTATIONS,
    BENCHMARKS,
    BENCHMARK_COLUMNS,
    BENCHMARK_NUM_TO_NAME,
    BENCHMARK_NAME_TO_NUM,
    BENCHMARK_TOKEN_BUDGETS,
    DATA_DIR,
    ELEMENTS,
    FUNCTIONAL_GROUPS,
    PROPERTIES,
    REPR_DISPLAY_NAMES,
    STOP_SEQUENCES,
    GENERATION_SAMPLE_SIZE,
    N_SHOT,
)

# ============================================================================
# Gemini 3 Flash Preview Model Configuration
# ============================================================================

MODEL = {
    "name": "Gemini-3-Flash-Preview",
    "id": "gemini-3-flash-preview",
    "api_model": "gemini-3-flash-preview",
    "api_base": "https://generativelanguage.googleapis.com/v1beta",
    "thinking_on_config": {
        "thinking_level": "medium",     # minimal, low, medium, high
        "include_thoughts": True,       # return thinking text in response
    },
    "generation_params": {
        "temperature": 1.0,             # Gemini default
        "top_p": 0.95,
        "max_tokens": 65536,            # Gemini 3 Flash output limit
    },
}

# ============================================================================
# Results Directory
# ============================================================================

RESULTS_DIR = PROJECT_ROOT / "results_gemini"

# ============================================================================
# Skip B7 (completion — not applicable to API models)
# ============================================================================

SKIP_BENCHMARKS = {7}

# ============================================================================
# API Settings
# ============================================================================

MAX_CONCURRENT_REQUESTS = 5
MAX_RETRIES = 5
REQUEST_TIMEOUT = 300  # seconds — reasoning models can be slow
