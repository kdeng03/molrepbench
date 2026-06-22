"""
GPT 5.5 configuration for the Molecular Representation Benchmark Suite.

Thin wrapper: imports shared constants from scripts/config.py, defines
GPT-specific model config and results directory.
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
# GPT 5.5 Model Configuration
# ============================================================================

MODEL = {
    "name": "GPT-5.4-mini",
    "id": "gpt-5.4-mini",
    "api_model": "gpt-5.4-mini",
    "api_base": "https://api.openai.com/v1",
    "thinking_on_config": {
        "reasoning_effort": "medium",
    },
    "generation_params": {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 128000,
    },
}

# ============================================================================
# Results Directory
# ============================================================================

RESULTS_DIR = PROJECT_ROOT / "results_gpt"

# ============================================================================
# Skip B7 (completion)
# ============================================================================

SKIP_BENCHMARKS = {7}

# ============================================================================
# API Settings
# ============================================================================

MAX_CONCURRENT_REQUESTS = 5
MAX_RETRIES = 5
REQUEST_TIMEOUT = 300  # seconds — reasoning models can be slow
