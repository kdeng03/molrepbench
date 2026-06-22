"""
Claude Haiku 4.5 configuration for the Molecular Representation Benchmark Suite.

Thin wrapper: imports shared constants from scripts/config.py, defines
Claude-specific model config and results directory.
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
# Claude Haiku 4.5 Model Configuration
# ============================================================================

MODEL = {
    "name": "Claude-Haiku-4.5",
    "id": "claude-haiku-4.5",
    "api_model": "claude-haiku-4-5-20251001",
    "api_base": "https://api.anthropic.com",
    "thinking_on_config": {
        "budget_tokens": 8192,
    },
    "generation_params": {
        "temperature": 1.0,       # Must be 1.0 when extended thinking is enabled
        "max_tokens": 16384,      # Max output tokens (excluding thinking)
    },
}

# ============================================================================
# Results Directory
# ============================================================================

RESULTS_DIR = PROJECT_ROOT / "results_claude"

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
ANTHROPIC_API_VERSION = "2023-06-01"
