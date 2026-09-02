"""
config.py - Single Source of Truth for Molecular Representation Benchmark Suite

All constants, paths, colors, model configurations, and hyperparameters are defined here.
Import from this file in all other scripts. NEVER hardcode these values elsewhere.
"""

from pathlib import Path
from collections import OrderedDict

# ============================================================================
# Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent  # benchmark/ directory
DATA_DIR = Path("data/benchmark_data_small")
RESULTS_DIR = PROJECT_ROOT / "results_small"
FIGURES_DIR = PROJECT_ROOT / "figures"


# Ensure directories exist
PATHS = {
    "data_dir": DATA_DIR,
    "results_dir": RESULTS_DIR,
    "figures_dir": FIGURES_DIR,
}

for directory in PATHS.values():
    Path(directory).mkdir(parents=True, exist_ok=True)

# ============================================================================
# Representations
# ============================================================================

REPRESENTATIONS = [
    "canonical_smiles",
    "isomeric_smiles",
    "randomized_smiles",
    "deepsmiles",
    "iupac",
    "selfies",
    "moljson",
    "cml",
    "inchi",
]

REPR_DISPLAY_NAMES = {
    "canonical_smiles": "Canonical SMILES",
    "isomeric_smiles": "Isomeric SMILES",
    "randomized_smiles": "Randomized SMILES",
    "deepsmiles": "DeepSMILES",
    "iupac": "IUPAC",
    "selfies": "SELFIES",
    "moljson": "MolJSON",
    "cml": "CML",
    "inchi": "InChI",
}

REPR_COLORS = {
    "canonical_smiles": "#1f77b4",  # blue
    "isomeric_smiles": "#ff7f0e",   # orange
    "randomized_smiles": "#2ca02c", # green
    "deepsmiles": "#d62728",        # red
    "iupac": "#9467bd",             # purple
    "selfies": "#e377c2",           # pink
    "moljson": "#8c564b",           # brown
    "cml": "#17becf",               # teal
    "inchi": "#7f7f7f",             # gray
}

# ============================================================================
# Benchmark-Specific Token Budgets
# ============================================================================
# Different benchmarks require different token budgets based on task complexity
# Each benchmark has separate budgets for thinking ON and thinking OFF modes

BENCHMARK_TOKEN_BUDGETS = {
    "atom_counting": {
        "thinking_on": {
            "max_tokens": 512,          # Total output budget (thinking + answer)
            "thinking_max_tokens": 256,  # Budget for <think>...</think> reasoning
        },
        "thinking_off": {
            "max_tokens": 100,            # Simple answer only (e.g., "5")
        },
    },
    "functional_groups": {
        "thinking_on": {
            "max_tokens": 2048,           # Total: 256 thinking + 512 for answer (room for verbose output + final answer)
            "thinking_max_tokens": 512,  # Budget for <think>...</think> reasoning
        },
        "thinking_off": {
            "max_tokens": 50,             # Yes/No answer
        },
    },
    "property_estimation": {
        "thinking_on": {
            "max_tokens": 2048,           # More complex reasoning for property prediction
            "thinking_max_tokens": 512,
        },
        "thinking_off": {
            "max_tokens": 200,            # Numeric answer with brief explanation
        },
    },
    "retrieval": {
        "thinking_on": {
            "max_tokens": 2048,
            "thinking_max_tokens": 512,
        },
        "thinking_off": {
            "max_tokens": 100,            # Index or identifier
        },
    },
    "isomer_discrimination": {
        "thinking_on": {
            "max_tokens": 2048,           # Complex reasoning about structural differences
            "thinking_max_tokens": 512,
        },
        "thinking_off": {
            "max_tokens": 100,            # Yes/No or choice
        },
    },
    "tautomer_recognition": {
        "thinking_on": {
            "max_tokens": 2048,           # Complex reasoning about tautomeric equivalence
            "thinking_max_tokens": 512,
        },
        "thinking_off": {
            "max_tokens": 100,            # Yes/No answer
        },
    },
    "protonation_recognition": {
        "thinking_on": {
            "max_tokens": 2048,           # Complex reasoning about protonation state equivalence
            "thinking_max_tokens": 512,
        },
        "thinking_off": {
            "max_tokens": 100,            # Yes/No answer
        },
    },
    "generation": {
        "thinking_on": {
            "max_tokens": 2048,           # Need space for reasoning + MolJSON (up to 3074 tokens)
            "thinking_max_tokens": 512,  # Leaves ~4096 tokens for answer
        },
        "thinking_off": {
            "max_tokens": 1024,           # MolJSON can be up to 3074 tokens (95th percentile: 1362)
        },
    },
    "completion": {
        "thinking_on": {
            "max_tokens": 2048,           # Reasoning + MolJSON completion (partial molecules can be large)
            "thinking_max_tokens": 512,  # Leaves ~4096 tokens for answer
        },
        "thinking_off": {
            "max_tokens": 1024,           # Completed MolJSON output
        },
    },
}

# ============================================================================
# Models
# ============================================================================

MODELS = [
    { ### kdeng03
        "name": "Qwen3-4B-Instruct-2507",
        "id": "qwen3-4b-instruct-2507",
        "hf_id": "Qwen/Qwen3-4B-Instruct-2507",
        "reasoning_parser": None,
        "thinking_on_config": {
            "chat_template_kwargs": {}
        },
        "thinking_off_config": {
            "chat_template_kwargs": {"enable_thinking": False}
        },
        "generation_params": {
            "temperature": 0.7,  # Qwen3-4B-Instruct best practice
            "top_p": 0.8,      # Qwen3-4B-Instruct best practice
            "top_k": 20,        # Qwen3-4B-Instruct best practice
            "max_tokens": 256,
        },
        "skip_stop_on_thinking_off": True,  # Reasons in plain text with thinking off
        "marker": "^",  # circle
    },
    { ### kdeng03
        "name": "MolQwen3-4B-Instruct-SFT",
        "id": "molqwen3-4b-instruct-sft",
        "hf_id": "kdeng03/MolQwen3-4B-Instruct-SFT",
        "reasoning_parser": None,
        "thinking_on_config": {
            "chat_template_kwargs": {}
        },
        "thinking_off_config": {
            "chat_template_kwargs": {"enable_thinking": False}
        },
        "generation_params": {
            "temperature": 0.7,  # Qwen3-4B-Instruct best practice
            "top_p": 0.8,      # Qwen3-4B-Instruct best practice
            "top_k": 20,        # Qwen3-4B-Instruct best practice
            "max_tokens": 256,
        },
        "skip_stop_on_thinking_off": True,  # Reasons in plain text with thinking off
        "marker": "A",  # circle
    },
    { ### kdeng03
        "name": "Qwen3-VL-4B-Instruct",
        "id": "qwen3-vl-4b-instruct",
        "hf_id": "Qwen/Qwen3-VL-4B-Instruct",
        "reasoning_parser": None,
        "thinking_on_config": {
            "chat_template_kwargs": {}
        },
        "thinking_off_config": {
            "chat_template_kwargs": {"enable_thinking": False}
        },
        "generation_params": {
            "temperature": 1.,  # Qwen3-4B-Instruct best practice
            "top_p": 1.,      # Qwen3-4B-Instruct best practice
            "top_k": 40,        # Qwen3-4B-Instruct best practice
            "max_tokens": 256,
        },
        "skip_stop_on_thinking_off": True,  # Reasons in plain text with thinking off
        "marker": "<",  # circle filled
    },
    { ### kdeng03
        "name": "MolQwen3-VL-4B-Instruct-SFT",
        "id": "molqwen3-vl-4b-instruct-sft",
        "hf_id": "kdeng03/MolQwen3-VL-4B-Instruct-SFT",
        "reasoning_parser": None,
        "thinking_on_config": {
            "chat_template_kwargs": {}
        },
        "thinking_off_config": {
            "chat_template_kwargs": {"enable_thinking": False}
        },
        "generation_params": {
            "temperature": 1.,  # Qwen3-4B-Instruct best practice
            "top_p": 1.,      # Qwen3-4B-Instruct best practice
            "top_k": 40,        # Qwen3-4B-Instruct best practice
            "max_tokens": 256,
        },
        "skip_stop_on_thinking_off": True,  # Reasons in plain text with thinking off
        "marker": ">",  # circle filled
    },
    {
        "name": "Qwen3-4B-Thinking-2507",
        "id": "qwen3-4b-thinking-2507",
        "hf_id": "/node2/arunraja/pretrained_llms/Qwen3-4B-Thinking-2507",
        "reasoning_parser": "qwen3",
        "thinking_on_config": {
            "chat_template_kwargs": {"enable_thinking": True}
        },
        "thinking_off_config": {
            "chat_template_kwargs": {"enable_thinking": False}
        },
        "generation_params": {
            "temperature": 0.6,  # Qwen3-4B-Thinking best practice
            "top_p": 0.95,      # Qwen3-4B-Thinking best practice
            "top_k": 20,        # Qwen3-4B-Thinking best practice
            "max_tokens": 32768,
        },
        "skip_stop_on_thinking_off": True,  # Reasons in plain text with thinking off
        "marker": "p",  # plus
    },
    {
        "name": "Qwen3-30B-A3B-Thinking-2507",
        "id": "qwen3-30b-a3b-thinking-2507",
        "hf_id": "/node2/arunraja/pretrained_llms/Qwen3-30B-A3B-Thinking-2507",
        "reasoning_parser": "qwen3",
        "thinking_on_config": {
            "chat_template_kwargs": {"enable_thinking": True}
        },
        "thinking_off_config": {
            "chat_template_kwargs": {"enable_thinking": False}
        },
        "generation_params": {
            "temperature": 0.6,  # Qwen3-30B-A3B-Thinking-2507 best practice
            "top_p": 0.95,      # Qwen3-30B-A3B-Thinking-2507 best practice
            "top_k": 20,        # Qwen3-30B-A3B-Thinking-2507 best practice
            "max_tokens": 32768,
        },
        "skip_stop_on_thinking_off": True,  # Reasons in plain text with thinking off
        "marker": "P",  # plus filled
    },
    {
        "name": "Phi-4",
        "id": "phi-4",
        "hf_id": "/node2/arunraja/pretrained_llms/phi-4",
        "max_model_len": 16384,
        "reasoning_parser": None,
        "thinking_off_config": {
            "chat_template_kwargs": {}
        },
        "generation_params": {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": 16384,
        },
        "marker": "s",  # square
    },
    {
        "name": "Phi-4-Reasoning",
        "id": "phi-4-reasoning",
        "hf_id": "/node2/arunraja/pretrained_llms/Phi-4-reasoning",
        "reasoning_parser": "deepseek_r1",
        "thinking_on_config": {
            "chat_template_kwargs": {}
        },
        "generation_params": {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": 32768,
        },
        "marker": "S",  # square filled
    },
    {
        "name": "Phi-4-Reasoning-Plus",
        "id": "phi-4-reasoning-plus",
        "hf_id": "/node2/arunraja/pretrained_llms/Phi-4-reasoning-plus",
        "reasoning_parser": "deepseek_r1",
        "thinking_on_config": {
            "chat_template_kwargs": {}
        },
        "generation_params": {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": 32768,
        },
        "marker": "◇",  # diamond open
    },
    {
        "name": "OLMo-3.1-32B-Instruct",
        "id": "olmo-3.1-32b-instruct",
        "hf_id": "/node2/arunraja/pretrained_llms/Olmo-3.1-32B-Instruct",
        "skip_stop_on_thinking_off": True,  # Reasons in plain text with thinking off
        "generation_params": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": 32768,
        },
        "marker": "v",  # triangle down
    },
    {
        "name": "OLMo-3.1-32B-Think",
        "id": "olmo-3.1-32b-think",
        "hf_id": "/node2/arunraja/pretrained_llms/Olmo-3.1-32B-Think",
        "reasoning_parser": "olmo3",
        "thinking_on_config": {
            "chat_template_kwargs": {}
        },
        "generation_params": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": 32768,
        },
        "marker": "D",  # diamond (matplotlib)
    },
    {
        "name": "Qwen3-235B-A22B-Thinking-2507",
        "id": "qwen3-235b-a22b-thinking-2507",
        "hf_id": "/node2/arunraja/pretrained_llms/Qwen3-235B-A22B-Thinking-2507/",
        "reasoning_parser": "deepseek_r1",  # Uses DeepSeek R1 parser instead of qwen3
        "thinking_on_config": {
            "chat_template_kwargs": {"enable_thinking": True}
        },
        "thinking_off_config": {
            "chat_template_kwargs": {"enable_thinking": False}
        },
        "generation_params": {
            "temperature": 0.6,  # Qwen3-235B-A22B-Thinking-2507 best practice
            "top_p": 0.95,      # Qwen3-235B-A22B-Thinking-2507 best practice
            "top_k": 20,        # Qwen3-235B-A22B-Thinking-2507 best practice
            "max_tokens": 32768,
        },
        "marker": "◆",  # diamond (largest thinking model)
    },
    {
        "name": "ChemDFM-v2.0-14B",
        "id": "chemdfm-v2.0-14b",
        "hf_id": "/node2/arunraja/pretrained_llms/ChemDFM-v2.0-14B",
        "max_model_len": 16384,
        "reasoning_parser": None,
        "skip_stop_on_thinking_off": True,  # Reasons in plain text with thinking off
        "thinking_off_config": {
            "chat_template_kwargs": {}
        },
        "generation_params": {
            "temperature": 0.9,
            "top_p": 0.9,
            "top_k": 20,
            "max_tokens": 8192,
        },
        "marker": "h",  # hexagon
    },
    {
        "name": "ChemDFM-R-14B",
        "id": "chemdfm-r-14b",
        "hf_id": "/node2/arunraja/pretrained_llms/ChemDFM-R-14B",
        "max_model_len": 16384,
        "reasoning_parser": None,  # plain-text <think> tags, no vLLM parser token support
        "thinking_on_config": {
            "system_message": (
                "You are a helpful assistant that is good at reasoning. "
                "You always reason thoroughly before giving response. "
                "The reasoning process and answer are enclosed within "
                "<think> </think> and <answer> </answer> tags, respectively."
            ),
            "chat_template_kwargs": {}
        },
        "generation_params": {
            "temperature": 0.9,
            "top_p": 0.9,
            "top_k": 20,
            "max_tokens": 8192,
        },
        "marker": "H",  # hexagon filled
    },
    {
        "name": "Ether0-24B",
        "id": "ether0-24b",
        "hf_id": "/node2/arunraja/pretrained_llms/ether0",
        "reasoning_parser": None,  # plain-text <think> tags, no vLLM parser
        "thinking_on_config": {
            "chat_template_kwargs": {}
        },
        "generation_params": {
            "temperature": 0.15,
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": 32768,
        },
        "marker": "★",  # star
    },
    {
        "name": "Mistral-Small-24B-Instruct-2501",
        "id": "mistral-small-24b",
        "hf_id": "/node2/arunraja/pretrained_llms/Mistral-Small-24B-Instruct-2501",
        "reasoning_parser": None,
        "thinking_on_config": {
            "chat_template_kwargs": {}
        },
        "generation_params": {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": 32768,
        },
        "marker": "d",  # thin diamond
    },
    {
        "name": "Qwen2.5-14B",
        "id": "qwen2.5-14b",
        "hf_id": "/node2/arunraja/pretrained_llms/Qwen2.5-14B",
        "reasoning_parser": None,
        "thinking_on_config": {
            "chat_template_kwargs": {}
        },
        "generation_params": {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 32768,
        },
        "marker": "8",  # octagon
    },
    {
        "name": "Qwen3-4B-Thinking-2507-MolJSON",
        "id": "qwen3-4b-thinking-2507-moljson",
        "hf_id": "/node2/arunraja/pretrained_llms/Qwen3-4B-Thinking-2507-MolJSON-Merged",
        "reasoning_parser": "qwen3",
        "thinking_on_config": {
            "chat_template_kwargs": {"enable_thinking": True}
        },
        "thinking_off_config": {
            "chat_template_kwargs": {"enable_thinking": False}
        },
        "generation_params": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 32768,
        },
        "skip_stop_on_thinking_off": True,
        "marker": "x",  # cross — fine-tuned variant
    },
    {
        "name": "Qwen3-4B-Thinking-2507-MolJSON-v2",
        "id": "qwen3-4b-thinking-2507-moljson-v2",
        "hf_id": "/node2/arunraja/pretrained_llms/Qwen3-4B-Thinking-2507-MolJSON-Merged-v2",
        "reasoning_parser": "qwen3",
        "thinking_on_config": {
            "chat_template_kwargs": {"enable_thinking": True}
        },
        "thinking_off_config": {
            "chat_template_kwargs": {"enable_thinking": False}
        },
        "generation_params": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 32768,
        },
        "skip_stop_on_thinking_off": True,
        "marker": "+",  # plus — fine-tuned v2 (thinking, r=16, 1ep)
    },
]

# Model ID lookup
MODEL_IDS = [m["id"] for m in MODELS]
MODEL_NAMES = {m["id"]: m["name"] for m in MODELS}

# ============================================================================
# Benchmarks
# ============================================================================

BENCHMARKS = [
    "atom_counting",
    "functional_groups",
    "property_estimation",
    "retrieval",
    "isomer_discrimination",
    "generation",
    "completion",
    "tautomer_recognition",
    "protonation_recognition",
]

BENCHMARK_DISPLAY_NAMES = {
    "atom_counting": "B1: Atom Counting",
    "functional_groups": "B2: Functional Groups",
    "property_estimation": "B3: Property Estimation",
    "retrieval": "B4: Molecule Retrieval",
    "isomer_discrimination": "B5: Isomer Discrimination",
    "generation": "B6: Caption-to-Molecule",
    "completion": "B7: Molecular Completion",
    "tautomer_recognition": "B9: Tautomer Recognition",
    "protonation_recognition": "B10: Protonation State Recognition",
}

# Benchmark number to name mapping (1-indexed)
BENCHMARK_NUM_TO_NAME = {
    1: "atom_counting",
    2: "functional_groups",
    3: "property_estimation",
    4: "retrieval",
    5: "isomer_discrimination",
    6: "generation",
    7: "completion",
    9: "tautomer_recognition",
    10: "protonation_recognition",
}

BENCHMARK_NAME_TO_NUM = {v: k for k, v in BENCHMARK_NUM_TO_NAME.items()}

# ============================================================================
# Functional Groups (for Benchmark 2)
# ============================================================================

# 5 groups selected for maximal discrimination between representations:
# rare/specific enough that trivial guessing fails, and string-level detectability
# varies significantly across SMILES, IUPAC, SELFIES, etc.
# Dropped: hydroxyl (>70% prevalence), aromatic (>60%), ketone (common),
#          carboxylic_acid (easily spotted in IUPAC), amide (too common in drugs).
FUNCTIONAL_GROUPS = OrderedDict([
    ("primary_amine", {
        "name": "Primary amine",
        "smarts": "[NX3;H2;!$(NC=O)]",
    }),
    ("ester", {
        "name": "Ester",
        "smarts": "[#6][CX3](=O)[OX2H0][#6]",
    }),
    ("aldehyde", {
        "name": "Aldehyde",
        "smarts": "[CX3H1](=O)[#6]",
    }),
    ("sulfonamide", {
        "name": "Sulfonamide",
        "smarts": "[SX4](=[OX1])(=[OX1])([NX3])",
    }),
    ("halide", {
        "name": "Halide",
        "smarts": "[CX4][F,Cl,Br,I]",
    }),
])

# ============================================================================
# Elements (for Benchmark 1)
# ============================================================================

ELEMENTS = ["C", "N", "O", "S", "F", "Cl"]

# ============================================================================
# Properties (for Benchmark 3)
# ============================================================================

PROPERTIES = OrderedDict([
    ("logp", {
        "name": "LogP (partition coefficient)",
        "display": "LogP",
    }),
    ("tpsa", {
        "name": "Topological Polar Surface Area",
        "display": "TPSA",
    }),
    ("hbd", {
        "name": "Number of H-bond donors",
        "display": "H-bond Donors",
    }),
    ("hba", {
        "name": "Number of H-bond acceptors",
        "display": "H-bond Acceptors",
    }),
])

# ============================================================================
# Generation Parameters
# ============================================================================
# NOTE: Generation parameters are now model-specific and defined in each model's
# "generation_params" field above. No global defaults exist - each model must
# have its parameters explicitly configured.

# Stop sequences for different benchmark types
STOP_SEQUENCES = {
    "comprehension": ["\n\n"],
    "generation": ["\nDescription:", "\n\n"],
    "completion": ["\nPartial:", "\n\n"],
    "retrieval": ["\nA:", "\nB:", "\nC:", "\nD:", "\n\n"],
}

# ============================================================================
# Sampling Configuration
# ============================================================================

# Sample size for comprehension benchmarks (1-5)
COMPREHENSION_SAMPLE_SIZE = 200

# Cap for generation/completion benchmarks (6-7); None = full test set
GENERATION_SAMPLE_SIZE = 250

# Complexity floor for comprehension subset — molecules below these thresholds
# are excluded before stratified sampling to avoid trivially easy cases
COMPLEXITY_FLOOR_MW = 300       # minimum molecular weight
COMPLEXITY_FLOOR_RINGS = 2      # minimum ring count

# Stratification bins for comprehension sampling
MOLECULAR_WEIGHT_BINS = [
    (0, 300, "small"),
    (300, 500, "medium"),
    (500, float("inf"), "large"),
]

RING_COUNT_BINS = [
    (0, 0, "acyclic"),
    (1, 1, "monocyclic"),
    (2, 2, "bicyclic"),
    (3, float("inf"), "polycyclic"),
]

# Isomer discrimination pairs (Benchmark 5)
ISOMER_PAIRS_POSITIVE = 250
ISOMER_PAIRS_NEGATIVE = 250
ISOMER_PAIRS_STEREOISOMER = 125  # of the negative pairs
ISOMER_PAIRS_SUBSTITUTION = 125  # of the negative pairs

# Few-shot learning (Benchmark 6)
N_SHOT = 2
FEW_SHOT_METHOD = "tfidf"  # TF-IDF cosine similarity

# ============================================================================
# Figure Settings
# ============================================================================

# Seaborn style
PLOT_STYLE = "whitegrid"
FONT_SCALE = 1.2

# DPI for PNG outputs
PLOT_DPI = 300

# Figure sizes (width, height) in inches
FIGURE_SIZES = {
    "fig1_main_heatmap": (7, 4),
    "fig2_generation_barplot": (7, 3.5),
    "fig3_comprehension_vs_generation_scatter": (5, 5),
    "fig4_thinking_ablation_delta": (7, 10),
    "fig5_representation_radar": (5, 5),
    "fig6_atom_counting_by_size": (5, 3.5),
    "fig7_functional_group_breakdown": (7, 4),
    "fig8_property_estimation_scatter": (7, 3),
    "fig9_validity_by_complexity": (7, 3.5),
    "fig10_generation_tanimoto_violin": (7, 4),
    "fig11_thinking_vs_benchmark_interaction": (5, 3.5),
    "fig12_isomer_discrimination_breakdown": (5, 3.5),
    "fig13_completion_validity_vs_recovery": (5, 5),
    "fig14_retrieval_distractor_confusion": (5, 3.5),
    "fig15_statistical_significance_matrix": (7, 3.5),
    "fig16_token_length_vs_performance": (5, 5),
    "fig_summary_table": (7, 3),
}

# Colormap for heatmaps
HEATMAP_CMAP = "YlOrRd"

# Line styles for thinking conditions
THINKING_STYLES = {
    True: {"linestyle": "-", "markerfacecolor": "full"},   # solid, filled
    False: {"linestyle": "--", "markerfacecolor": "none"}, # dashed, open
}

# ============================================================================
# Statistical Testing
# ============================================================================

ALPHA = 0.05

# Bonferroni correction: 9 representations → C(9,2) = 36 pairwise comparisons
N_COMPARISONS = 36
BONFERRONI_ALPHA = ALPHA / N_COMPARISONS

# Bootstrap parameters
BOOTSTRAP_N_RESAMPLES = 1000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

# ============================================================================
# Molecular Complexity Bins
# ============================================================================

# For analysis breakdown by molecule size
HEAVY_ATOM_BINS = [
    (0, 10, "5-10"),
    (10, 15, "11-15"),
    (15, 20, "16-20"),
    (20, 25, "21-25"),
    (25, 30, "26-30"),
    (30, 35, "31-35"),
    (35, float("inf"), "35+"),
]

# ============================================================================
# Random Seed
# ============================================================================

SEED = 42

# ============================================================================
# CSV Column Names (for consistency across scripts)
# ============================================================================

# Standard columns present in all benchmark CSVs
STANDARD_COLUMNS = ["molecule_id", "representation", "model", "thinking"]

# Additional columns per benchmark
BENCHMARK_COLUMNS = {
    "atom_counting": ["element", "raw_response", "parsed_answer", "predicted", "ground_truth", "correct"],
    "functional_groups": ["group", "raw_response", "parsed_answer", "predicted", "ground_truth", "correct"],
    "property_estimation": ["property", "raw_response", "parsed_answer", "predicted", "ground_truth", "error"],
    "retrieval": ["raw_response", "parsed_answer", "predicted_letter", "correct_letter", "distractor_type", "correct"],
    "isomer_discrimination": ["pair_type", "raw_response", "parsed_answer", "predicted", "ground_truth", "correct"],
    "generation": ["description", "raw_response", "parsed_answer", "generated_string", "valid", "exact_match", "tanimoto_morgan", "tanimoto_maccs"],
    "completion": ["partial_input", "raw_response", "parsed_answer", "generated_string", "valid", "recovery", "tanimoto"],
    "tautomer_recognition": ["pair_type", "tautomer_class", "raw_response", "parsed_answer", "predicted", "ground_truth", "correct"],
    "protonation_recognition": ["pair_type", "ionizable_group", "charge_1", "charge_2", "raw_response", "parsed_answer", "predicted", "ground_truth", "correct"],
}

# ============================================================================
# Result File Naming
# ============================================================================

def get_result_filename(benchmark_name: str, suffix: str = "") -> Path:
    """
    Get the path for a benchmark result CSV file.

    Args:
        benchmark_name: Name of benchmark (e.g., "atom_counting")
        suffix: Optional suffix (e.g., "_scored", "_raw")

    Returns:
        Path to result file
    """
    benchmark_num = BENCHMARK_NAME_TO_NUM.get(benchmark_name, 0)
    filename = f"benchmark_{benchmark_num}_{benchmark_name}{suffix}.csv"
    return RESULTS_DIR / filename

def get_figure_path(figure_name: str, ext: str = "pdf") -> Path:
    """
    Get the path for a figure file.

    Args:
        figure_name: Name of figure (e.g., "fig1_main_heatmap")
        ext: File extension ("pdf" or "png")

    Returns:
        Path to figure file
    """
    return FIGURES_DIR / f"{figure_name}.{ext}"

def get_checkpoint_path(benchmark_name: str, model_id: str, representation: str, thinking: bool) -> Path:
    """
    Get the path for a checkpoint file.

    Args:
        benchmark_name: Name of benchmark
        model_id: Model identifier
        representation: Representation name
        thinking: Whether thinking is enabled

    Returns:
        Path to checkpoint file
    """
    checkpoint_dir = RESULTS_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    thinking_str = "thinking_on" if thinking else "thinking_off"
    filename = f"checkpoint_{benchmark_name}_{model_id}_{representation}_{thinking_str}.csv"
    return checkpoint_dir / filename

# ============================================================================
# Validation
# ============================================================================

def validate_config():
    """Validate that configuration is consistent."""
    assert len(REPRESENTATIONS) == 9, "Must have exactly 9 representations"
    assert len(BENCHMARKS) == 9, "Must have exactly 9 benchmarks"
    assert len(MODELS) == 15, "Must have exactly 15 models"
    assert len(FUNCTIONAL_GROUPS) == 5, "Must have exactly 5 functional groups"
    assert len(ELEMENTS) == 6, "Must have exactly 6 elements"

    # Check all representations have display names and colors
    for rep in REPRESENTATIONS:
        assert rep in REPR_DISPLAY_NAMES, f"Missing display name for {rep}"
        assert rep in REPR_COLORS, f"Missing color for {rep}"

    # Check all benchmarks have display names
    for bench in BENCHMARKS:
        assert bench in BENCHMARK_DISPLAY_NAMES, f"Missing display name for {bench}"

    # Check all models have required fields
    for model in MODELS:
        required = ["name", "id", "hf_id", "marker"]
        for field in required:
            assert field in model, f"Model missing field: {field}"

    print("✓ Configuration validated successfully")

if __name__ == "__main__":
    validate_config()
    print(f"\nConfiguration Summary:")
    print(f"  Representations: {len(REPRESENTATIONS)}")
    print(f"  Models: {len(MODELS)}")
    print(f"  Benchmarks: {len(BENCHMARKS)}")
    print(f"  Total conditions: {len(REPRESENTATIONS)} × {len(MODELS)} × 2 (thinking on/off) = {len(REPRESENTATIONS) * len(MODELS) * 2}")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Results directory: {RESULTS_DIR}")
    print(f"  Figures directory: {FIGURES_DIR}")
