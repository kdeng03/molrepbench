# MolRepBench: Molecular Representation Benchmark for LLMs

MolRepBench is a comprehensive benchmark suite for evaluating how well large language models understand and generate molecules across different molecular representations.

## Representations

The benchmark tests **9 molecular representations**:

| Representation | Example (Aspirin) |
|---|---|
| Canonical SMILES | `CC(=O)Oc1ccccc1C(=O)O` |
| Isomeric SMILES | `CC(=O)Oc1ccccc1C(=O)O` |
| Randomized SMILES | `O=C(O)c1ccccc1OC(C)=O` |
| DeepSMILES | `CC=O)Oc1ccccc1C=O)O` |
| IUPAC | `2-acetyloxybenzoic acid` |
| SELFIES | `[C][C][=Branch1][C][=O][O][C][=C]...` |
| MolJSON | `{"atoms": [...], "bonds": [...]}` |
| CML | `<molecule>...</molecule>` |
| InChI | `InChI=1S/C9H8O4/c1-6(10)...` |

## Benchmark Tasks

| # | Task | Type | What it tests |
|---|---|---|---|
| B1 | Atom Counting | Comprehension | Count occurrences of a specific element |
| B2 | Functional Group Identification | Comprehension | Detect presence of a functional group |
| B3 | Property Estimation | Comprehension | Predict molecular properties (LogP, TPSA, HBD, HBA) |
| B4 | Molecule Retrieval | Comprehension | Identify a molecule from 4 choices given a description |
| B5 | Isomer Discrimination | Comprehension | Determine whether two molecules are isomers |
| B6 | Caption-to-Molecule Generation | Generation | Generate a molecule from a text description |
| B7 | Molecular Completion | Generation | Complete a partially given molecule |
| B9 | Tautomer Recognition | Comprehension | Determine whether two molecules are tautomers |
| B10 | Protonation State Recognition | Comprehension | Determine whether two molecules differ only by protonation |

All tasks are evaluated with **thinking on** (chain-of-thought reasoning enabled) and **thinking off** conditions.

## Repository Structure

```
molrepbench/
├── scripts/
│   ├── config.py                 # All constants, model configs, paths, colors
│   ├── prepare_dataset.py        # Download & prepare ChEBI-20, convert to 9 representations
│   ├── run_inference.py          # Run benchmarks via vLLM (local) or API
│   ├── evaluate.py               # Score results, compute metrics (no GPU needed)
│   ├── plot_figures.py           # Generate all figures from scored CSVs
│   ├── statistical_tests.py      # Significance tests, p-value tables
│   ├── tabulate_results.py       # Build LaTeX/CSV result tables
│   ├── qualitative_eval.py       # LLM-as-judge evaluation for B6
│   └── utils/
│       ├── chemistry.py          # RDKit helper functions
│       ├── representations.py    # Molecule → representation converters
│       ├── prompts.py            # Prompt templates for all benchmarks
│       └── parsing.py            # Response parsing utilities
├── claude/                       # Claude API inference module
├── gemini/                       # Gemini API inference module
├── gpt/                          # GPT API inference module
├── moljson/                      # MolJSON schema & conversion utilities
├── requirements.txt
└── README.md
```

## Setup

### Requirements

- Python 3.10+
- RDKit
- CUDA-capable GPU (for local inference via vLLM)

### Installation

```bash
pip install -r requirements.txt
```

For IUPAC name conversion, you also need Open Babel:
```bash
conda install -c conda-forge openbabel
```

## Usage

### 1. Prepare the dataset

Downloads ChEBI-20 and converts molecules to all 9 representations:

```bash
python scripts/prepare_dataset.py
```

This creates benchmark-specific data files (stratified samples, distractor sets, isomer pairs, few-shot examples, etc.) under the configured `DATA_DIR`.

### 2. Run inference

Run benchmarks against a model served via vLLM:

```bash
# Start vLLM server (in a separate terminal)
vllm serve <model_path> --port 8000

# Run all benchmarks for a specific model
python scripts/run_inference.py --model qwen3-4b-thinking-2507 --thinking on

# Run a specific benchmark and representation
python scripts/run_inference.py --benchmark 1 --model qwen3-4b-thinking-2507 \
    --representation canonical_smiles --thinking on

# Dry run (print first prompt only)
python scripts/run_inference.py --benchmark 6 --model qwen3-4b-thinking-2507 --dry-run
```

> kdeng03
```bash
# Serve qwen3_4b_i
HF_ENDPOINT=https://hf-mirror.com \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
--gpu-memory-utilization 0.88 \
--max-model-len 16384 \
--port 8000

# Infer
## qwen3_4b_i
HF_ENDPOINT=https://hf-mirror.com \
python scripts/run_inference.py \
--model qwen3-4b-instruct-2507 \
--direct_vllm \
--batch_size 4096 \
--thinking off 2>&1 | tee logs/infer_qwen3_4b_i.log

## qwen3_vl_4b_i
# HF_ENDPOINT=https://hf-mirror.com \
# python scripts/run_inference.py \
# --model qwen3-vl-4b-instruct \
# --direct_vllm \
# --batch_size 4096 \
# --thinking off 2>&1 | tee logs/infer_qwen3_vl_4b_i.log

# No batch (50 default)
HF_ENDPOINT=https://hf-mirror.com \
python scripts/run_inference.py \
--model qwen3-vl-4b-instruct \
--direct_vllm \
--thinking off 2>&1 | tee logs/infer_qwen3_vl_4b_i.log

## molqwen3_4b_i_sft
# HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=$(pwd) \
# python scripts/run_inference.py \
# --model molqwen3-4b-instruct-sft \
# --direct_vllm \
# --batch_size 4096 \
# --thinking off 2>&1 | tee logs/infer_molqwen3_4b_i_sft.log

# No batch
HF_ENDPOINT=https://hf-mirror.com \
python scripts/run_inference.py \
--model molqwen3-4b-instruct-sft \
--direct_vllm \
--thinking off 2>&1 | tee logs/infer_molqwen3_4b_i_sft.log

## molqwen3_vl_4b_i_sft
HF_ENDPOINT=https://hf-mirror.com \
python scripts/run_inference.py \
--model molqwen3-vl-4b-instruct-sft \
--direct_vllm \
--thinking off 2>&1 | tee logs/infer_molqwen3_vl_4b_i_sft.log
```

Results are saved as JSONL files with per-row checkpointing for crash recovery.

For API-based models (Claude, GPT, Gemini), see the respective directories under `claude/`, `gpt/`, `gemini/`.

### 3. Evaluate

Score raw results and compute metrics (no GPU needed):

```bash
python scripts/evaluate.py --benchmark all 2>&1 | tee logs/eval.log
```

Produces scored CSVs and aggregated metrics under the results directory.

### 4. Generate figures

```bash
python scripts/plot_figures.py --all

# Or generate a specific figure
python scripts/plot_figures.py --figure fig1_main_heatmap
```

All figures are saved as both PDF and PNG (300 DPI).

### 5. Statistical tests

```bash
python scripts/statistical_tests.py --all
```

### 6. Result tables

```bash
python scripts/tabulate_results.py
```

## Configuration

All constants are centralized in [`scripts/config.py`](scripts/config.py):

- **Representations**: names, display names, color palette
- **Models**: HuggingFace IDs, generation parameters, reasoning parsers
- **Benchmarks**: task names, token budgets, sampling sizes
- **Figures**: sizes, styles, colormaps

## Models Tested

The benchmark has been evaluated on models including:

- **Qwen3** family (4B, 30B-A3B) with thinking variants
- **Phi-4** 
- **OLMo-3.1-32B** (Instruct and Think)
- **ChemDFM** v2.0 and ChemDFM-R (chemistry-specialized)
- **Ether0-24B** (chemistry-specialized)
- **Mistral-Small-24B**
- **Qwen2.5-14B**
- **GPT-5.4-mini**, **Claude-Haiku-4.5** (via API)

## Acknolwedgements

This work was done
during A.R.’s internship at DSO National Laboratories as part of the DSO-AISG LLM Incentive Award.We would like to thank DSO and AI Singapore for
the computational resources, which played a significant role in this research. We would also like
to thank Dr Hongtao Zhao, Dr Christian Tyrchan,
Dr Eva Nittinger, Prof. Charlotte M. Deane, and
Prof. Michael M. Bronstein for their advice in this
project.


## Citation
If you found our work useful, please help to cite our paper

https://arxiv.org/abs/2606.03057 

```
@misc{raja2026rethinkingmoleculartextrepresentations,
      title={Rethinking Molecular Text Representations for LLMs: An Empirical Study}, 
      author={Arun Raja and Garrett M. Morris and Kian Ming A. Chai},
      year={2026},
      eprint={2606.03057},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.03057}, 
}
```