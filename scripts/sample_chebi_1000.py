"""
sample_chebi_1000.py - Randomly sample 1000 molecules from ChEBI-20 with all 7 representations.

Combines train/val/test splits, samples 1000 molecules (seed=42),
converts to all 7 representations, and saves to:
  /node2/arunraja/benchmark_data/chebi_sample_1000.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem

from config import (
    CHEBI20_TRAIN, CHEBI20_VAL, CHEBI20_TEST,
    DATA_DIR, REPRESENTATIONS, SEED,
)
from utils.representations import convert_to_representation

np.random.seed(SEED)

OUTPUT_PATH = DATA_DIR / "chebi_1000_analysis_study.csv"
N_SAMPLE = 1000


def load_all_splits():
    dfs = []
    for path, split in [(CHEBI20_TRAIN, "train"), (CHEBI20_VAL, "val"), (CHEBI20_TEST, "test")]:
        df = pd.read_csv(path)
        df["split"] = split
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} total molecules across all splits")
    return combined


def convert_molecules(df):
    results = {rep: [] for rep in REPRESENTATIONS}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting representations"):
        smiles = row["SMILES"]

        for rep in REPRESENTATIONS:
            if rep == "iupac":
                val = row.get("iupacname", None)
                results[rep].append(val if pd.notna(val) else None)
            elif rep == "selfies":
                val = row.get("SELFIES", None)
                results[rep].append(val if pd.notna(val) else None)
            else:
                results[rep].append(convert_to_representation(smiles, rep))

    for rep in REPRESENTATIONS:
        df[rep] = results[rep]

    return df


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_splits()

    # Drop rows with invalid SMILES
    valid_mask = df["SMILES"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
    df = df[valid_mask].reset_index(drop=True)
    print(f"Valid molecules: {len(df)}")

    # Random sample
    sample_df = df.sample(n=N_SAMPLE, random_state=SEED).reset_index(drop=True)
    print(f"Sampled {len(sample_df)} molecules")

    # Convert to all 7 representations
    sample_df = convert_molecules(sample_df)

    # Report success rates
    print("\nConversion success rates:")
    for rep in REPRESENTATIONS:
        n_ok = sample_df[rep].notna().sum()
        print(f"  {rep:20s}: {n_ok}/{N_SAMPLE} ({100*n_ok/N_SAMPLE:.1f}%)")

    # Save
    sample_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(sample_df)} molecules to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
