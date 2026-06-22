#!/usr/bin/env python3
"""
create_retrieval_distractors.py - Generate distractor molecules for B4 from ChEBI

Uses the prepared ChEBI dataset to create 3 distractors per molecule:
1. Same scaffold (Murcko scaffold match)
2. Similar molecular weight (±10%)
3. Random
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

from config import DATA_DIR, SEED

np.random.seed(SEED)

print("=" * 80)
print("Creating Retrieval Distractors for B4 (ChEBI)")
print("=" * 80)

# Load prepared ChEBI dataset
chebi_df = pd.read_csv(DATA_DIR / "chebi_prepared_500.csv")
print(f"Loaded {len(chebi_df)} ChEBI molecules")
print()

# Generate distractors for each molecule
distractors_list = []

for idx, row in tqdm(chebi_df.iterrows(), total=len(chebi_df), desc="Generating distractors"):
    mol_id = row['CID']
    smiles = row['SMILES']

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        mw = Descriptors.MolWt(mol)

        # Get scaffold
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else None
        except:
            scaffold_smiles = None

        # Exclude current molecule from candidates
        candidates = chebi_df[chebi_df['CID'] != mol_id].copy()

        # Distractor 1: Same scaffold
        scaffold_match = None
        if scaffold_smiles:
            for _, cand in candidates.iterrows():
                try:
                    cand_mol = Chem.MolFromSmiles(cand['SMILES'])
                    if cand_mol:
                        cand_scaffold = MurckoScaffold.GetScaffoldForMol(cand_mol)
                        cand_scaffold_smiles = Chem.MolToSmiles(cand_scaffold) if cand_scaffold else None
                        if cand_scaffold_smiles == scaffold_smiles:
                            scaffold_match = cand['CID']
                            break
                except:
                    continue

        if scaffold_match is None:
            scaffold_match = candidates.sample(n=1, random_state=SEED+idx)['CID'].values[0]

        # Distractor 2: Similar MW (±10%)
        mw_min = mw * 0.9
        mw_max = mw * 1.1
        mw_candidates = []

        for _, cand in candidates.iterrows():
            if cand['CID'] == scaffold_match:
                continue
            try:
                cand_mol = Chem.MolFromSmiles(cand['SMILES'])
                if cand_mol:
                    cand_mw = Descriptors.MolWt(cand_mol)
                    if mw_min <= cand_mw <= mw_max:
                        mw_candidates.append(cand['CID'])
            except:
                continue

        if mw_candidates:
            mw_match = np.random.choice(mw_candidates)
        else:
            mw_match = candidates[candidates['CID'] != scaffold_match].sample(n=1, random_state=SEED+idx)['CID'].values[0]

        # Distractor 3: Random
        random_match = candidates[
            (candidates['CID'] != scaffold_match) &
            (candidates['CID'] != mw_match)
        ].sample(n=1, random_state=SEED+idx)['CID'].values[0]

        distractors_list.append({
            'molecule_id': mol_id,
            'distractor_1_scaffold': scaffold_match,
            'distractor_2_mw': mw_match,
            'distractor_3_random': random_match,
        })

    except Exception as e:
        print(f"\nWarning: Failed for {mol_id}: {e}")
        continue

# Save distractors
distractors_df = pd.DataFrame(distractors_list)
output_path = DATA_DIR / "retrieval_distractors.csv"
distractors_df.to_csv(output_path, index=False)

print()
print("=" * 80)
print(f"✓ Created {output_path}")
print(f"  {len(distractors_df)} molecules with distractors")
print("=" * 80)
