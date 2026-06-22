"""
sample_500_chebi.py - Sample 500 molecules from ChEBI-20 with complete representation coverage

This script:
1. Loads ChEBI-20 test set
2. Samples molecules and converts to all 7 representations
3. Keeps sampling until we get 500 molecules with ALL representations successfully converted
4. Saves the prepared dataset for benchmarks B3, B4, B5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from tqdm import tqdm

# Import from project
from config import (
    CHEBI20_TEST,
    DATA_DIR, REPRESENTATIONS, SEED
)
from utils.representations import convert_to_representation
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

np.random.seed(SEED)

# ============================================================================
# Helper Functions
# ============================================================================

def convert_molecules_to_representations(df):
    """Convert all molecules in dataframe to 7 representations.

    Removes any molecules that fail conversion for ANY representation.
    Uses native IUPAC and SELFIES from ChEBI if available, otherwise uses PubChem API.
    """
    print(f"\nConverting molecules to all {len(REPRESENTATIONS)} representations...")

    results = {rep: [] for rep in REPRESENTATIONS}
    conversion_stats = {rep: {"success": 0, "fail": 0} for rep in REPRESENTATIONS}
    failures = {rep: [] for rep in REPRESENTATIONS}
    molecules_to_keep = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
        smiles = row['SMILES']
        mol_id = row['CID']

        # Convert to all representations
        converted_values = {}
        all_successful = True

        for rep in REPRESENTATIONS:
            # For IUPAC and SELFIES, prefer native ChEBI values
            if rep == "iupac":
                native_val = row.get('iupacname', None)
                converted = native_val if pd.notna(native_val) else convert_to_representation(smiles, rep)
            elif rep == "selfies":
                native_val = row.get('SELFIES', None)
                converted = native_val if pd.notna(native_val) else convert_to_representation(smiles, rep)
            else:
                converted = convert_to_representation(smiles, rep)

            converted_values[rep] = converted

            if converted is not None:
                conversion_stats[rep]["success"] += 1
            else:
                conversion_stats[rep]["fail"] += 1
                all_successful = False
                failures[rep].append({
                    "molecule_id": mol_id,
                    "smiles": smiles,
                    "representation": rep,
                })

        # Only keep molecule if ALL representations succeeded
        if all_successful:
            molecules_to_keep.append(idx)
            for rep in REPRESENTATIONS:
                results[rep].append(converted_values[rep])

    # Filter dataframe to only successful molecules
    df_filtered = df.loc[molecules_to_keep].reset_index(drop=True)

    # Add representation columns to filtered dataframe
    for rep in REPRESENTATIONS:
        df_filtered[rep] = results[rep]

    # Print conversion stats
    print(f"\nConversion Statistics:")
    for rep in REPRESENTATIONS:
        success = conversion_stats[rep]["success"]
        total = success + conversion_stats[rep]["fail"]
        rate = (success / total * 100) if total > 0 else 0
        print(f"  {rep:20s}: {success:5d}/{total:5d} ({rate:5.1f}%)")

    print(f"\nMolecules retained: {len(df_filtered)}/{len(df)} ({len(df_filtered)/len(df)*100:.1f}%)")
    print(f"Molecules removed: {len(df) - len(df_filtered)}")

    return df_filtered, failures


# ============================================================================
# Main Processing
# ============================================================================

def main():
    print("=" * 80)
    print("ChEBI-20 Sampling: 500 molecules with complete representation coverage")
    print("=" * 80)

    # ========================================================================
    # 1. Load full ChEBI-20 test set
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 1: Loading ChEBI-20 test set")
    print("=" * 80)

    if not CHEBI20_TEST.exists():
        print(f"ERROR: ChEBI-20 test file not found at {CHEBI20_TEST}")
        sys.exit(1)

    df_full = pd.read_csv(CHEBI20_TEST)
    print(f"Loaded {len(df_full)} molecules from ChEBI-20 test set")
    print(f"Columns: {df_full.columns.tolist()}")

    # ========================================================================
    # 2. Sample and convert until we get 500 successful molecules
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Sampling and converting to all representations")
    print("=" * 80)
    print("Target: 500 successful conversions")
    print("Strategy: Sample in batches, convert, keep successes, resample if needed")

    successful_molecules = []
    all_failures = []
    attempted_indices = set()
    batch_size = 200  # Sample in batches
    batch_num = 0

    while True:
        # Count total successful molecules so far
        current_total = sum(len(df) for df in successful_molecules)

        if current_total >= 500:
            break

        remaining_needed = 500 - current_total
        batch_num += 1

        print(f"\n--- Batch {batch_num} ---")
        print(f"Current successes: {current_total}/500 molecules")
        print(f"Still need: {remaining_needed} molecules")

        # Sample new molecules (not yet attempted)
        available_indices = [i for i in df_full.index if i not in attempted_indices]

        if len(available_indices) == 0:
            print("ERROR: Ran out of molecules in ChEBI-20 test set!")
            print(f"Only managed to convert {current_total} molecules successfully")
            break

        n_sample = min(batch_size, len(available_indices))
        sample_indices = np.random.choice(available_indices, size=n_sample, replace=False)
        attempted_indices.update(sample_indices)

        df_batch = df_full.loc[sample_indices].reset_index(drop=True)
        print(f"Sampled {len(df_batch)} new molecules to try...")

        # Try converting this batch
        df_converted, batch_failures = convert_molecules_to_representations(df_batch)

        # Collect failures
        for rep, fail_list in batch_failures.items():
            all_failures.extend(fail_list)

        # Add successful conversions
        if len(df_converted) > 0:
            successful_molecules.append(df_converted)
            print(f"Added {len(df_converted)} successful conversions")

    # Combine all successful molecules
    if len(successful_molecules) == 0:
        print("ERROR: No molecules converted successfully!")
        sys.exit(1)

    df = pd.concat(successful_molecules, ignore_index=True)

    # Take exactly 500 if we got more
    if len(df) > 500:
        df = df.sample(n=500, random_state=SEED).reset_index(drop=True)

    print(f"\n{'='*80}")
    print(f"Successfully converted {len(df)} molecules!")
    print(f"Total molecules attempted: {len(attempted_indices)}")
    print(f"Success rate: {len(df)/len(attempted_indices)*100:.1f}%")

    # ========================================================================
    # 3. Save prepared dataset
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Saving prepared dataset")
    print("=" * 80)

    # Save full dataset with all representations
    output_path = DATA_DIR / "chebi_prepared_500.csv"
    df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")
    print(f"  Size: {len(df)} molecules")
    print(f"  Columns: {len(df.columns)}")

    # Save just molecule IDs (for benchmark scripts to reference)
    ids_path = DATA_DIR / "chebi_molecule_ids.csv"
    df[['CID']].rename(columns={'CID': 'molecule_id'}).to_csv(ids_path, index=False)
    print(f"✓ Saved molecule IDs: {ids_path}")

    # Save conversion failures
    if all_failures:
        failures_df = pd.DataFrame(all_failures)
        failures_path = DATA_DIR / "chebi_conversion_failures.csv"
        failures_df.to_csv(failures_path, index=False)
        print(f"✓ Saved {len(all_failures)} conversion failures to: {failures_path}")

    # ========================================================================
    # 4. Create benchmark-specific metadata for B4 and B5
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Creating benchmark-specific metadata")
    print("=" * 80)

    # B4: Retrieval distractors
    print("\nCreating retrieval distractors for B4...")
    distractors_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Generating distractors"):
        mol_id = row['CID']
        smiles = row['SMILES']

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            mw = Descriptors.MolWt(mol)
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else None
            except:
                scaffold_smiles = None

            candidates = df[df['CID'] != mol_id].copy()

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
        except:
            continue

    distractors_df = pd.DataFrame(distractors_list)
    distractors_path = DATA_DIR / "retrieval_distractors.csv"
    distractors_df.to_csv(distractors_path, index=False)
    print(f"✓ Saved: {distractors_path} ({len(distractors_df)} molecules)")

    # B5: Isomer pairs
    print("\nCreating isomer pairs for B5...")
    isomer_pairs = []

    for i in tqdm(range(len(df)), desc="  Searching for isomers"):
        mol1_id = df.iloc[i]['CID']
        mol1_smiles = df.iloc[i]['SMILES']

        try:
            mol1 = Chem.MolFromSmiles(mol1_smiles)
            if mol1 is None:
                continue

            formula1 = Chem.rdMolDescriptors.CalcMolFormula(mol1)

            for j in range(i+1, len(df)):
                mol2_id = df.iloc[j]['CID']
                mol2_smiles = df.iloc[j]['SMILES']

                try:
                    mol2 = Chem.MolFromSmiles(mol2_smiles)
                    if mol2 is None:
                        continue

                    formula2 = Chem.rdMolDescriptors.CalcMolFormula(mol2)

                    if formula1 == formula2 and mol1_smiles != mol2_smiles:
                        isomer_pairs.append({
                            'molecule_id_1': mol1_id,
                            'smiles_1': mol1_smiles,
                            'molecule_id_2': mol2_id,
                            'smiles_2': mol2_smiles,
                            'molecular_formula': formula1,
                        })

                        if len(isomer_pairs) >= 100:
                            break
                except:
                    continue

            if len(isomer_pairs) >= 100:
                break
        except:
            continue

    if len(isomer_pairs) < 10:
        print("  Warning: Few isomer pairs found, creating synthetic stereoisomers...")
        for idx, row in df.head(50).iterrows():
            if len(isomer_pairs) >= 50:
                break
            mol_id = row['CID']
            smiles = row['SMILES']
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
                if len(chiral_centers) > 0:
                    mol_copy = Chem.Mol(mol)
                    Chem.RemoveStereochemistry(mol_copy)
                    smiles_no_stereo = Chem.MolToSmiles(mol_copy)
                    isomer_pairs.append({
                        'molecule_id_1': mol_id,
                        'smiles_1': smiles,
                        'molecule_id_2': f"{mol_id}_stereoisomer",
                        'smiles_2': smiles_no_stereo,
                        'molecular_formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
                    })
            except:
                continue

    isomer_pairs_df = pd.DataFrame(isomer_pairs)
    isomer_pairs_path = DATA_DIR / "isomer_pairs.csv"
    isomer_pairs_df.to_csv(isomer_pairs_path, index=False)
    print(f"✓ Saved: {isomer_pairs_path} ({len(isomer_pairs_df)} pairs)")

    # ========================================================================
    # 5. Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nPrepared ChEBI-20 dataset:")
    print(f"  Source: {CHEBI20_TEST}")
    print(f"  Output: {output_path}")
    print(f"  Molecules: {len(df)}")
    print(f"  Total attempted: {len(attempted_indices)}")
    print(f"  Success rate: {len(df)/len(attempted_indices)*100:.1f}%")

    print(f"\nRepresentations prepared:")
    for rep in REPRESENTATIONS:
        count = df[rep].notna().sum()
        print(f"  {rep:20s}: {count}/{len(df)} molecules")

    print(f"\nFiles created:")
    print(f"  ✓ {output_path.name} - Full dataset with all representations")
    print(f"  ✓ {ids_path.name} - Molecule IDs only")
    print(f"  ✓ {distractors_path.name} - Distractors for B4 ({len(distractors_df)} molecules)")
    print(f"  ✓ {isomer_pairs_path.name} - Isomer pairs for B5 ({len(isomer_pairs_df)} pairs)")
    if all_failures:
        print(f"  ✓ chebi_conversion_failures.csv - Failed conversions")

    print("\n" + "=" * 80)
    print("Dataset sampling complete!")
    print("=" * 80)

    print("\nDataset Usage:")
    print("  • ChEBI (500 molecules) → B3, B4, B5")
    print("  • ZINC (500 molecules) → B1, B2, B6, B7")
    print("\nNext step:")
    print("  Run 01_prepare_dataset.py to create benchmark-specific files")


if __name__ == "__main__":
    main()
