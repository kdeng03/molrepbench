"""
sample_500_zinc250k_tiers.py - Sample 500 molecules from ZINC250K with complexity-based stratification

This script:
1. Loads ZINC250K dataset
2. Computes molecular complexity scores based on multiple features
3. Stratifies molecules into 5 tiers (simple to very hard)
4. Samples 100 molecules from each tier (500 total)
5. Saves the sampled molecule IDs for use in benchmarks B1, B2, B6, B7
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm

from config import DATA_DIR, SEED

# Set random seed
np.random.seed(SEED)

# ============================================================================
# Molecular Feature Computation
# ============================================================================

def compute_molecular_features(smiles):
    """
    Compute molecular features needed for complexity scoring.

    Returns dict with:
    - num_heavy_atoms
    - num_rings
    - num_stereocenters
    - max_paren_depth
    - num_ring_closures
    - smiles_length
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Basic properties
        num_heavy_atoms = mol.GetNumHeavyAtoms()
        num_rings = rdMolDescriptors.CalcNumRings(mol)

        # Stereochemistry
        num_stereocenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))

        # SMILES-specific features
        max_paren_depth = compute_max_parenthesis_depth(smiles)
        num_ring_closures = sum(1 for c in smiles if c.isdigit())
        smiles_length = len(smiles)

        return {
            'num_heavy_atoms': num_heavy_atoms,
            'num_rings': num_rings,
            'num_stereocenters': num_stereocenters,
            'max_paren_depth': max_paren_depth,
            'num_ring_closures': num_ring_closures,
            'smiles_length': smiles_length,
        }
    except Exception as e:
        print(f"Error computing features for {smiles}: {e}")
        return None


def compute_max_parenthesis_depth(smiles):
    """Compute maximum nesting depth of parentheses in SMILES."""
    depth = 0
    max_depth = 0
    for char in smiles:
        if char == '(':
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == ')':
            depth -= 1
    return max_depth


def compute_complexity_score(row):
    """
    Weighted sum of features that make representation parsing harder.

    Formula:
    - 30% size (heavy atoms)
    - 20% ring complexity
    - 15% stereochemistry
    - 15% SMILES nesting (parenthesis depth)
    - 10% ring closure burden
    - 10% token length
    """
    return (
        0.3 * row['num_heavy_atoms'] / 50 +          # size
        0.2 * row['num_rings'] / 6 +                  # ring complexity
        0.15 * row['num_stereocenters'] / 5 +         # stereochemistry
        0.15 * row['max_paren_depth'] / 8 +           # SMILES nesting
        0.1 * row['num_ring_closures'] / 10 +         # ring closure burden
        0.1 * row['smiles_length'] / 100              # token length
    )


# ============================================================================
# Main Processing
# ============================================================================

def main():
    print("=" * 80)
    print("ZINC250K Sampling with Complexity-Based Stratification")
    print("=" * 80)

    # ========================================================================
    # 1. Load ZINC250K dataset
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 1: Loading ZINC250K dataset")
    print("=" * 80)

    zinc_path = Path("/node1/arunraja/benchmark/zinc250k.csv")

    if not zinc_path.exists():
        print(f"ERROR: ZINC250K file not found at {zinc_path}")
        print("Please verify the path is correct.")
        sys.exit(1)

    print(f"Loading from: {zinc_path}")
    df = pd.read_csv(zinc_path)
    print(f"Loaded {len(df)} molecules")

    # Check if SMILES column exists
    smiles_col = None
    for col in ['smiles', 'SMILES', 'Smiles']:
        if col in df.columns:
            smiles_col = col
            break

    if smiles_col is None:
        print(f"ERROR: No SMILES column found. Available columns: {df.columns.tolist()}")
        sys.exit(1)

    print(f"Using SMILES column: '{smiles_col}'")

    # Rename to standard 'smiles' column
    if smiles_col != 'smiles':
        df = df.rename(columns={smiles_col: 'smiles'})

    # Add molecule ID if not present
    if 'molecule_id' not in df.columns:
        if 'zinc_id' in df.columns:
            df['molecule_id'] = df['zinc_id']
        else:
            df['molecule_id'] = [f"ZINC_{i:06d}" for i in range(len(df))]
        print(f"Added molecule_id column")

    # ========================================================================
    # 2. Compute molecular features
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Computing molecular features for complexity scoring")
    print("=" * 80)

    features_list = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing features"):
        features = compute_molecular_features(row['smiles'])
        if features is not None:
            features_list.append(features)
            valid_indices.append(idx)
        else:
            print(f"Skipping invalid SMILES at index {idx}: {row['smiles']}")

    # Create features dataframe
    features_df = pd.DataFrame(features_list)
    df_valid = df.loc[valid_indices].reset_index(drop=True)

    # Merge features
    for col in features_df.columns:
        df_valid[col] = features_df[col].values

    print(f"\nValid molecules: {len(df_valid)}/{len(df)}")
    print(f"Invalid/skipped: {len(df) - len(df_valid)}")

    # ========================================================================
    # 3. Compute complexity scores
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Computing complexity scores")
    print("=" * 80)

    df_valid['complexity_score'] = df_valid.apply(compute_complexity_score, axis=1)

    print(f"Complexity score statistics:")
    print(df_valid['complexity_score'].describe())

    # ========================================================================
    # 4. Stratify into 5 tiers
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Stratifying into 5 complexity tiers")
    print("=" * 80)

    df_valid['tier'] = pd.qcut(
        df_valid['complexity_score'],
        q=5,
        labels=['tier1_simple', 'tier2_medium', 'tier3_complex', 'tier4_hard', 'tier5_veryhard'],
        duplicates='drop'
    )

    print("\nTier distribution:")
    tier_counts = df_valid['tier'].value_counts().sort_index()
    for tier, count in tier_counts.items():
        print(f"  {tier:20s}: {count:6d} molecules")

    # ========================================================================
    # 5. Sample 100 molecules from each tier
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 5: Sampling 100 molecules from each tier")
    print("=" * 80)

    sampled_molecules = []

    for tier in ['tier1_simple', 'tier2_medium', 'tier3_complex', 'tier4_hard', 'tier5_veryhard']:
        tier_df = df_valid[df_valid['tier'] == tier]

        if len(tier_df) < 100:
            print(f"WARNING: {tier} has only {len(tier_df)} molecules (need 100)")
            n_sample = len(tier_df)
        else:
            n_sample = 100

        sampled = tier_df.sample(n=n_sample, random_state=SEED)
        sampled_molecules.append(sampled)

        print(f"  {tier:20s}: sampled {len(sampled)}/{len(tier_df)} molecules")

    # Concatenate all samples
    df_sampled = pd.concat(sampled_molecules, ignore_index=True)

    print(f"\nTotal sampled: {len(df_sampled)} molecules")

    # ========================================================================
    # 6. Save results
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 6: Saving results")
    print("=" * 80)

    # Create data directory if needed
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save full sampled dataset with all features
    output_full = DATA_DIR / "zinc250k_sampled_500.csv"
    df_sampled.to_csv(output_full, index=False)
    print(f"✓ Saved full dataset: {output_full}")

    # Save just molecule IDs for benchmark use (matching ChEBI format)
    output_ids = DATA_DIR / "zinc_subset_ids.csv"
    df_sampled[['molecule_id']].to_csv(output_ids, index=False)
    print(f"✓ Saved molecule IDs: {output_ids}")

    # Save tier statistics
    output_stats = DATA_DIR / "zinc_sampling_stats.csv"
    stats_df = df_sampled.groupby('tier').agg({
        'complexity_score': ['mean', 'std', 'min', 'max'],
        'num_heavy_atoms': 'mean',
        'num_rings': 'mean',
        'num_stereocenters': 'mean',
        'smiles_length': 'mean',
    }).round(3)
    stats_df.to_csv(output_stats)
    print(f"✓ Saved statistics: {output_stats}")

    # ========================================================================
    # 7. Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nSampled 500 molecules from ZINC250K:")
    print(f"  Source: {zinc_path}")
    print(f"  Output: {output_full}")
    print(f"\nTier distribution (target: 100 each):")
    for tier, count in df_sampled['tier'].value_counts().sort_index().items():
        print(f"  {tier:20s}: {count:3d} molecules")

    print(f"\nComplexity score range per tier:")
    for tier in df_sampled['tier'].unique():
        tier_data = df_sampled[df_sampled['tier'] == tier]
        min_score = tier_data['complexity_score'].min()
        max_score = tier_data['complexity_score'].max()
        mean_score = tier_data['complexity_score'].mean()
        print(f"  {tier:20s}: {mean_score:.3f} (range: {min_score:.3f} - {max_score:.3f})")

    print("\n" + "=" * 80)
    print("Sampling complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Update config.py to use ZINC dataset instead of ChEBI-20")
    print("  2. Run 01_prepare_dataset.py to convert to all 7 representations")
    print("  3. Run inference scripts for B1, B2, B6, B7")


if __name__ == "__main__":
    main()
