"""
01_prepare_dataset.py - Prepare ChEBI-20 dataset for benchmarking

This script:
1. Loads ChEBI-20 from the already-downloaded files
2. Converts molecules to all 6 representations
3. Creates stratified comprehension subset (500 molecules)
4. Prepares benchmark-specific data (distractors, pairs, partials, few-shot examples)
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rdkit import Chem
from rdkit.Chem import Descriptors, Scaffolds, AllChem, DataStructs, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

# Import from project
from config import (
    CHEBI20_TRAIN, CHEBI20_VAL, CHEBI20_TEST,
    DATA_DIR, REPRESENTATIONS, SEED,
    COMPREHENSION_SAMPLE_SIZE,
    MOLECULAR_WEIGHT_BINS, RING_COUNT_BINS,
    ISOMER_PAIRS_POSITIVE, ISOMER_PAIRS_NEGATIVE,
    ISOMER_PAIRS_STEREOISOMER, ISOMER_PAIRS_SUBSTITUTION,
    N_SHOT,
    COMPLEXITY_FLOOR_MW, COMPLEXITY_FLOOR_RINGS,
)
from utils.representations import convert_to_representation
from utils.chemistry import (
    calculate_molecular_weight, count_rings, has_stereocenters,
    molecules_have_same_scaffold, canonicalize_smiles
)

np.random.seed(SEED)
random.seed(SEED)

# Path to full ZINC250K dataset (used for B9 and B10)
ZINC250K_PATH = Path("/node1/arunraja/benchmark/zinc250k.csv")

# ============================================================================
# Helper Functions
# ============================================================================

def categorize_molecular_weight(mw):
    """Categorize molecule by weight."""
    for min_mw, max_mw, label in MOLECULAR_WEIGHT_BINS:
        if min_mw <= mw < max_mw:
            return label
    return "unknown"

def categorize_rings(n_rings):
    """Categorize molecule by ring count."""
    for min_r, max_r, label in RING_COUNT_BINS:
        if min_r <= n_rings <= max_r:
            return label
    return "unknown"

def load_chebi20_split(csv_path):
    """Load a ChEBI-20 split CSV."""
    print(f"Loading {csv_path.name}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} molecules")
    return df

def convert_molecules_to_representations(df, split_name):
    """Convert all molecules in dataframe to 6 representations."""
    print(f"\nConverting {split_name} molecules to all representations...")

    results = {rep: [] for rep in REPRESENTATIONS}
    conversion_stats = {rep: {"success": 0, "fail": 0} for rep in REPRESENTATIONS}
    iupac_failures = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Converting {split_name}"):
        smiles = row['SMILES']
        mol_id = row['CID']

        for rep in REPRESENTATIONS:
            # For IUPAC and SELFIES, use existing columns from ChEBI-20 instead of converting
            if rep == "iupac":
                native_val = row.get('iupacname', None)
                # Handle NaN values from pandas
                converted = native_val if pd.notna(native_val) else None
            elif rep == "selfies":
                native_val = row.get('SELFIES', None)
                # Handle NaN values from pandas
                converted = native_val if pd.notna(native_val) else None
            else:
                converted = convert_to_representation(smiles, rep)

            if converted is not None:
                results[rep].append(converted)
                conversion_stats[rep]["success"] += 1
            else:
                results[rep].append(None)
                conversion_stats[rep]["fail"] += 1

                # Log IUPAC failures
                if rep == "iupac":
                    iupac_failures.append({
                        "molecule_id": mol_id,
                        "smiles": smiles,
                        "split": split_name
                    })

    # Add representation columns to dataframe
    for rep in REPRESENTATIONS:
        df[rep] = results[rep]

    # Print conversion stats
    print(f"\n{split_name} Conversion Statistics:")
    for rep in REPRESENTATIONS:
        success = conversion_stats[rep]["success"]
        total = success + conversion_stats[rep]["fail"]
        rate = (success / total * 100) if total > 0 else 0
        print(f"  {rep:20s}: {success:5d}/{total:5d} ({rate:5.1f}%)")

    return df, iupac_failures

def compute_molecular_properties(df):
    """Compute molecular properties for stratification."""
    print("\nComputing molecular properties...")

    mw_list = []
    ring_list = []
    stereo_list = []

    for smiles in tqdm(df['SMILES'], desc="Computing properties"):
        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            mw = calculate_molecular_weight(mol)
            rings = count_rings(mol)
            stereo = has_stereocenters(mol)
        else:
            mw = None
            rings = 0
            stereo = False

        mw_list.append(mw)
        ring_list.append(rings)
        stereo_list.append(stereo)

    df['molecular_weight'] = mw_list
    df['n_rings'] = ring_list
    df['has_stereocenters'] = stereo_list

    # Add categorical bins
    df['mw_bin'] = df['molecular_weight'].apply(
        lambda x: categorize_molecular_weight(x) if pd.notna(x) else "unknown"
    )
    df['ring_bin'] = df['n_rings'].apply(categorize_rings)

    return df

def create_stratified_comprehension_subset(test_df, n_samples=500):
    """Create stratified subset for comprehension benchmarks.

    Applies a complexity floor (COMPLEXITY_FLOOR_MW, COMPLEXITY_FLOOR_RINGS) before
    stratified sampling so that trivially easy molecules are excluded. This ensures
    representations are tested on cases where structural encoding actually matters.
    """
    print(f"\nCreating stratified comprehension subset ({n_samples} molecules)...")

    # Apply complexity floor before sampling
    before = len(test_df)
    test_df = test_df[
        (test_df['molecular_weight'] >= COMPLEXITY_FLOOR_MW) &
        (test_df['n_rings'] >= COMPLEXITY_FLOOR_RINGS)
    ].copy()
    print(f"  Complexity floor (MW≥{COMPLEXITY_FLOOR_MW}, rings≥{COMPLEXITY_FLOOR_RINGS}): "
          f"{before} → {len(test_df)} molecules")
    if len(test_df) < n_samples:
        raise ValueError(
            f"Only {len(test_df)} molecules pass complexity floor but {n_samples} requested. "
            "Lower COMPLEXITY_FLOOR_MW/RINGS or reduce COMPREHENSION_SAMPLE_SIZE."
        )

    # Create stratification key
    test_df['strat_key'] = (
        test_df['mw_bin'].astype(str) + "_" +
        test_df['ring_bin'].astype(str) + "_" +
        test_df['has_stereocenters'].astype(str)
    )

    # Sample proportionally from each stratum
    sampled_ids = []
    strat_counts = test_df['strat_key'].value_counts()

    print(f"  Found {len(strat_counts)} unique strata")

    # Proportional sampling
    for strat, count in strat_counts.items():
        proportion = count / len(test_df)
        n_from_strat = int(np.ceil(proportion * n_samples))

        strat_df = test_df[test_df['strat_key'] == strat]
        sample_size = min(n_from_strat, len(strat_df))

        sampled = strat_df.sample(n=sample_size, random_state=SEED)
        sampled_ids.extend(sampled['CID'].tolist())

    # If we oversampled, randomly drop excess
    if len(sampled_ids) > n_samples:
        sampled_ids = np.random.choice(sampled_ids, size=n_samples, replace=False).tolist()

    # If we undersampled, fill with random molecules
    if len(sampled_ids) < n_samples:
        remaining = test_df[~test_df['CID'].isin(sampled_ids)]
        additional = remaining.sample(n=n_samples - len(sampled_ids), random_state=SEED)
        sampled_ids.extend(additional['CID'].tolist())

    subset_df = test_df[test_df['CID'].isin(sampled_ids)].copy()

    # Print stratification distribution
    print("\n  Stratification distribution:")
    print(f"    MW bins: {subset_df['mw_bin'].value_counts().to_dict()}")
    print(f"    Ring bins: {subset_df['ring_bin'].value_counts().to_dict()}")
    print(f"    Stereocenters: {subset_df['has_stereocenters'].value_counts().to_dict()}")

    return sampled_ids

def _process_single_distractor(args):
    """Worker function to process a single molecule's distractors."""
    row_data, test_df_dict = args

    mol_id = row_data['CID']
    smiles = row_data['SMILES']
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return []

    mw = calculate_molecular_weight(mol)

    distractors = []

    # Find same scaffold distractor
    scaffold_dist = None
    for cand_id, cand_data in test_df_dict.items():
        if cand_id == mol_id:
            continue
        cand_mol = Chem.MolFromSmiles(cand_data['SMILES'])
        if cand_mol and molecules_have_same_scaffold(mol, cand_mol):
            scaffold_dist = cand_data
            break

    # Find similar MW distractor (±10%)
    mw_dist = None
    if mw:
        mw_min, mw_max = mw * 0.9, mw * 1.1
        for cand_id, cand_data in test_df_dict.items():
            if cand_id == mol_id:
                continue
            cand_mw = cand_data.get('molecular_weight')
            if cand_mw and mw_min <= cand_mw <= mw_max:
                mw_dist = cand_data
                break

    # Random distractor (exclude self, scaffold, and mw distractors)
    used_ids = {mol_id}
    if scaffold_dist:
        used_ids.add(scaffold_dist['CID'])
    if mw_dist:
        used_ids.add(mw_dist['CID'])

    random_candidates = [v for k, v in test_df_dict.items() if k not in used_ids]
    random_dist = random.choice(random_candidates) if random_candidates else None

    # Add distractors
    for dist_type, dist_data in [("scaffold", scaffold_dist), ("molecular_weight", mw_dist), ("random", random_dist)]:
        if dist_data is not None:
            distractors.append({
                "target_molecule_id": mol_id,
                "distractor_molecule_id": dist_data['CID'],
                "distractor_smiles": dist_data['SMILES'],
            })

    return distractors


def create_retrieval_distractors(test_df):
    """Create distractor molecules for retrieval benchmark (parallelized)."""
    from multiprocessing import Pool, cpu_count

    print("\nCreating retrieval distractors...")

    # Convert dataframe to dict for faster access in workers
    test_df_dict = {row['CID']: row.to_dict() for _, row in test_df.iterrows()}

    # Prepare arguments for parallel processing
    args_list = [(row.to_dict(), test_df_dict) for _, row in test_df.iterrows()]

    # Use all available CPUs
    n_workers = 8
    #  cpu_count()
    print(f"  Using {n_workers} worker processes...")

    # Process in parallel with progress bar
    all_distractors = []
    with Pool(n_workers) as pool:
        for result in tqdm(
            pool.imap(_process_single_distractor, args_list),
            total=len(args_list),
            desc="Creating distractors"
        ):
            all_distractors.extend(result)

    distractor_df = pd.DataFrame(all_distractors)
    print(f"  Created {len(distractor_df)} distractors ({len(distractor_df) / 3:.0f} test molecules)")

    return distractor_df

def create_isomer_pairs(test_df):
    """Create positive and negative pairs for isomer discrimination."""
    print("\nCreating isomer discrimination pairs...")

    pairs = []

    # Positive pairs (same molecule, different ordering)
    print(f"  Creating {ISOMER_PAIRS_POSITIVE} positive pairs...")
    sampled_mols = test_df.sample(n=ISOMER_PAIRS_POSITIVE, random_state=SEED)

    for _, row in tqdm(sampled_mols.iterrows(), total=len(sampled_mols), desc="Positive pairs"):
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol is None:
            continue

        # Generate two different randomized SMILES
        smiles1 = Chem.MolToSmiles(mol, doRandom=True)
        smiles2 = Chem.MolToSmiles(mol, doRandom=True)

        pairs.append({
            "molecule_id_1": row['CID'],
            "molecule_id_2": row['CID'],
            "smiles_1": smiles1,
            "smiles_2": smiles2,
            "pair_type": "positive",
            "ground_truth": True,
        })

    # Negative pairs - stereoisomers
    print(f"  Creating {ISOMER_PAIRS_STEREOISOMER} stereoisomer pairs...")
    # For simplicity, sample pairs with different stereocenters
    stereo_mols = test_df[test_df['has_stereocenters'] == True]

    for i in tqdm(range(ISOMER_PAIRS_STEREOISOMER), desc="Stereoisomer pairs"):
        if len(stereo_mols) < 2:
            break
        sample = stereo_mols.sample(n=2, random_state=SEED+i)
        row1, row2 = sample.iloc[0], sample.iloc[1]

        pairs.append({
            "molecule_id_1": row1['CID'],
            "molecule_id_2": row2['CID'],
            "smiles_1": row1['SMILES'],
            "smiles_2": row2['SMILES'],
            "pair_type": "stereoisomer",
            "ground_truth": False,
        })

    # Negative pairs - atom substitution
    print(f"  Creating {ISOMER_PAIRS_SUBSTITUTION} substitution pairs...")
    for i in tqdm(range(ISOMER_PAIRS_SUBSTITUTION), desc="Substitution pairs"):
        if len(test_df) < 2:
            break
        sample = test_df.sample(n=2, random_state=SEED+i+1000)
        row1, row2 = sample.iloc[0], sample.iloc[1]

        pairs.append({
            "molecule_id_1": row1['CID'],
            "molecule_id_2": row2['CID'],
            "smiles_1": row1['SMILES'],
            "smiles_2": row2['SMILES'],
            "pair_type": "substitution",
            "ground_truth": False,
        })

    pairs_df = pd.DataFrame(pairs)
    print(f"  Created {len(pairs_df)} total pairs")
    print(f"    Positive: {len(pairs_df[pairs_df['pair_type'] == 'positive'])}")
    print(f"    Stereoisomer: {len(pairs_df[pairs_df['pair_type'] == 'stereoisomer'])}")
    print(f"    Substitution: {len(pairs_df[pairs_df['pair_type'] == 'substitution'])}")

    # Pre-compute iupac_1 / iupac_2 from test_df so script 02 never calls PubChem.
    # The 'iupac' column was populated from ChEBI-20's native iupacname column.
    cid_to_iupac = {}
    for _, r in test_df.iterrows():
        cid = r.get("CID") if pd.notna(r.get("CID", None)) else r.get("molecule_id")
        val = r.get("iupac", None)
        cid_to_iupac[cid] = val if pd.notna(val) else None

    pairs_df["iupac_1"] = pairs_df["molecule_id_1"].map(cid_to_iupac)
    pairs_df["iupac_2"] = pairs_df["molecule_id_2"].map(cid_to_iupac)

    iupac_coverage = pairs_df["iupac_1"].notna().sum()
    print(f"    IUPAC coverage: {iupac_coverage}/{len(pairs_df)} pairs have iupac_1")

    return pairs_df

def compute_fewshot_examples(train_df, test_df, n_shot=2):
    """Pre-compute few-shot examples using TF-IDF similarity."""
    print(f"\nComputing {n_shot}-shot examples for generation benchmark...")

    # Fit TF-IDF on training descriptions
    train_descriptions = train_df['description'].fillna("").tolist()
    test_descriptions = test_df['description'].fillna("").tolist()

    print("  Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    train_vectors = vectorizer.fit_transform(train_descriptions)
    test_vectors = vectorizer.transform(test_descriptions)

    print("  Computing similarities...")
    similarities = cosine_similarity(test_vectors, train_vectors)

    # For each test molecule, find top-N most similar train molecules
    fewshot_mappings = []

    for test_idx, test_row in tqdm(test_df.iterrows(), total=len(test_df), desc="Finding examples"):
        # Get top N similar indices
        sim_scores = similarities[test_idx - test_df.index[0]]
        top_indices = np.argsort(sim_scores)[-n_shot:][::-1]

        example_ids = train_df.iloc[top_indices]['CID'].tolist()

        mapping = {"test_molecule_id": test_row['CID']}
        for i, ex_id in enumerate(example_ids, 1):
            mapping[f"example_{i}_id"] = ex_id

        fewshot_mappings.append(mapping)

    fewshot_df = pd.DataFrame(fewshot_mappings)
    print(f"  Created {len(fewshot_df)} few-shot mappings")

    return fewshot_df

def create_completion_partials(test_df):
    """Create partial strings for completion benchmark."""
    print("\nCreating completion partial strings...")

    partials = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Creating partials"):
        mol_id = row['CID']

        for rep in REPRESENTATIONS:
            rep_string = row[rep]

            if rep_string is None or pd.isna(rep_string):
                continue

            # Take first 50% of characters
            mid_point = len(rep_string) // 2
            partial = rep_string[:mid_point]

            partials.append({
                "molecule_id": mol_id,
                "representation": rep,
                "full_string": rep_string,
                "partial_string": partial,
            })

    partials_df = pd.DataFrame(partials)
    print(f"  Created {len(partials_df)} partial strings")

    return partials_df

# ============================================================================
# B9 / B10 Shared Utilities
# ============================================================================

def load_zinc250k():
    """Load the full ZINC250K dataset."""
    print(f"Loading ZINC250K from {ZINC250K_PATH}...")
    if not ZINC250K_PATH.exists():
        raise FileNotFoundError(
            f"ZINC250K not found at {ZINC250K_PATH}. "
            "Download from https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
        )
    df = pd.read_csv(ZINC250K_PATH)
    # Normalise column name
    if "smiles" not in df.columns:
        if "SMILES" in df.columns:
            df = df.rename(columns={"SMILES": "smiles"})
        else:
            raise ValueError(f"ZINC250K CSV has no 'smiles' column. Columns: {df.columns.tolist()}")
    df = df.dropna(subset=["smiles"])
    print(f"  Loaded {len(df)} molecules")
    return df


def _morgan_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _tanimoto(smi1, smi2):
    fp1, fp2 = _morgan_fp(smi1), _morgan_fp(smi2)
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def _add_representation_columns(pairs_df, chebi_df=None):
    """Add precomputed representation columns (except randomized_smiles) to a pairs DataFrame.

    When chebi_df is provided and pairs have molecule_id_1/molecule_id_2 columns, iupac and
    selfies are looked up from ChEBI's native iupacname/SELFIES columns — no PubChem calls.
    """
    REP_COLS = ["canonical_smiles", "isomeric_smiles", "deepsmiles", "iupac", "selfies", "cml", "inchi"]

    cid_to_iupac = {}
    cid_to_selfies = {}
    if chebi_df is not None:
        id_col = "CID" if "CID" in chebi_df.columns else "molecule_id"
        for _, r in chebi_df.iterrows():
            cid = r[id_col]
            iupac_val = r.get("iupacname") if "iupacname" in r.index else r.get("iupac")
            selfies_val = r.get("SELFIES")
            if pd.notna(iupac_val):
                cid_to_iupac[cid] = str(iupac_val)
            if pd.notna(selfies_val):
                cid_to_selfies[cid] = str(selfies_val)

    for mol_num in [1, 2]:
        smi_col = f"smiles_{mol_num}"
        mid_col = f"molecule_id_{mol_num}"
        has_mol_ids = mid_col in pairs_df.columns

        for rep in REP_COLS:
            col = f"{rep}_{mol_num}"
            if rep == "iupac" and has_mol_ids and cid_to_iupac:
                pairs_df[col] = pairs_df[mid_col].map(cid_to_iupac)
            elif rep == "selfies" and has_mol_ids and cid_to_selfies:
                pairs_df[col] = pairs_df[mid_col].map(cid_to_selfies)
            else:
                pairs_df[col] = pairs_df[smi_col].apply(
                    lambda s: convert_to_representation(s, rep)
                )
    return pairs_df


# ============================================================================
# Benchmark 9: Tautomer Pairs (TautomerPairs-250)
# ============================================================================

def create_tautomer_pairs(chebi_df):
    """
    Build TautomerPairs-250 from ChEBI-20 by GENERATING tautomers.

    Positive pairs: take any ChEBI molecule with ≥2 distinct tautomers,
    pair (canonical_tautomer, alternate_tautomer).
    Negative pairs: pair unrelated ChEBI molecules (verified not tautomers).

    Returns a DataFrame with 125 positive + 125 negative pairs.
    """
    print("\nBuilding TautomerPairs-250...")

    enumerator = rdMolStandardize.TautomerEnumerator()

    # ------------------------------------------------------------------
    # Ground-truth function (spec §B9)
    # ------------------------------------------------------------------
    def are_tautomers(smi1, smi2):
        mol1, mol2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)
        if mol1 is None or mol2 is None:
            return False
        try:
            c1 = Chem.MolToSmiles(enumerator.Canonicalize(mol1))
            c2 = Chem.MolToSmiles(enumerator.Canonicalize(mol2))
            return c1 == c2
        except Exception:
            return False

    # Tautomer type classifiers — applied post-hoc to label pair metadata
    TYPE_PATTERNS = [
        ("keto_enol",     ["[CX3](=O)[CH]", "OC=C"]),
        ("amide_imidic",  ["[NX3H][CX3](=[OX1])", "N=C-O"]),
        ("lactam_lactim", ["[NX3H1;r][CX3;r](=[OX1])"]),
        ("heterocyclic",  ["n1cc[nH]c1", "[nH]"]),
        ("nitroso_oxime", ["[NX2](=O)", "NO"]),
        ("thione_thiol",  ["[CX3](=S)", "S-"]),
    ]
    compiled_types = []
    for name, smarts_list in TYPE_PATTERNS:
        patts = [p for p in (Chem.MolFromSmarts(s) for s in smarts_list) if p is not None]
        compiled_types.append((name, patts))

    def classify_pair(smi1, smi2):
        for name, patts in compiled_types:
            for smi in (smi1, smi2):
                mol = Chem.MolFromSmiles(smi)
                if mol and any(mol.HasSubstructMatch(p) for p in patts):
                    return name
        return "other"

    chebi_records = list(zip(chebi_df["SMILES"].tolist(), chebi_df["CID"].tolist()))

    # ----------------------------------------------------------------
    # Positive pairs: enumerate tautomers for each ChEBI molecule
    # ----------------------------------------------------------------
    print("  Generating positive pairs via tautomer enumeration...")
    positive_pairs = []
    seen_pos = set()

    for smi, cid in tqdm(chebi_records, desc="  Enumerating tautomers", leave=False):
        if len(positive_pairs) >= 125:
            break
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            taut_mols = list(enumerator.Enumerate(mol))
            if len(taut_mols) < 2:
                continue
            taut_smiles = list({Chem.MolToSmiles(t) for t in taut_mols})
            if len(taut_smiles) < 2:
                continue
            canon_smi = Chem.MolToSmiles(enumerator.Canonicalize(mol))
            alts = [s for s in taut_smiles if s != canon_smi]
            if not alts:
                continue
            alt_smi = alts[0]
            key = (min(canon_smi, alt_smi), max(canon_smi, alt_smi))
            if key in seen_pos:
                continue
            seen_pos.add(key)
            taut_class = classify_pair(canon_smi, alt_smi)
            positive_pairs.append({
                "smiles_1": canon_smi,
                "smiles_2": alt_smi,
                "molecule_id_1": cid,
                "molecule_id_2": cid,
                "ground_truth": "Yes",
                "pair_type": f"positive_{taut_class}",
                "tautomer_class": taut_class,
                "source": "ChEBI-20",
            })
        except Exception:
            continue

    print(f"  Total positive pairs: {len(positive_pairs)}")

    # ----------------------------------------------------------------
    # Negative pairs: pair unrelated ChEBI molecules
    # ----------------------------------------------------------------
    print("  Generating negative pairs from unrelated ChEBI molecules...")
    pos_smiles_used = {p["smiles_1"] for p in positive_pairs} | {p["smiles_2"] for p in positive_pairs}

    neg_candidates = []
    for smi, cid in chebi_records:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in pos_smiles_used:
            neg_candidates.append((canon, cid))
        if len(neg_candidates) >= 600:
            break

    rng = random.Random(SEED)
    rng.shuffle(neg_candidates)

    negative_pairs = []
    seen_neg = set()
    i = 0
    while len(negative_pairs) < 125 and i + 1 < len(neg_candidates):
        (smi1, cid1), (smi2, cid2) = neg_candidates[i], neg_candidates[i + 1]
        i += 2
        if smi1 == smi2:
            continue
        key = (min(smi1, smi2), max(smi1, smi2))
        if key in seen_neg:
            continue
        if are_tautomers(smi1, smi2):
            continue
        seen_neg.add(key)
        negative_pairs.append({
            "smiles_1": smi1,
            "smiles_2": smi2,
            "molecule_id_1": cid1,
            "molecule_id_2": cid2,
            "ground_truth": "No",
            "pair_type": "negative_unrelated",
            "tautomer_class": "none",
            "source": "ChEBI-20",
        })

    print(f"  Total negative pairs: {len(negative_pairs)}")

    # ----------------------------------------------------------------
    # Assemble and verify
    # ----------------------------------------------------------------
    all_pairs = positive_pairs + negative_pairs
    pairs_df = pd.DataFrame(all_pairs)
    pairs_df = pairs_df.reset_index(drop=True)
    pairs_df.insert(0, "pair_id", pairs_df.index)

    # Compute tanimoto between each pair
    print("  Computing Tanimoto similarities for all pairs...")
    tanimotos = []
    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="    Tanimoto", leave=False):
        pre = row.get("_tanimoto")
        if pre is not None and not pd.isna(pre):
            tanimotos.append(float(pre))
        else:
            tanimotos.append(_tanimoto(row["smiles_1"], row["smiles_2"]))
    pairs_df["tanimoto_between_pair"] = tanimotos
    if "_tanimoto" in pairs_df.columns:
        pairs_df.drop(columns=["_tanimoto"], inplace=True)

    # Verify all labels
    print("  Verifying pair labels...")
    mismatches = 0
    for _, row in pairs_df.iterrows():
        is_taut = are_tautomers(row["smiles_1"], row["smiles_2"])
        expected = (row["ground_truth"] == "Yes")
        if is_taut != expected:
            mismatches += 1
    print(f"    Label mismatches: {mismatches} (should be 0)")

    # Add redundant canonical_smiles columns as per spec
    pairs_df["canonical_smiles_1"] = pairs_df["smiles_1"].apply(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
    )
    pairs_df["canonical_smiles_2"] = pairs_df["smiles_2"].apply(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
    )

    # ------------------------------------------------------------------
    # Balance positive / negative to equal counts
    # ------------------------------------------------------------------
    pos_df = pairs_df[pairs_df["ground_truth"] == "Yes"]
    neg_df = pairs_df[pairs_df["ground_truth"] == "No"]
    n_pos, n_neg = len(pos_df), len(neg_df)
    n_keep = min(n_pos, n_neg)

    if n_pos != 125 or n_neg != 125:
        print(f"\n  WARNING: collected {n_pos} positive / {n_neg} negative pairs "
              f"(targets: 125/125). Trimming both to {n_keep} for balance.")

    pos_df = pos_df.sample(n=n_keep, random_state=SEED).reset_index(drop=True)
    neg_df = neg_df.sample(n=n_keep, random_state=SEED).reset_index(drop=True)
    pairs_df = pd.concat([pos_df, neg_df], ignore_index=True)
    pairs_df["pair_id"] = range(len(pairs_df))

    # Add precomputed representation columns (IUPAC, SELFIES, DeepSMILES, isomeric)
    print("  Adding representation columns...")
    pairs_df = _add_representation_columns(pairs_df, chebi_df)

    print(f"\n  TautomerPairs-250 summary:")
    print(f"    Total pairs:    {len(pairs_df)}")
    print(f"    Positive (Yes): {(pairs_df['ground_truth'] == 'Yes').sum()}")
    print(f"    Negative (No):  {(pairs_df['ground_truth'] == 'No').sum()}")
    print(f"    IUPAC coverage: {pairs_df['iupac_1'].notna().sum()}/{len(pairs_df)}")
    print(f"    Pair types:\n{pairs_df['pair_type'].value_counts().to_string()}")

    return pairs_df


# ============================================================================
# Benchmark 10: Protonation Pairs (ProtonationPairs-250)
# ============================================================================

def create_protonation_pairs(chebi_df):
    """
    Build ProtonationPairs-250 from ChEBI-20.

    Returns a DataFrame with 125 positive (protonation variant) + 125 negative pairs.
    """
    print("\nBuilding ProtonationPairs-250...")

    uncharger = rdMolStandardize.Uncharger()

    # ------------------------------------------------------------------
    # Ground-truth function (spec §B10)
    # ------------------------------------------------------------------
    def are_protonation_variants(smi1, smi2):
        mol1, mol2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)
        if mol1 is None or mol2 is None:
            return False
        try:
            n1 = Chem.MolToSmiles(uncharger.uncharge(Chem.RWMol(mol1)))
            n2 = Chem.MolToSmiles(uncharger.uncharge(Chem.RWMol(mol2)))
            return n1 == n2
        except Exception:
            return False

    def net_charge(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return 0
        return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

    # ------------------------------------------------------------------
    # Protonation functions (spec §B10)
    # ------------------------------------------------------------------
    def deprotonate_acid(smi):
        mol = Chem.RWMol(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else None
        if mol is None:
            return None
        patt = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            return None
        oh_idx = matches[0][2]
        atom = mol.GetAtomWithIdx(oh_idx)
        atom.SetFormalCharge(-1)
        atom.SetNumExplicitHs(0)
        try:
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    def protonate_amine(smi):
        mol = Chem.RWMol(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else None
        if mol is None:
            return None
        patt = Chem.MolFromSmarts("[NX3;H1,H2;!$(NC=O);!$(NS=O)]")
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            return None
        n_idx = matches[0][0]
        atom = mol.GetAtomWithIdx(n_idx)
        atom.SetFormalCharge(1)
        atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
        try:
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    def deprotonate_phenol(smi):
        mol = Chem.RWMol(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else None
        if mol is None:
            return None
        patt = Chem.MolFromSmarts("[cX3][OX2H1]")
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            return None
        oh_idx = matches[0][1]
        atom = mol.GetAtomWithIdx(oh_idx)
        atom.SetFormalCharge(-1)
        atom.SetNumExplicitHs(0)
        try:
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    def make_zwitterion(smi):
        mol = Chem.RWMol(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else None
        if mol is None:
            return None
        acid_patt = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
        base_patt = Chem.MolFromSmarts("[NX3;H1,H2;!$(NC=O)]")
        acid_matches = mol.GetSubstructMatches(acid_patt)
        base_matches = mol.GetSubstructMatches(base_patt)
        if not acid_matches or not base_matches:
            return None
        oh_idx = acid_matches[0][2]
        mol.GetAtomWithIdx(oh_idx).SetFormalCharge(-1)
        mol.GetAtomWithIdx(oh_idx).SetNumExplicitHs(0)
        n_idx = base_matches[0][0]
        n_atom = mol.GetAtomWithIdx(n_idx)
        n_atom.SetFormalCharge(1)
        n_atom.SetNumExplicitHs(n_atom.GetTotalNumHs() + 1)
        try:
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    def deprotonate_sulfonamide(smi):
        mol = Chem.RWMol(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else None
        if mol is None:
            return None
        # Try sulfonamide NH first
        patt_nh = Chem.MolFromSmarts("[SX4](=O)(=O)[NH1]")
        # Try sulfonic acid OH
        patt_oh = Chem.MolFromSmarts("[SX4](=O)(=O)[OX2H1]")
        for patt, idx in [(patt_nh, 3), (patt_oh, 3)]:
            matches = mol.GetSubstructMatches(patt)
            if matches:
                atom = mol.GetAtomWithIdx(matches[0][idx])
                atom.SetFormalCharge(-1)
                atom.SetNumExplicitHs(0)
                try:
                    Chem.SanitizeMol(mol)
                    return Chem.MolToSmiles(mol)
                except Exception:
                    return None
        return None

    def deprotonate_phosphate(smi):
        mol = Chem.RWMol(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else None
        if mol is None:
            return None
        patt = Chem.MolFromSmarts("[PX4](=O)([OX2H1])")
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            return None
        oh_idx = matches[0][2]
        atom = mol.GetAtomWithIdx(oh_idx)
        atom.SetFormalCharge(-1)
        atom.SetNumExplicitHs(0)
        try:
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    chebi_records = list(zip(chebi_df["SMILES"].tolist(), chebi_df["CID"].tolist()))

    # ----------------------------------------------------------------
    # Positive pairs — stratified by ionizable group type
    # ----------------------------------------------------------------
    SUBTYPE_CONFIG = [
        ("acid_base",        40, "[CX3](=O)[OX2H1]",              deprotonate_acid),
        ("amine_ammonium",   30, "[NX3;H1,H2;!$(NC=O);!$(NS=O)]", protonate_amine),
        ("phenol_phenolate", 15, "[cX3][OX2H1]",                  deprotonate_phenol),
        ("zwitterion",       20, None,                              make_zwitterion),   # needs both acid+base
        ("sulfonamide",      10, "[SX4](=O)(=O)[NH1]",             deprotonate_sulfonamide),
        ("phosphate",        10, "[PX4](=O)([OX2H1])",             deprotonate_phosphate),
    ]

    # For zwitterion subtype, filter needs both COOH and amine
    zwitterion_acid = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
    zwitterion_base = Chem.MolFromSmarts("[NX3;H1,H2;!$(NC=O)]")

    positive_pairs = []

    for subtype, target_n, smarts_str, transform_fn in SUBTYPE_CONFIG:
        if smarts_str is not None:
            patt = Chem.MolFromSmarts(smarts_str)
        else:
            patt = None

        collected = 0
        print(f"  Positive [{subtype}]: targeting {target_n} pairs...")
        for smi, cid in tqdm(chebi_records, desc=f"    {subtype}", leave=False):
            if collected >= target_n:
                break
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            # Substructure filter
            if patt is not None and not mol.HasSubstructMatch(patt):
                continue
            if subtype == "zwitterion":
                if not (mol.HasSubstructMatch(zwitterion_acid) and mol.HasSubstructMatch(zwitterion_base)):
                    continue
            # Neutral form
            neutral_smi = Chem.MolToSmiles(uncharger.uncharge(Chem.RWMol(mol)))
            charged_smi = transform_fn(neutral_smi)
            if charged_smi is None:
                continue
            if charged_smi == neutral_smi:
                continue
            # Verify different charge
            if net_charge(neutral_smi) == net_charge(charged_smi):
                continue
            # Verify ground truth
            if not are_protonation_variants(neutral_smi, charged_smi):
                continue
            positive_pairs.append({
                "smiles_1": neutral_smi,
                "smiles_2": charged_smi,
                "molecule_id_1": cid,
                "molecule_id_2": cid,
                "ground_truth": "Yes",
                "pair_type": f"positive_{subtype}",
                "ionizable_group": subtype,
                "charge_1": net_charge(neutral_smi),
                "charge_2": net_charge(charged_smi),
                "source": "ChEBI-20",
            })
            collected += 1
        print(f"    Collected {collected}/{target_n}")

    print(f"  Total positive pairs: {len(positive_pairs)}")

    # ----------------------------------------------------------------
    # Negative pairs: pair unrelated ZINC molecules
    # ----------------------------------------------------------------
    print("  Generating negative pairs from unrelated ChEBI molecules...")
    pos_smiles_used = {p["smiles_1"] for p in positive_pairs} | {p["smiles_2"] for p in positive_pairs}

    neg_candidates = []
    for smi, cid in chebi_records:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in pos_smiles_used and net_charge(canon) == 0:
            neg_candidates.append((canon, cid))
        if len(neg_candidates) >= 600:
            break

    rng_neg = random.Random(SEED + 10)
    rng_neg.shuffle(neg_candidates)

    negative_pairs = []
    seen_neg = set()
    i = 0
    while len(negative_pairs) < 125 and i + 1 < len(neg_candidates):
        (smi1, cid1), (smi2, cid2) = neg_candidates[i], neg_candidates[i + 1]
        i += 2
        if smi1 == smi2:
            continue
        key = (min(smi1, smi2), max(smi1, smi2))
        if key in seen_neg:
            continue
        if are_protonation_variants(smi1, smi2):
            continue
        seen_neg.add(key)
        negative_pairs.append({
            "smiles_1": smi1,
            "smiles_2": smi2,
            "molecule_id_1": cid1,
            "molecule_id_2": cid2,
            "ground_truth": "No",
            "pair_type": "negative_unrelated",
            "ionizable_group": "none",
            "charge_1": 0,
            "charge_2": 0,
            "source": "ChEBI-20",
        })

    print(f"  Total negative pairs: {len(negative_pairs)}")

    # ----------------------------------------------------------------
    # Assemble and verify
    # ----------------------------------------------------------------
    all_pairs = positive_pairs + negative_pairs
    pairs_df = pd.DataFrame(all_pairs)
    pairs_df = pairs_df.reset_index(drop=True)
    pairs_df.insert(0, "pair_id", pairs_df.index)

    # Compute tanimoto where missing
    print("  Computing Tanimoto similarities for all pairs...")
    tanimotos = []
    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="    Tanimoto", leave=False):
        pre = row.get("_tanimoto")
        if pre is not None and not pd.isna(pre):
            tanimotos.append(float(pre))
        else:
            tanimotos.append(_tanimoto(row["smiles_1"], row["smiles_2"]))
    pairs_df["tanimoto_between_pair"] = tanimotos
    if "_tanimoto" in pairs_df.columns:
        pairs_df.drop(columns=["_tanimoto"], inplace=True)

    # Verify all labels
    print("  Verifying pair labels...")
    mismatches = 0
    charge_mismatches = 0
    for _, row in pairs_df.iterrows():
        is_variant = are_protonation_variants(row["smiles_1"], row["smiles_2"])
        expected = (row["ground_truth"] == "Yes")
        if is_variant != expected:
            mismatches += 1
        # For positive pairs, verify different charge
        if expected and row.get("charge_1") == row.get("charge_2"):
            charge_mismatches += 1
    print(f"    Label mismatches: {mismatches} (should be 0)")
    print(f"    Positive pairs with same charge: {charge_mismatches} (should be 0)")

    # ------------------------------------------------------------------
    # Balance positive / negative to equal counts
    # ------------------------------------------------------------------
    pos_df = pairs_df[pairs_df["ground_truth"] == "Yes"]
    neg_df = pairs_df[pairs_df["ground_truth"] == "No"]
    n_pos, n_neg = len(pos_df), len(neg_df)
    n_keep = min(n_pos, n_neg)

    if n_pos != 125 or n_neg != 125:
        print(f"\n  WARNING: collected {n_pos} positive / {n_neg} negative pairs "
              f"(targets: 125/125). Trimming both to {n_keep} for balance.")

    pos_df = pos_df.sample(n=n_keep, random_state=SEED).reset_index(drop=True)
    neg_df = neg_df.sample(n=n_keep, random_state=SEED).reset_index(drop=True)
    pairs_df = pd.concat([pos_df, neg_df], ignore_index=True)
    pairs_df["pair_id"] = range(len(pairs_df))

    # Add representation columns
    print("  Adding representation columns...")
    pairs_df["canonical_smiles_1"] = pairs_df["smiles_1"].apply(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
    )
    pairs_df["canonical_smiles_2"] = pairs_df["smiles_2"].apply(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
    )
    pairs_df = _add_representation_columns(pairs_df, chebi_df)

    print(f"\n  ProtonationPairs-250 summary:")
    print(f"    Total pairs:    {len(pairs_df)}")
    print(f"    Positive (Yes): {(pairs_df['ground_truth'] == 'Yes').sum()}")
    print(f"    Negative (No):  {(pairs_df['ground_truth'] == 'No').sum()}")
    print(f"    IUPAC coverage: {pairs_df['iupac_1'].notna().sum()}/{len(pairs_df)}")
    print(f"    Pair types:\n{pairs_df['pair_type'].value_counts().to_string()}")

    return pairs_df


# ============================================================================
# Main Processing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare datasets for the benchmark suite.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Which benchmark datasets to build. "
            "Omit to build ALL (1-7 + 9 + 10). "
            "Pass benchmark numbers to build only those, e.g. --benchmarks 9 10"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_all = args.benchmarks is None
    selected = set(args.benchmarks) if args.benchmarks else set()

    def should_run(step_nums):
        """Return True if at least one of step_nums is selected (or we're running all)."""
        if run_all:
            return True
        return bool(selected.intersection(step_nums))

    print("=" * 80)
    print("ChEBI-20 Dataset Preparation")
    print("=" * 80)

    # Create output directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Dataset routing:
    #   B1-B6  → ChEBI-20 (this script)
    #   B7     → ZINC-500  (01_prepare_zinc_dataset.py)
    #   B9-B10 → ZINC250K  (this script, steps 10b/10c)
    needs_chebi = should_run({1, 2, 3, 4, 5, 6})
    needs_zinc  = should_run({9, 10})

    # Sentinel variables so the summary never crashes on a partial run
    train_df = val_df = test_df = None
    comprehension_ids = []
    retrieval_distractors_df = pd.DataFrame()
    isomer_pairs_df = pd.DataFrame()
    fewshot_df = pd.DataFrame()
    completion_partials_df = pd.DataFrame()
    all_iupac_failures = []
    tautomer_pairs_df = pd.DataFrame()
    protonation_pairs_df = pd.DataFrame()

    if needs_chebi:
        # ====================================================================
        # 1. Load ChEBI-20 splits
        # ====================================================================
        print("\n" + "=" * 80)
        print("Step 1: Loading ChEBI-20 splits")
        print("=" * 80)

        train_df = load_chebi20_split(CHEBI20_TRAIN)
        val_df   = load_chebi20_split(CHEBI20_VAL)
        test_df  = load_chebi20_split(CHEBI20_TEST)

        train_df['split'] = 'train'
        val_df['split']   = 'val'
        test_df['split']  = 'test'

        train_df = train_df.rename(columns={'CID': 'molecule_id', 'SMILES': 'smiles'})
        val_df   = val_df.rename(columns={'CID': 'molecule_id', 'SMILES': 'smiles'})
        test_df  = test_df.rename(columns={'CID': 'molecule_id', 'SMILES': 'smiles'})

        cols = ['molecule_id', 'split', 'smiles', 'description', 'iupacname', 'SELFIES']
        train_df = train_df[cols]
        val_df   = val_df[cols]
        test_df  = test_df[cols]

        # ====================================================================
        # 2. Convert test set to all representations
        # ====================================================================
        print("\n" + "=" * 80)
        print("Step 2: Converting test molecules to all representations")
        print("=" * 80)

        test_df['SMILES'] = test_df['smiles']
        test_df['CID']    = test_df['molecule_id']

        test_df, iupac_failures_test = convert_molecules_to_representations(test_df, "test")
        # Drop iupacname — identical to the 'iupac' representation column
        test_df = test_df.drop(columns=['iupacname'], errors='ignore')
        test_df.to_csv(DATA_DIR / "chebi20_test.csv", index=False)
        print(f"\n✓ Saved intermediate: chebi20_test.csv ({len(test_df)} molecules)")

        # ====================================================================
        # 3. Convert train set to all representations (for few-shot)
        # ====================================================================
        print("\n" + "=" * 80)
        print("Step 3: Converting train molecules to all representations")
        print("=" * 80)

        train_df['SMILES'] = train_df['smiles']
        train_df['CID']    = train_df['molecule_id']

        train_df, iupac_failures_train = convert_molecules_to_representations(train_df, "train")
        # Drop iupacname — identical to the 'iupac' representation column
        train_df = train_df.drop(columns=['iupacname'], errors='ignore')
        train_df.to_csv(DATA_DIR / "chebi20_train.csv", index=False)
        print(f"\n✓ Saved intermediate: chebi20_train.csv ({len(train_df)} molecules)")

        all_iupac_failures = iupac_failures_test + iupac_failures_train

        # ====================================================================
        # 4. Compute molecular properties
        # ====================================================================
        print("\n" + "=" * 80)
        print("Step 4: Computing molecular properties")
        print("=" * 80)

        test_df = compute_molecular_properties(test_df)

        # ====================================================================
        # 5. Create stratified comprehension subset  (B1-B5)
        # ====================================================================
        if should_run({1, 2, 3, 4, 5}):
            print("\n" + "=" * 80)
            print("Step 5: Creating stratified comprehension subset")
            print("=" * 80)

            comprehension_ids = create_stratified_comprehension_subset(test_df, COMPREHENSION_SAMPLE_SIZE)
            pd.DataFrame({"molecule_id": comprehension_ids}).to_csv(
                DATA_DIR / "comprehension_subset_ids.csv", index=False
            )
            print(f"\n✓ Saved intermediate: comprehension_subset_ids.csv ({len(comprehension_ids)} IDs)")

        # ====================================================================
        # 6. Create retrieval distractors  (B4)
        # ====================================================================
        if should_run({4}):
            print("\n" + "=" * 80)
            print("Step 6: Creating retrieval distractors")
            print("=" * 80)

            retrieval_distractors_df = create_retrieval_distractors(test_df)
            retrieval_distractors_df.to_csv(DATA_DIR / "retrieval_distractors.csv", index=False)
            print(f"\n✓ Saved intermediate: retrieval_distractors.csv ({len(retrieval_distractors_df)} distractors)")

        # ====================================================================
        # 7. Create isomer discrimination pairs  (B5)
        # ====================================================================
        if should_run({5}):
            print("\n" + "=" * 80)
            print("Step 7: Creating isomer discrimination pairs")
            print("=" * 80)

            isomer_pairs_df = create_isomer_pairs(test_df)
            isomer_pairs_df.to_csv(DATA_DIR / "isomer_pairs.csv", index=False)
            print(f"\n✓ Saved intermediate: isomer_pairs.csv ({len(isomer_pairs_df)} pairs)")

        # ====================================================================
        # 8. Compute few-shot examples  (B6)
        # ====================================================================
        if should_run({6}):
            print("\n" + "=" * 80)
            print("Step 8: Computing few-shot examples")
            print("=" * 80)

            fewshot_df = compute_fewshot_examples(train_df, test_df, N_SHOT)
            fewshot_df.to_csv(DATA_DIR / "fewshot_examples.csv", index=False)
            print(f"\n✓ Saved intermediate: fewshot_examples.csv ({len(fewshot_df)} mappings)")

        # ====================================================================
        # 9. Create completion partials from ChEBI-20 (auxiliary; B7 uses ZINC)
        # ====================================================================
        if run_all:
            print("\n" + "=" * 80)
            print("Step 9: Creating completion partial strings")
            print("=" * 80)

            completion_partials_df = create_completion_partials(test_df)
            completion_partials_df.to_csv(DATA_DIR / "completion_partials.csv", index=False)
            print(f"\n✓ Saved intermediate: completion_partials.csv ({len(completion_partials_df)} partials)")

        # Save IUPAC failures whenever ChEBI was processed
        pd.DataFrame(all_iupac_failures).to_csv(DATA_DIR / "iupac_failures.csv", index=False)
        print(f"\n✓ Saved: iupac_failures.csv ({len(all_iupac_failures)} failures)")

    if needs_zinc:
        # Load raw ChEBI-20 (all splits) as molecule pool for B9/B10.
        # Using ChEBI ensures IUPAC names and SELFIES are available natively,
        # eliminating the need for PubChem lookups at dataset prep or inference time.
        print("\nLoading ChEBI-20 for B9/B10 molecule pool...")
        chebi_b9b10_df = pd.concat([
            pd.read_csv(CHEBI20_TRAIN),
            pd.read_csv(CHEBI20_TEST),
            pd.read_csv(CHEBI20_VAL),
        ], ignore_index=True).dropna(subset=["SMILES"])
        print(f"  Loaded {len(chebi_b9b10_df)} ChEBI molecules")

        # ====================================================================
        # 10b. Build TautomerPairs-250  (B9)
        # ====================================================================
        if should_run({9}):
            print("\n" + "=" * 80)
            print("Step 10b: Building TautomerPairs-250 (Benchmark 9)")
            print("=" * 80)

            tautomer_pairs_df = create_tautomer_pairs(chebi_b9b10_df)
            tautomer_pairs_df.to_csv(DATA_DIR / "tautomer_pairs250.csv", index=False)
            print(f"\n✓ Saved: tautomer_pairs250.csv ({len(tautomer_pairs_df)} pairs)")

        # ====================================================================
        # 10c. Build ProtonationPairs-250  (B10)
        # ====================================================================
        if should_run({10}):
            print("\n" + "=" * 80)
            print("Step 10c: Building ProtonationPairs-250 (Benchmark 10)")
            print("=" * 80)

            protonation_pairs_df = create_protonation_pairs(chebi_b9b10_df)
            protonation_pairs_df.to_csv(DATA_DIR / "protonation_pairs250.csv", index=False)
            print(f"\n✓ Saved: protonation_pairs250.csv ({len(protonation_pairs_df)} pairs)")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Dataset Preparation Complete")
    print("=" * 80)
    if test_df is not None:
        print(f"  ✓ chebi20_test.csv  ({len(test_df)} molecules)")
    if train_df is not None:
        print(f"  ✓ chebi20_train.csv ({len(train_df)} molecules)")
    if comprehension_ids:
        print(f"  ✓ comprehension_subset_ids.csv ({len(comprehension_ids)} IDs)")
    if len(retrieval_distractors_df):
        print(f"  ✓ retrieval_distractors.csv ({len(retrieval_distractors_df)} distractors)")
    if len(isomer_pairs_df):
        print(f"  ✓ isomer_pairs.csv ({len(isomer_pairs_df)} pairs)")
    if len(fewshot_df):
        print(f"  ✓ fewshot_examples.csv ({len(fewshot_df)} mappings)")
    if len(completion_partials_df):
        print(f"  ✓ completion_partials.csv ({len(completion_partials_df)} partials)")
    if all_iupac_failures:
        print(f"  ✓ iupac_failures.csv ({len(all_iupac_failures)} failures)")
    if len(tautomer_pairs_df):
        print(f"  ✓ tautomer_pairs250.csv ({len(tautomer_pairs_df)} pairs)")
    if len(protonation_pairs_df):
        print(f"  ✓ protonation_pairs250.csv ({len(protonation_pairs_df)} pairs)")

if __name__ == "__main__":
    main()
