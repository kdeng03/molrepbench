"""
Chemistry utility functions using RDKit.

This module provides wrappers around RDKit for common chemical operations:
- Atom counting
- Functional group detection (SMARTS matching)
- Molecular property calculation
- Molecular fingerprints and similarity
- Scaffold analysis
"""

from typing import Optional, Dict, List, Tuple
import logging

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors
    from rdkit.Chem import DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import RDLogger

    # Suppress RDKit warnings
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    raise ImportError("RDKit is required. Install with: pip install rdkit")


# ============================================================================
# Atom Counting
# ============================================================================

def count_atoms_by_element(mol: Chem.Mol, element: str) -> int:
    """
    Count the number of atoms of a specific element in a molecule.

    Args:
        mol: RDKit Mol object
        element: Element symbol (e.g., "C", "N", "O")

    Returns:
        Number of atoms of that element
    """
    if mol is None:
        return 0

    count = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == element:
            count += 1

    return count


def count_heavy_atoms(mol: Chem.Mol) -> int:
    """
    Count the number of heavy (non-hydrogen) atoms in a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        Number of heavy atoms
    """
    if mol is None:
        return 0

    return mol.GetNumHeavyAtoms()


def get_atom_counts(mol: Chem.Mol) -> Dict[str, int]:
    """
    Get counts of all elements in a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        Dictionary mapping element symbols to counts
    """
    if mol is None:
        return {}

    counts = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1

    return counts


# ============================================================================
# Functional Group Detection
# ============================================================================

def has_substructure(mol: Chem.Mol, smarts: str) -> bool:
    """
    Check if a molecule contains a substructure defined by a SMARTS pattern.

    Args:
        mol: RDKit Mol object
        smarts: SMARTS pattern string

    Returns:
        True if substructure is present, False otherwise
    """
    if mol is None:
        return False

    try:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            logging.error(f"Invalid SMARTS pattern: {smarts}")
            return False

        return mol.HasSubstructMatch(pattern)

    except Exception as e:
        logging.error(f"Error matching SMARTS pattern '{smarts}': {e}")
        return False


def count_substructure_matches(mol: Chem.Mol, smarts: str) -> int:
    """
    Count the number of times a substructure appears in a molecule.

    Args:
        mol: RDKit Mol object
        smarts: SMARTS pattern string

    Returns:
        Number of matches
    """
    if mol is None:
        return 0

    try:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            logging.error(f"Invalid SMARTS pattern: {smarts}")
            return 0

        matches = mol.GetSubstructMatches(pattern)
        return len(matches)

    except Exception as e:
        logging.error(f"Error counting SMARTS matches '{smarts}': {e}")
        return 0


def check_functional_groups(mol: Chem.Mol, functional_groups: Dict[str, str]) -> Dict[str, bool]:
    """
    Check for presence of multiple functional groups.

    Args:
        mol: RDKit Mol object
        functional_groups: Dictionary mapping group names to SMARTS patterns

    Returns:
        Dictionary mapping group names to presence (bool)
    """
    results = {}

    for name, smarts in functional_groups.items():
        results[name] = has_substructure(mol, smarts)

    return results


# ============================================================================
# Molecular Properties
# ============================================================================

def calculate_logp(mol: Chem.Mol) -> Optional[float]:
    """
    Calculate LogP (Wildman-Crippen partition coefficient).

    Args:
        mol: RDKit Mol object

    Returns:
        LogP value, or None if calculation fails
    """
    if mol is None:
        return None

    try:
        return Crippen.MolLogP(mol)
    except Exception as e:
        logging.error(f"Failed to calculate LogP: {e}")
        return None


def calculate_tpsa(mol: Chem.Mol) -> Optional[float]:
    """
    Calculate Topological Polar Surface Area.

    Args:
        mol: RDKit Mol object

    Returns:
        TPSA value, or None if calculation fails
    """
    if mol is None:
        return None

    try:
        return Descriptors.TPSA(mol)
    except Exception as e:
        logging.error(f"Failed to calculate TPSA: {e}")
        return None


def count_hbd(mol: Chem.Mol) -> Optional[int]:
    """
    Count hydrogen bond donors.

    Args:
        mol: RDKit Mol object

    Returns:
        Number of H-bond donors, or None if calculation fails
    """
    if mol is None:
        return None

    try:
        return Lipinski.NumHDonors(mol)
    except Exception as e:
        logging.error(f"Failed to count H-bond donors: {e}")
        return None


def count_hba(mol: Chem.Mol) -> Optional[int]:
    """
    Count hydrogen bond acceptors.

    Args:
        mol: RDKit Mol object

    Returns:
        Number of H-bond acceptors, or None if calculation fails
    """
    if mol is None:
        return None

    try:
        return Lipinski.NumHAcceptors(mol)
    except Exception as e:
        logging.error(f"Failed to count H-bond acceptors: {e}")
        return None


def calculate_molecular_weight(mol: Chem.Mol) -> Optional[float]:
    """
    Calculate molecular weight.

    Args:
        mol: RDKit Mol object

    Returns:
        Molecular weight, or None if calculation fails
    """
    if mol is None:
        return None

    try:
        return Descriptors.MolWt(mol)
    except Exception as e:
        logging.error(f"Failed to calculate molecular weight: {e}")
        return None


def calculate_all_properties(mol: Chem.Mol) -> Dict[str, Optional[float]]:
    """
    Calculate all common molecular properties.

    Args:
        mol: RDKit Mol object

    Returns:
        Dictionary of property names to values
    """
    return {
        "logp": calculate_logp(mol),
        "tpsa": calculate_tpsa(mol),
        "hbd": count_hbd(mol),
        "hba": count_hba(mol),
        "molecular_weight": calculate_molecular_weight(mol),
        "heavy_atoms": count_heavy_atoms(mol),
    }


# ============================================================================
# Ring Analysis
# ============================================================================

def count_rings(mol: Chem.Mol) -> int:
    """
    Count the number of rings in a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        Number of rings
    """
    if mol is None:
        return 0

    try:
        # GetSSSR returns the ring info, use rdMolDescriptors for the count
        return rdMolDescriptors.CalcNumRings(mol)
    except Exception as e:
        logging.error(f"Failed to count rings: {e}")
        return 0


def count_aromatic_rings(mol: Chem.Mol) -> int:
    """
    Count the number of aromatic rings in a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        Number of aromatic rings
    """
    if mol is None:
        return 0

    try:
        return Descriptors.NumAromaticRings(mol)
    except Exception as e:
        logging.error(f"Failed to count aromatic rings: {e}")
        return 0


def has_stereocenters(mol: Chem.Mol) -> bool:
    """
    Check if a molecule has any stereocenters.

    Args:
        mol: RDKit Mol object

    Returns:
        True if stereocenters present, False otherwise
    """
    if mol is None:
        return False

    try:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        for atom in mol.GetAtoms():
            if atom.HasProp('_CIPCode'):
                return True
        return False
    except Exception as e:
        logging.error(f"Failed to check stereocenters: {e}")
        return False


# ============================================================================
# Molecular Fingerprints and Similarity
# ============================================================================

def compute_morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> Optional[DataStructs.ExplicitBitVect]:
    """
    Compute Morgan (circular) fingerprint.

    Args:
        mol: RDKit Mol object
        radius: Radius of circular fingerprint (default: 2)
        n_bits: Number of bits in fingerprint (default: 2048)

    Returns:
        Morgan fingerprint bit vector, or None if computation fails
    """
    if mol is None:
        return None

    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception as e:
        logging.error(f"Failed to compute Morgan fingerprint: {e}")
        return None


def compute_maccs_keys(mol: Chem.Mol) -> Optional[DataStructs.ExplicitBitVect]:
    """
    Compute MACCS keys fingerprint.

    Args:
        mol: RDKit Mol object

    Returns:
        MACCS keys bit vector, or None if computation fails
    """
    if mol is None:
        return None

    try:
        return AllChem.GetMACCSKeysFingerprint(mol)
    except Exception as e:
        logging.error(f"Failed to compute MACCS keys: {e}")
        return None


def calculate_tanimoto_similarity(mol1: Chem.Mol, mol2: Chem.Mol, fp_type: str = "morgan") -> Optional[float]:
    """
    Calculate Tanimoto similarity between two molecules.

    Args:
        mol1: First RDKit Mol object
        mol2: Second RDKit Mol object
        fp_type: Fingerprint type ("morgan" or "maccs")

    Returns:
        Tanimoto similarity (0-1), or None if calculation fails
    """
    if mol1 is None or mol2 is None:
        return None

    try:
        if fp_type == "morgan":
            fp1 = compute_morgan_fingerprint(mol1)
            fp2 = compute_morgan_fingerprint(mol2)
        elif fp_type == "maccs":
            fp1 = compute_maccs_keys(mol1)
            fp2 = compute_maccs_keys(mol2)
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")

        if fp1 is None or fp2 is None:
            return None

        return DataStructs.TanimotoSimilarity(fp1, fp2)

    except Exception as e:
        logging.error(f"Failed to calculate Tanimoto similarity: {e}")
        return None


# ============================================================================
# Scaffold Analysis
# ============================================================================

def get_murcko_scaffold(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Get the Murcko scaffold of a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        Scaffold as RDKit Mol object, or None if extraction fails
    """
    if mol is None:
        return None

    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return scaffold
    except Exception as e:
        logging.error(f"Failed to extract Murcko scaffold: {e}")
        return None


def molecules_have_same_scaffold(mol1: Chem.Mol, mol2: Chem.Mol) -> bool:
    """
    Check if two molecules have the same Murcko scaffold.

    Args:
        mol1: First RDKit Mol object
        mol2: Second RDKit Mol object

    Returns:
        True if scaffolds match, False otherwise
    """
    scaffold1 = get_murcko_scaffold(mol1)
    scaffold2 = get_murcko_scaffold(mol2)

    if scaffold1 is None or scaffold2 is None:
        return False

    # Compare canonical SMILES of scaffolds
    smiles1 = Chem.MolToSmiles(scaffold1, canonical=True)
    smiles2 = Chem.MolToSmiles(scaffold2, canonical=True)

    return smiles1 == smiles2


# ============================================================================
# Molecular Complexity Metrics
# ============================================================================

def categorize_by_size(mol: Chem.Mol) -> str:
    """
    Categorize a molecule by size (molecular weight).

    Args:
        mol: RDKit Mol object

    Returns:
        Size category: "small", "medium", or "large"
    """
    mw = calculate_molecular_weight(mol)

    if mw is None:
        return "unknown"

    if mw < 300:
        return "small"
    elif mw < 500:
        return "medium"
    else:
        return "large"


def categorize_by_rings(mol: Chem.Mol) -> str:
    """
    Categorize a molecule by number of rings.

    Args:
        mol: RDKit Mol object

    Returns:
        Ring category: "acyclic", "monocyclic", "bicyclic", or "polycyclic"
    """
    n_rings = count_rings(mol)

    if n_rings == 0:
        return "acyclic"
    elif n_rings == 1:
        return "monocyclic"
    elif n_rings == 2:
        return "bicyclic"
    else:
        return "polycyclic"


def categorize_by_heavy_atoms(mol: Chem.Mol) -> str:
    """
    Categorize a molecule by number of heavy atoms.

    Args:
        mol: RDKit Mol object

    Returns:
        Size category: "very_small", "small", "medium", or "large"
    """
    n_heavy = count_heavy_atoms(mol)

    if n_heavy < 10:
        return "very_small"
    elif n_heavy < 20:
        return "small"
    elif n_heavy < 35:
        return "medium"
    else:
        return "large"


# ============================================================================
# Validation and Canonicalization
# ============================================================================

def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Convert a SMILES string to canonical form.

    Args:
        smiles: Input SMILES string

    Returns:
        Canonical SMILES, or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception as e:
        logging.error(f"Failed to canonicalize SMILES '{smiles}': {e}")
        return None


def are_same_molecule(smiles1: str, smiles2: str) -> bool:
    """
    Check if two SMILES strings represent the same molecule.

    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string

    Returns:
        True if same molecule, False otherwise
    """
    canon1 = canonicalize_smiles(smiles1)
    canon2 = canonicalize_smiles(smiles2)

    if canon1 is None or canon2 is None:
        return False

    return canon1 == canon2


# ============================================================================
# Testing
# ============================================================================

def test_chemistry_functions():
    """Test chemistry functions with example molecules."""
    print("Testing Chemistry Functions\n" + "=" * 50)

    # Test molecule: Ibuprofen
    ibuprofen = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
    mol = Chem.MolFromSmiles(ibuprofen)

    print(f"\nTest molecule: Ibuprofen")
    print(f"SMILES: {ibuprofen}\n")

    # Atom counting
    print("1. Atom Counts:")
    for element in ["C", "H", "O"]:
        count = count_atoms_by_element(mol, element)
        print(f"   {element}: {count}")

    # Properties
    print("\n2. Molecular Properties:")
    props = calculate_all_properties(mol)
    for name, value in props.items():
        print(f"   {name}: {value}")

    # Ring analysis
    print("\n3. Ring Analysis:")
    print(f"   Total rings: {count_rings(mol)}")
    print(f"   Aromatic rings: {count_aromatic_rings(mol)}")
    print(f"   Has stereocenters: {has_stereocenters(mol)}")

    # Functional groups
    from config import FUNCTIONAL_GROUPS
    print("\n4. Functional Groups:")
    fg_smarts = {name: config["smarts"] for name, config in FUNCTIONAL_GROUPS.items()}
    results = check_functional_groups(mol, fg_smarts)
    for name, present in results.items():
        if present:
            print(f"   ✓ {name}")

    # Categorization
    print("\n5. Categorization:")
    print(f"   Size: {categorize_by_size(mol)}")
    print(f"   Rings: {categorize_by_rings(mol)}")
    print(f"   Heavy atoms: {categorize_by_heavy_atoms(mol)}")

    # Fingerprints
    print("\n6. Fingerprints:")
    ethanol = "CCO"
    mol2 = Chem.MolFromSmiles(ethanol)
    tanimoto = calculate_tanimoto_similarity(mol, mol2, "morgan")
    print(f"   Tanimoto similarity (Ibuprofen vs Ethanol): {tanimoto:.3f}")

    # Canonicalization
    print("\n7. Canonicalization:")
    random_smiles = "c1ccc(cc1)C(C)C(=O)O"  # Different ordering
    canon1 = canonicalize_smiles(ibuprofen)
    canon2 = canonicalize_smiles(random_smiles)
    print(f"   Original: {ibuprofen}")
    print(f"   Canonical: {canon1}")
    print(f"   Are same: {are_same_molecule(ibuprofen, random_smiles)}")

    print("\n✓ Chemistry tests completed!")


if __name__ == "__main__":
    test_chemistry_functions()
