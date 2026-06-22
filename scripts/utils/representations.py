"""
Molecular representation conversion utilities.

This module provides functions to convert SMILES strings to various molecular
representations: canonical SMILES, isomeric SMILES, randomized SMILES,
DeepSMILES, IUPAC names, SELFIES, and MolJSON.
"""

from typing import Optional, Tuple
import logging
import json
import sys
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError:
    raise ImportError("RDKit is required. Install with: pip install rdkit")

try:
    import deepsmiles
except ImportError:
    deepsmiles = None
    logging.warning("deepsmiles not installed. Install with: pip install deepsmiles")

try:
    import selfies as sf
except ImportError:
    sf = None
    logging.warning("selfies not installed. Install with: pip install selfies")

try:
    from openbabel import openbabel
    from openbabel import pybel as _pybel
    _openbabel_available = True
    # Initialize OpenBabel converter for IUPAC
    _ob_conversion = openbabel.OBConversion()
    _ob_conversion.SetInAndOutFormats("smi", "iupac")
except ImportError:
    _openbabel_available = False
    _pybel = None
    logging.warning("openbabel not installed. Install with: pip install openbabel-wheel")

# Try to import STOUT for IUPAC conversion (preferred over OpenBabel)
try:
    from STOUT import translate_forward
    _stout_available = True
except ImportError:
    _stout_available = False
    logging.warning("STOUT not installed. Install with: pip install STOUT-pypi")

# OPSIN support for IUPAC->SMILES conversion via REST API
import requests
import threading
_opsin_api_url = "https://opsin.ch.cam.ac.uk/opsin/"

# Persistent disk cache for IUPAC->SMILES lookups
_iupac_cache_path = Path(__file__).parent.parent.parent / "results" / "cache" / "iupac_cache.json"
_iupac_cache: dict = {}
_iupac_cache_lock = threading.Lock()
_iupac_cache_dirty = False

def _load_iupac_cache():
    global _iupac_cache
    if _iupac_cache_path.exists():
        try:
            with open(_iupac_cache_path) as f:
                _iupac_cache = json.load(f)
        except Exception:
            _iupac_cache = {}

def _save_iupac_cache():
    global _iupac_cache_dirty
    _iupac_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(_iupac_cache_path, "w") as f:
        json.dump(_iupac_cache, f)
    _iupac_cache_dirty = False

_load_iupac_cache()

# Import MolJSON conversion functions
try:
    # Add moljson directory to path
    _moljson_dir = Path(__file__).parent.parent.parent / "moljson"
    if str(_moljson_dir) not in sys.path:
        sys.path.insert(0, str(_moljson_dir))
    from conversion import MolToJSON, MolFromJSON
    _moljson_available = True
except ImportError:
    _moljson_available = False
    MolToJSON = None
    MolFromJSON = None
    logging.warning("MolJSON conversion not available. Check moljson directory.")

# Initialize DeepSMILES converter (if available)
if deepsmiles is not None:
    _deepsmiles_converter = deepsmiles.Converter(rings=True, branches=True)
else:
    _deepsmiles_converter = None


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """
    Convert a SMILES string to an RDKit molecule object.

    Args:
        smiles: SMILES string

    Returns:
        RDKit Mol object, or None if parsing fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception as e:
        logging.error(f"Failed to parse SMILES '{smiles}': {e}")
        return None


def mol_to_canonical_smiles(mol: Chem.Mol) -> Optional[str]:
    """
    Convert RDKit molecule to canonical SMILES.

    Args:
        mol: RDKit Mol object

    Returns:
        Canonical SMILES string, or None if conversion fails
    """
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception as e:
        logging.error(f"Failed to convert to canonical SMILES: {e}")
        return None


def mol_to_isomeric_smiles(mol: Chem.Mol) -> Optional[str]:
    """
    Convert RDKit molecule to isomeric SMILES (includes stereochemistry).

    Args:
        mol: RDKit Mol object

    Returns:
        Isomeric SMILES string, or None if conversion fails
    """
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception as e:
        logging.error(f"Failed to convert to isomeric SMILES: {e}")
        return None


def mol_to_randomized_smiles(mol: Chem.Mol) -> Optional[str]:
    """
    Convert RDKit molecule to randomized SMILES (non-canonical ordering).

    NOTE: This generates a FRESH random SMILES on each call. Do not cache
    the result if you want different random orderings across multiple calls.

    Args:
        mol: RDKit Mol object

    Returns:
        Randomized SMILES string, or None if conversion fails
    """
    try:
        return Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
    except Exception as e:
        logging.error(f"Failed to convert to randomized SMILES: {e}")
        return None


def smiles_to_deepsmiles(smiles: str) -> Optional[str]:
    """
    Convert SMILES to DeepSMILES.

    DeepSMILES removes explicit ring closure digits and parentheses,
    making it easier for sequence models to learn.

    Args:
        smiles: SMILES string

    Returns:
        DeepSMILES string, or None if conversion fails
    """
    if _deepsmiles_converter is None:
        logging.error("DeepSMILES converter not available")
        return None

    try:
        return _deepsmiles_converter.encode(smiles)
    except Exception as e:
        logging.error(f"Failed to convert SMILES to DeepSMILES '{smiles}': {e}")
        return None


def deepsmiles_to_smiles(deepsmiles_str: str) -> Optional[str]:
    """
    Convert DeepSMILES back to SMILES.

    Args:
        deepsmiles_str: DeepSMILES string

    Returns:
        SMILES string, or None if conversion fails
    """
    if _deepsmiles_converter is None:
        logging.error("DeepSMILES converter not available")
        return None

    try:
        return _deepsmiles_converter.decode(deepsmiles_str)
    except Exception as e:
        logging.error(f"Failed to decode DeepSMILES '{deepsmiles_str}': {e}")
        return None


def smiles_to_selfies(smiles: str) -> Optional[str]:
    """
    Convert SMILES to SELFIES.

    SELFIES (Self-Referencing Embedded Strings) is a 100% robust molecular
    string representation - every SELFIES string corresponds to a valid molecule.

    Args:
        smiles: SMILES string

    Returns:
        SELFIES string, or None if conversion fails
    """
    if sf is None:
        logging.error("SELFIES library not available")
        return None

    try:
        return sf.encoder(smiles)
    except Exception as e:
        logging.error(f"Failed to convert SMILES to SELFIES '{smiles}': {e}")
        return None


def selfies_to_smiles(selfies_str: str) -> Optional[str]:
    """
    Convert SELFIES back to SMILES.

    Args:
        selfies_str: SELFIES string

    Returns:
        SMILES string, or None if conversion fails
    """
    if sf is None:
        logging.error("SELFIES library not available")
        return None

    try:
        return sf.decoder(selfies_str)
    except Exception as e:
        logging.error(f"Failed to decode SELFIES '{selfies_str}': {e}")
        return None


def smiles_to_iupac(smiles: str) -> Optional[str]:
    """
    Convert SMILES to IUPAC name using PubChem PUG REST API.

    Uses PubChem's REST API to look up IUPAC names. This is more reliable than
    STOUT (model downloads broken) or OpenBabel (doesn't generate IUPAC names).

    NOTE: This requires internet connectivity. For offline use, consider pre-generating
    IUPAC names or using a local database.

    Args:
        smiles: SMILES string

    Returns:
        IUPAC name string, or None if conversion fails
    """
    import urllib.request
    import urllib.parse
    import time

    try:
        # Use POST to avoid HTTP 400 errors caused by complex SMILES with
        # stereochemistry or charges in the URL path
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/property/IUPACName/TXT"
        post_data = urllib.parse.urlencode({"smiles": smiles}).encode("utf-8")

        req = urllib.request.Request(url, data=post_data, method="POST")
        req.add_header('User-Agent', 'Mozilla/5.0 (Python urllib)')
        req.add_header('Content-Type', 'application/x-www-form-urlencoded')

        with urllib.request.urlopen(req, timeout=10) as response:
            iupac_name = response.read().decode('utf-8').strip()

            # Return None if empty or conversion failed
            if not iupac_name or iupac_name == smiles:
                return None

            return iupac_name

    except urllib.error.HTTPError as e:
        if e.code == 404:
            # Compound not found in PubChem
            logging.debug(f"PubChem: compound not found for SMILES '{smiles}'")
        else:
            logging.warning(f"PubChem HTTP error {e.code} for SMILES '{smiles}'")
        return None
    except urllib.error.URLError as e:
        logging.warning(f"PubChem connection error for SMILES '{smiles}': {e}")
        return None
    except Exception as e:
        logging.warning(f"Failed to convert SMILES to IUPAC via PubChem '{smiles}': {e}")
        return None


def iupac_to_smiles(iupac_name: str) -> Optional[str]:
    """
    Convert IUPAC name to SMILES using OPSIN REST API, with persistent disk cache.

    Results are cached in results/cache/iupac_cache.json so repeated evaluations
    do not re-hit the network.

    Args:
        iupac_name: IUPAC chemical name

    Returns:
        SMILES string, or None if conversion fails
    """
    import urllib.request
    import urllib.parse

    # Check cache first (sentinel value None means "known-bad")
    with _iupac_cache_lock:
        if iupac_name in _iupac_cache:
            return _iupac_cache[iupac_name]

    result = None
    try:
        # URL encode the IUPAC name
        encoded_name = urllib.parse.quote(iupac_name)

        # OPSIN REST API endpoint - returns SMILES
        url = f"{_opsin_api_url}{encoded_name}.smi"

        # Make request with timeout
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Python urllib)')

        with urllib.request.urlopen(req, timeout=10) as response:
            smiles = response.read().decode('utf-8').strip()

            # Return None if empty or looks like an error message
            if not smiles or len(smiles) > 500 or smiles.startswith('Error'):
                logging.debug(f"OPSIN: could not parse IUPAC name '{iupac_name}'")
            else:
                result = smiles

    except urllib.error.HTTPError as e:
        logging.debug(f"OPSIN HTTP error {e.code} for IUPAC '{iupac_name}'")
    except urllib.error.URLError as e:
        logging.warning(f"OPSIN connection error for IUPAC '{iupac_name}': {e}")
    except Exception as e:
        logging.debug(f"Failed to convert IUPAC to SMILES via OPSIN '{iupac_name}': {e}")

    # Write result (including None for failures) into cache and flush
    with _iupac_cache_lock:
        _iupac_cache[iupac_name] = result
        global _iupac_cache_dirty
        _iupac_cache_dirty = True
        # Flush every 50 new entries to avoid losing work on interruption
        if len(_iupac_cache) % 50 == 0:
            _save_iupac_cache()

    return result


def smiles_to_moljson(smiles: str) -> Optional[str]:
    """
    Convert SMILES to MolJSON (JSON-based graph representation).

    MolJSON encodes the molecular graph as a JSON object with atoms, bonds,
    charges, and aromatic nitrogen hydrogens. Note: MolJSON does NOT support
    stereochemistry.

    Args:
        smiles: SMILES string

    Returns:
        MolJSON string (compact JSON), or None if conversion fails
    """
    if not _moljson_available:
        logging.error("MolJSON library not available")
        return None

    try:
        mol = smiles_to_mol(smiles)
        if mol is None:
            return None

        # Convert to MolJSON dict with 'element' style atom IDs (C1, C2, N1, ...)
        moljson_dict = MolToJSON(mol, atom_id_style="element")

        # Serialize to compact JSON string
        return json.dumps(moljson_dict, separators=(',', ':'), sort_keys=False)
    except Exception as e:
        logging.error(f"Failed to convert SMILES to MolJSON '{smiles}': {e}")
        return None


def moljson_to_smiles(moljson_str: str) -> Optional[str]:
    """
    Convert MolJSON back to canonical SMILES.

    Args:
        moljson_str: MolJSON string (JSON format)

    Returns:
        Canonical SMILES string, or None if conversion fails
    """
    if not _moljson_available:
        logging.error("MolJSON library not available")
        return None

    try:
        # Parse JSON string to dict
        moljson_dict = json.loads(moljson_str)

        # Convert to RDKit molecule
        mol = MolFromJSON(moljson_dict)

        if mol is None:
            return None

        # Convert to canonical SMILES (no stereochemistry since MolJSON doesn't support it)
        return mol_to_canonical_smiles(mol)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse MolJSON string as JSON: {e}")
        return None
    except Exception as e:
        logging.error(f"Failed to convert MolJSON to SMILES: {e}")
        return None


def smiles_to_cml(smiles: str) -> Optional[str]:
    """
    Convert SMILES to CML (Chemical Markup Language) using OpenBabel.

    Args:
        smiles: SMILES string

    Returns:
        CML string (XML format), or None if conversion fails
    """
    if not _openbabel_available:
        logging.error("OpenBabel not available for CML conversion")
        return None

    try:
        mol = _pybel.readstring("smi", smiles)
        cml = mol.write("cml").strip()
        return cml
    except Exception as e:
        logging.error(f"Failed to convert SMILES to CML '{smiles}': {e}")
        return None


def cml_to_smiles(cml_str: str) -> Optional[str]:
    """
    Convert CML back to canonical SMILES using OpenBabel via subprocess.

    Uses subprocess isolation because OpenBabel can segfault on malformed CML.

    Args:
        cml_str: CML string (XML format)

    Returns:
        Canonical SMILES string, or None if conversion fails
    """
    import subprocess, tempfile, os

    # Pre-validate: CML must be non-trivial XML
    stripped = cml_str.strip()
    if not stripped.startswith("<") or len(stripped) < 20:
        logging.error(f"Failed to convert CML to SMILES: not valid XML: '{stripped[:80]}'")
        return None
    import re as _re
    content = _re.sub(r'<\?xml[^?]*\?>\s*', '', stripped)
    if not content or len(content) < 10:
        logging.error(f"Failed to convert CML to SMILES: XML declaration only, no molecule data")
        return None

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cml', delete=False) as f:
            f.write(cml_str)
            tmp_path = f.name
        result = subprocess.run(
            ['obabel', tmp_path, '-osmi'],
            capture_output=True, text=True, timeout=10,
        )
        os.unlink(tmp_path)
        if result.returncode != 0 or not result.stdout.strip():
            logging.error(f"Failed to convert CML to SMILES: obabel returned {result.returncode}")
            return None
        smiles = result.stdout.strip().split()[0]  # first token is SMILES
        return smiles if smiles else None
    except Exception as e:
        logging.error(f"Failed to convert CML to SMILES: {e}")
        return None


def smiles_to_inchi(smiles: str) -> Optional[str]:
    """
    Convert SMILES to InChI (International Chemical Identifier).

    Args:
        smiles: SMILES string

    Returns:
        InChI string, or None if conversion fails
    """
    try:
        mol = smiles_to_mol(smiles)
        if mol is None:
            return None
        from rdkit.Chem.inchi import MolToInchi
        inchi = MolToInchi(mol)
        return inchi
    except Exception as e:
        logging.error(f"Failed to convert SMILES to InChI '{smiles}': {e}")
        return None


def inchi_to_smiles(inchi_str: str) -> Optional[str]:
    """
    Convert InChI back to canonical SMILES.

    Args:
        inchi_str: InChI string

    Returns:
        Canonical SMILES string, or None if conversion fails
    """
    try:
        from rdkit.Chem.inchi import MolFromInchi
        mol = MolFromInchi(inchi_str)
        if mol is None:
            return None
        return mol_to_canonical_smiles(mol)
    except Exception as e:
        logging.error(f"Failed to convert InChI to SMILES '{inchi_str}': {e}")
        return None


def convert_to_representation(smiles: str, representation: str) -> Optional[str]:
    """
    Convert a SMILES string to the specified representation.

    Args:
        smiles: Input SMILES string
        representation: Target representation name. Must be one of:
            - "canonical_smiles"
            - "isomeric_smiles"
            - "randomized_smiles"
            - "deepsmiles"
            - "iupac"
            - "selfies"
            - "moljson"

    Returns:
        Converted representation string, or None if conversion fails
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    if representation == "canonical_smiles":
        return mol_to_canonical_smiles(mol)
    elif representation == "isomeric_smiles":
        return mol_to_isomeric_smiles(mol)
    elif representation == "randomized_smiles":
        return mol_to_randomized_smiles(mol)
    elif representation == "deepsmiles":
        canonical = mol_to_canonical_smiles(mol)
        if canonical is None:
            return None
        return smiles_to_deepsmiles(canonical)
    elif representation == "iupac":
        return smiles_to_iupac(smiles)
    elif representation == "selfies":
        return smiles_to_selfies(smiles)
    elif representation == "moljson":
        return smiles_to_moljson(smiles)
    elif representation == "cml":
        return smiles_to_cml(smiles)
    elif representation == "inchi":
        return smiles_to_inchi(smiles)
    else:
        raise ValueError(f"Unknown representation: {representation}")


def parse_representation(representation_str: str, representation: str) -> Optional[Chem.Mol]:
    """
    Parse a molecular string representation back to an RDKit Mol object.

    Args:
        representation_str: The molecular string in the given representation
        representation: The representation type (e.g., "selfies", "deepsmiles", "moljson")

    Returns:
        RDKit Mol object, or None if parsing fails
    """
    try:
        if representation in ["canonical_smiles", "isomeric_smiles", "randomized_smiles"]:
            return Chem.MolFromSmiles(representation_str)

        elif representation == "deepsmiles":
            smiles = deepsmiles_to_smiles(representation_str)
            if smiles is None:
                return None
            return Chem.MolFromSmiles(smiles)

        elif representation == "selfies":
            smiles = selfies_to_smiles(representation_str)
            if smiles is None:
                return None
            return Chem.MolFromSmiles(smiles)

        elif representation == "moljson":
            if not _moljson_available:
                logging.error("MolJSON library not available")
                return None
            try:
                moljson_dict = json.loads(representation_str)
                return MolFromJSON(moljson_dict)
            except Exception as e:
                logging.error(f"Failed to parse MolJSON: {e}")
                return None

        elif representation == "iupac":
            # IUPAC to SMILES using OPSIN REST API
            smiles = iupac_to_smiles(representation_str)
            if smiles is None:
                return None
            return Chem.MolFromSmiles(smiles)

        elif representation == "cml":
            smiles = cml_to_smiles(representation_str)
            if smiles is None:
                return None
            return Chem.MolFromSmiles(smiles)

        elif representation == "inchi":
            smiles = inchi_to_smiles(representation_str)
            if smiles is None:
                return None
            return Chem.MolFromSmiles(smiles)

        else:
            raise ValueError(f"Unknown representation: {representation}")

    except Exception as e:
        logging.error(f"Failed to parse {representation} '{representation_str}': {e}")
        return None


def is_valid_representation(representation_str: str, representation: str) -> bool:
    """
    Check if a representation string is valid (can be parsed to a molecule).

    Args:
        representation_str: The molecular string
        representation: The representation type

    Returns:
        True if valid, False otherwise
    """
    mol = parse_representation(representation_str, representation)
    return mol is not None


def get_representation_display_name(representation: str) -> str:
    """
    Get the human-readable display name for a representation.

    Args:
        representation: Internal representation name

    Returns:
        Display name
    """
    display_names = {
        "canonical_smiles": "Canonical SMILES",
        "isomeric_smiles": "Isomeric SMILES",
        "randomized_smiles": "Randomized SMILES",
        "deepsmiles": "DeepSMILES",
        "iupac": "IUPAC",
        "selfies": "SELFIES",
        "moljson": "MolJSON",
        "inchi": "InChI",
    }
    return display_names.get(representation, representation)


def compute_token_length(representation_str: str, tokenizer=None) -> int:
    """
    Compute the token length of a representation string.

    Args:
        representation_str: The molecular string
        tokenizer: Optional tokenizer (e.g., from transformers). If None,
                   uses character-level length as approximation.

    Returns:
        Number of tokens
    """
    if tokenizer is None:
        # Approximate: count non-whitespace characters
        return len(representation_str.replace(" ", ""))
    else:
        # Use actual tokenizer
        tokens = tokenizer.encode(representation_str, add_special_tokens=False)
        return len(tokens)


# ============================================================================
# Batch conversion utilities
# ============================================================================

def convert_dataset_to_all_representations(smiles_list: list) -> dict:
    """
    Convert a list of SMILES to all representations.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        Dictionary mapping representation names to lists of converted strings.
        Invalid conversions are stored as None in the lists.
    """
    from config import REPRESENTATIONS

    results = {rep: [] for rep in REPRESENTATIONS}

    for smiles in smiles_list:
        for rep in REPRESENTATIONS:
            converted = convert_to_representation(smiles, rep)
            results[rep].append(converted)

    return results


def get_conversion_success_rate(smiles_list: list, representation: str) -> Tuple[int, int, float]:
    """
    Compute the success rate of converting SMILES to a representation.

    Args:
        smiles_list: List of SMILES strings
        representation: Target representation

    Returns:
        Tuple of (successful_count, total_count, success_rate)
    """
    successful = 0
    total = len(smiles_list)

    for smiles in smiles_list:
        converted = convert_to_representation(smiles, representation)
        if converted is not None:
            successful += 1

    success_rate = successful / total if total > 0 else 0.0
    return successful, total, success_rate


if __name__ == "__main__":
    # Test conversions
    test_smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"  # Ibuprofen

    print("Testing molecular representation conversions")
    print(f"Input SMILES: {test_smiles}\n")

    representations_to_test = [
        "canonical_smiles",
        "isomeric_smiles",
        "randomized_smiles",
        "deepsmiles",
        "selfies",
        "iupac",
        "moljson",
        "inchi",
    ]

    for rep in representations_to_test:
        converted = convert_to_representation(test_smiles, rep)
        status = "✓" if converted is not None else "✗"
        print(f"{status} {rep:20s}: {converted}")

    # Test randomized SMILES generates different outputs
    print("\nTesting randomized SMILES (3 samples):")
    mol = smiles_to_mol(test_smiles)
    for i in range(3):
        random_smiles = mol_to_randomized_smiles(mol)
        print(f"  {i+1}. {random_smiles}")
