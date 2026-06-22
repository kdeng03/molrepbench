# MolJSON Representation

## Overview

MolJSON is a JSON-based explicit graph representation for molecules that encodes:
- **Atoms**: List of atom objects with unique IDs and element symbols
- **Bonds**: List of bonds with source/target atom IDs and bond orders
- **Charges**: Sparse list of non-zero formal charges
- **Aromatic N-H**: Special handling for aromatic nitrogens with explicit hydrogen counts

## Key Characteristics

### Advantages
- **Explicit structure**: Graph structure is directly readable (unlike SMILES)
- **Human-readable**: JSON format is easy to parse and understand
- **Structured data**: Tests LLM's ability to work with JSON representations
- **No syntax errors**: Well-formed JSON is always parseable

### Limitations
- **No stereochemistry support**: MolJSON does NOT encode stereoisomers
- **Verbose**: Much longer than SMILES (typically 10-20x more characters)
- **No hydrogen atoms**: Explicit hydrogens are removed during conversion

## Example

**Input SMILES**: `CC(C)Cc1ccc(C(C)C(=O)O)cc1` (Ibuprofen)

**MolJSON output** (formatted for readability):
```json
{
  "atoms": [
    {"id": "C1", "element": "C"},
    {"id": "C2", "element": "C"},
    ...
  ],
  "bonds": [
    {"source": "C1", "target": "C2", "order": 1.0},
    {"source": "C5", "target": "C6", "order": 1.5},
    ...
  ]
}
```

Note: Bond order 1.5 indicates aromatic bonds.

## API

### Python API

```python
from conversion import MolToJSON, MolFromJSON
from rdkit import Chem

# SMILES -> MolJSON
mol = Chem.MolFromSmiles("CCO")
moljson_dict = MolToJSON(mol, atom_id_style="element")  # or "a" for a1, a2, a3...

# MolJSON -> RDKit Mol
mol_reconstructed = MolFromJSON(moljson_dict)

# Round-trip check
from conversion import CheckRoundTrip
ok, input_smi, output_smi, moljson = CheckRoundTrip(mol)
```

### Integration with Benchmark

The representation is automatically available in the benchmark suite:

```python
from scripts.utils.representations import convert_to_representation

# Convert SMILES to MolJSON string
moljson_str = convert_to_representation("CCO", "moljson")
# Returns: '{"atoms":[{"id":"C1","element":"C"},{"id":"C2","element":"C"},...'
```

## Files

- `schema.py`: JSON schema definition for MolJSON
- `conversion.py`: Conversion functions (MolToJSON, MolFromJSON)
- `README.md`: This documentation

## Notes for Benchmark Analysis

When analyzing MolJSON results, keep in mind:

1. **Stereochemistry tasks**: MolJSON will perform poorly on tasks requiring stereochemistry (isomer discrimination with stereoisomers)
2. **Token length**: MolJSON representations are much longer than SMILES, which may affect LLM performance due to context limits
3. **Generation task**: Models must generate valid JSON with correct atom IDs referenced in bonds
4. **Validity checks**: Use JSON parsing + MolFromJSON to validate generated MolJSON strings
