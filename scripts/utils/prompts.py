"""
Prompt templates for all benchmarks.

This module provides functions to generate prompts for each of the 7 benchmarks,
with support for different molecular representations and thinking modes.
"""

from typing import List, Dict, Optional
import sys
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import REPR_DISPLAY_NAMES


def get_representation_name_for_prompt(representation: str) -> str:
    """
    Get the display name for a representation to use in prompts.

    Args:
        representation: Internal representation name

    Returns:
        Human-readable representation name
    """
    return REPR_DISPLAY_NAMES.get(representation, representation)


# ============================================================================
# Benchmark 1: Atom Counting
# ============================================================================

def create_atom_counting_prompt(
    molecule_str: str,
    element: str,
    representation: str,
) -> str:
    """
    Create prompt for atom counting task.

    Args:
        molecule_str: Molecular representation string
        element: Element symbol to count (e.g., "C", "N", "O")
        representation: Representation type

    Returns:
        Formatted prompt string
    """
    return f"""How many {element} atoms are in the following molecule?
Molecule: {molecule_str}

Output ONLY the answer in the format: \\boxed{{your_count}}. Do not output any additional text or explanation."""


# ============================================================================
# Benchmark 2: Functional Group Identification
# ============================================================================

def create_functional_group_prompt(
    molecule_str: str,
    functional_group_name: str,
    representation: str,
) -> str:
    """
    Create prompt for functional group identification task.

    Args:
        molecule_str: Molecular representation string
        functional_group_name: Name of functional group (e.g., "hydroxyl")
        representation: Representation type

    Returns:
        Formatted prompt string
    """
    return f"""Does the following molecule contain a {functional_group_name}?
Molecule: {molecule_str}

Output ONLY the answer in the format: \\boxed{{Yes}} or \\boxed{{No}}. Do not output any additional text or explanation."""


# ============================================================================
# Benchmark 3: Molecular Property Estimation
# ============================================================================

def create_property_estimation_prompt(
    molecule_str: str,
    property_name: str,
    representation: str,
) -> str:
    """
    Create prompt for molecular property estimation task.

    Args:
        molecule_str: Molecular representation string
        property_name: Property to estimate (e.g., "LogP", "TPSA")
        representation: Representation type

    Returns:
        Formatted prompt string
    """
    # Format property name for display
    property_display = {
        "logp": "LogP (partition coefficient)",
        "tpsa": "Topological Polar Surface Area (TPSA)",
        "hbd": "number of hydrogen bond donors",
        "hba": "number of hydrogen bond acceptors",
    }.get(property_name.lower(), property_name)

    return f"""Estimate the {property_display} of the following molecule.
Molecule: {molecule_str}

Output ONLY the answer in the format: \\boxed{{your_estimation}}. Do not output any additional text or explanation."""


# ============================================================================
# Benchmark 4: Molecule Retrieval / Discrimination
# ============================================================================

def create_retrieval_prompt(
    description: str,
    molecules: Dict[str, str],
    representation: str,
) -> str:
    """
    Create prompt for molecule retrieval task.

    Args:
        description: Natural language description of the molecule
        molecules: Dictionary mapping letters (A, B, C, D) to molecule strings
        representation: Representation type

    Returns:
        Formatted prompt string
    """
    choices = "\n".join([f"{letter}: {mol_str}" for letter, mol_str in sorted(molecules.items())])

    return f"""Which of the following molecules matches this description?
Description: {description}
{choices}

Output ONLY the answer in the format: \\boxed{{your_choice}}. Do not output any additional text or explanation."""


# ============================================================================
# Benchmark 5: Isomer / Identity Discrimination
# ============================================================================

def create_isomer_discrimination_prompt(
    molecule_1: str,
    molecule_2: str,
    representation: str,
) -> str:
    """
    Create prompt for isomer discrimination task.

    Args:
        molecule_1: First molecular representation string
        molecule_2: Second molecular representation string
        representation: Representation type

    Returns:
        Formatted prompt string
    """
    return f"""Do the following two molecular representations refer to the same molecule?
Molecule 1: {molecule_1}
Molecule 2: {molecule_2}

Output ONLY the answer in the format: \\boxed{{Yes}} or \\boxed{{No}}. Do not output any additional text or explanation."""


# ============================================================================
# Benchmark 9: Tautomer Recognition
# ============================================================================

def create_tautomer_recognition_prompt(
    molecule_1: str,
    molecule_2: str,
    representation: str,
) -> str:
    """
    Create prompt for tautomer recognition task.

    Args:
        molecule_1: First molecular representation string
        molecule_2: Second molecular representation string
        representation: Representation type

    Returns:
        Formatted prompt string
    """
    return f"""Are the following two molecules tautomeric forms of the same compound?
Molecule 1: {molecule_1}
Molecule 2: {molecule_2}

Output ONLY the answer in the format: \\boxed{{Yes}} or \\boxed{{No}}. Do not output any additional text or explanation."""


# ============================================================================
# Benchmark 10: Protonation State Recognition
# ============================================================================

def create_protonation_recognition_prompt(
    molecule_1: str,
    molecule_2: str,
    representation: str,
) -> str:
    """
    Create prompt for protonation state recognition task.

    Args:
        molecule_1: First molecular representation string
        molecule_2: Second molecular representation string
        representation: Representation type

    Returns:
        Formatted prompt string
    """
    return f"""Are the following two molecules different protonation states of the same compound?
Molecule 1: {molecule_1}
Molecule 2: {molecule_2}

Output ONLY the answer in the format: \\boxed{{Yes}} or \\boxed{{No}}. Do not output any additional text or explanation."""


# ============================================================================
# Benchmark 6: Caption-to-Molecule Generation (ChEBI-20)
# ============================================================================

def create_generation_prompt(
    description: str,
    representation: str,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
    use_json_format: bool = False,
) -> str:
    """
    Create prompt for molecule generation task with optional few-shot examples.

    Args:
        description: Natural language description of the target molecule
        representation: Target representation type
        few_shot_examples: List of dicts with keys "description" and "molecule"
                          (in the target representation). If None, zero-shot.
        use_json_format: If True, instruct model to output JSON format

    Returns:
        Formatted prompt string
    """
    rep_name = get_representation_name_for_prompt(representation)

    # Base instruction with token efficiency guidance
    instruction = f"""Given a description of a molecule, generate the corresponding {rep_name} string.

IMPORTANT: Don't restate the full description - get straight to generating the molecule. Keep any reasoning brief and concise. Output tokens are limited, so provide your final answer as quickly as possible."""

    # JSON format instruction
    if use_json_format:
        if representation == "moljson":
            format_instruction = '''Your final output must be ONLY a valid JSON object with "atoms" and "bonds" arrays.

Format:
{"atoms": [{"id": "C1", "element": "C"}, ...], "bonds": [{"source": "C1", "target": "C2", "order": 1.0}, ...], "charges": null, "aromatic_n_h": null}

Keep the reasoning concise and short. Don't ramble, be decisive. As soon as possible, output the JSON object.'''
        else:
            format_instruction = f'''Your final output must be a JSON object in this format:
{{"molecule": "..."}}

Where the value is the complete {rep_name} string. Keep the reasoning concise and short. Don't ramble, be decisive. As soon as possible, output the JSON object.'''
    else:
        format_instruction = "Provide your final answer in the format: Final answer: <molecule_string>"

    # Add few-shot examples if provided
    if few_shot_examples:
        examples = []
        for example in few_shot_examples:
            examples.append(f"""Description: {example['description']}
Molecule: {example['molecule']}""")
        examples_str = "\n\n".join(examples)
        prompt = f"{instruction}\n\n{examples_str}\n\nDescription: {description}\n\n{format_instruction}"
    else:
        # Zero-shot
        prompt = f"{instruction}\n\nDescription: {description}\n\n{format_instruction}"

    return prompt


# ============================================================================
# Benchmark 7: Molecular Completion / Infilling
# ============================================================================

def create_completion_prompt(
    partial_molecule: str,
    representation: str,
    use_json_format: bool = False,
) -> str:
    """
    Create prompt for molecule completion task.

    Args:
        partial_molecule: First 50% of the molecule string
        representation: Representation type
        use_json_format: If True, instruct model to output JSON format

    Returns:
        Formatted prompt string
    """
    rep_name = get_representation_name_for_prompt(representation)

    # JSON format instruction
    if use_json_format:
        if representation == "moljson":
            format_instruction = '''Your final output must be ONLY a valid JSON object with "atoms" and "bonds" arrays representing the COMPLETE molecular graph.

Format:
{"atoms": [{"id": "C1", "element": "C"}, ...], "bonds": [{"source": "C1", "target": "C2", "order": 1.0}, ...], "charges": null, "aromatic_n_h": null}

Keep the reasoning concise and short. Don't ramble, be decisive. As soon as possible, output the complete JSON object.'''
        else:
            format_instruction = f'''Your final output must be a JSON object in this format:
{{"molecule": "..."}}

Where the value is the COMPLETE {rep_name} string (not just the continuation). Keep the reasoning concise and short. Don't ramble, be decisive. As soon as possible, output the JSON object.'''
    else:
        format_instruction = "Provide your final answer in the format: Final answer: <complete_molecule_string>"

    return f"""Complete the following partial {rep_name} string to form a valid molecule.
Partial: {partial_molecule}

IMPORTANT: Get straight to completing the molecule. Keep any reasoning brief and concise. Output tokens are limited, so provide your final answer as quickly as possible.

{format_instruction}"""


# ============================================================================
# Chat Template Formatting
# ============================================================================

def format_with_chat_template(
    prompt: str,
    model_name: str,
    thinking: bool = False,
) -> str:
    """
    Wrap a prompt with the appropriate chat template for the model.

    Args:
        prompt: The task prompt
        model_name: Model identifier (e.g., "qwen3-8b", "nemotron-nano-8b")
        thinking: Whether to enable thinking mode

    Returns:
        Formatted prompt with chat template
    """
    from config import MODELS

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    model_config = MODELS[model_name]
    chat_template = model_config["chat_template"]

    if chat_template == "qwen":
        # Qwen template: <|im_start|>system\n...<|im_end|>
        # Thinking is controlled via model generation parameters, not system message
        system_message = "You are a helpful assistant."
        formatted = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        return formatted

    elif chat_template == "llama3.1":
        # Llama 3.1 / Nemotron template
        # Thinking is controlled via system message
        if thinking:
            system_message = "detailed thinking on"
        else:
            system_message = "detailed thinking off"

        formatted = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return formatted

    else:
        raise ValueError(f"Unknown chat template: {chat_template}")


# ============================================================================
# Stop Sequences
# ============================================================================

def get_stop_sequences(benchmark_num: int) -> List[str]:
    """
    Get appropriate stop sequences for a benchmark.

    Args:
        benchmark_num: Benchmark number (1-7)

    Returns:
        List of stop sequences
    """
    # Common stop sequences
    common = ["\n\n"]

    # Benchmark-specific stops
    if benchmark_num in [1, 2, 3]:
        # Short answer tasks
        return common + ["\n"]
    elif benchmark_num == 4:
        # Multiple choice
        return common + ["\nA:", "\nB:", "\nC:", "\nD:"]
    elif benchmark_num == 5:
        # Binary classification
        return common + ["\n"]
    elif benchmark_num == 6:
        # Generation
        return common + ["\nDescription:", "\n\n\n"]
    elif benchmark_num == 7:
        # Completion
        return common + ["\nPartial:", "\n\n\n"]
    else:
        return common


# ============================================================================
# Prompt Validation and Testing
# ============================================================================

def validate_prompt(prompt: str, benchmark_num: int) -> bool:
    """
    Validate that a prompt is well-formed.

    Args:
        prompt: The prompt string
        benchmark_num: Benchmark number

    Returns:
        True if valid, False otherwise
    """
    # Check that prompt is non-empty
    if not prompt or not prompt.strip():
        return False

    # Check for required components based on benchmark
    required_keywords = {
        1: ["atoms", "molecule"],
        2: ["molecule", "Yes", "No"],
        3: ["Estimate", "molecule"],
        4: ["Description", "A:", "B:", "C:", "D:"],
        5: ["same molecule", "Yes", "No"],
        6: ["Description", "Molecule"],
        7: ["Complete", "Partial"],
    }

    keywords = required_keywords.get(benchmark_num, [])
    for keyword in keywords:
        if keyword not in prompt:
            return False

    return True


def test_all_prompts():
    """Test all prompt generation functions with example inputs."""
    print("Testing Prompt Generation\n" + "=" * 50)

    # Test Benchmark 1: Atom Counting
    print("\n1. Atom Counting:")
    prompt = create_atom_counting_prompt("CCO", "C", "canonical_smiles")
    print(prompt)
    assert validate_prompt(prompt, 1)

    # Test Benchmark 2: Functional Groups
    print("\n2. Functional Group Identification:")
    prompt = create_functional_group_prompt("CCO", "hydroxyl", "canonical_smiles")
    print(prompt)
    assert validate_prompt(prompt, 2)

    # Test Benchmark 3: Property Estimation
    print("\n3. Property Estimation:")
    prompt = create_property_estimation_prompt("CCO", "logp", "canonical_smiles")
    print(prompt)
    assert validate_prompt(prompt, 3)

    # Test Benchmark 4: Retrieval
    print("\n4. Molecule Retrieval:")
    molecules = {"A": "CCO", "B": "CC", "C": "C", "D": "CCN"}
    prompt = create_retrieval_prompt("ethanol", molecules, "canonical_smiles")
    print(prompt)
    assert validate_prompt(prompt, 4)

    # Test Benchmark 5: Isomer Discrimination
    print("\n5. Isomer Discrimination:")
    prompt = create_isomer_discrimination_prompt("CCO", "OCC", "canonical_smiles")
    print(prompt)
    assert validate_prompt(prompt, 5)

    # Test Benchmark 6: Generation (zero-shot)
    print("\n6. Generation (zero-shot):")
    prompt = create_generation_prompt("ethanol", "canonical_smiles")
    print(prompt)
    assert validate_prompt(prompt, 6)

    # Test Benchmark 6: Generation (few-shot)
    print("\n6b. Generation (2-shot):")
    examples = [
        {"description": "methane", "molecule": "C"},
        {"description": "ethanol", "molecule": "CCO"},
    ]
    prompt = create_generation_prompt("propanol", "canonical_smiles", examples)
    print(prompt)
    assert validate_prompt(prompt, 6)

    # Test Benchmark 7: Completion
    print("\n7. Molecular Completion:")
    prompt = create_completion_prompt("CC", "canonical_smiles")
    print(prompt)
    assert validate_prompt(prompt, 7)

    # Test chat template formatting
    print("\n8. Chat Template (Qwen3, thinking OFF):")
    base_prompt = "How many C atoms are in CCO?"
    formatted = format_with_chat_template(base_prompt, "qwen3-8b", thinking=False)
    print(formatted[:200] + "...")

    print("\n9. Chat Template (Nemotron, thinking ON):")
    formatted = format_with_chat_template(base_prompt, "nemotron-nano-8b", thinking=True)
    print(formatted[:200] + "...")

    print("\n✓ All prompt tests passed!")


if __name__ == "__main__":
    test_all_prompts()
