"""
Output parsing utilities for LLM responses.

This module provides functions to extract and validate answers from model outputs,
including handling thinking tags, extracting numbers, and parsing categorical responses.
"""

import re
from typing import Optional, Tuple, Union
import logging


# ============================================================================
# Thinking Tag Extraction
# ============================================================================

def extract_answer_from_thinking_response(response: str) -> Tuple[str, str]:
    """
    Extract the answer from a thinking-enabled model response.

    Supports multiple formats:
    1. XML tags: <think>...</think>[answer] - CHECKED FIRST
    2. "Final answer:" marker - extract everything after this marker
    3. Qwen format: "Thinking Process:\n...\n\n[answer]"
    4. Plain text: treat entire response as answer

    Args:
        response: Full model response (may contain thinking markers)

    Returns:
        Tuple of (full_response, answer_only)
        - full_response: The complete response including thinking
        - answer_only: Just the answer after thinking, or full response if no markers
    """
    # FIRST: Try to find XML-style closing </think> tag
    # This must be checked BEFORE "Final answer:" to avoid matching inside thinking block
    match = re.search(r'</think>\s*(.*)', response, re.DOTALL)
    if match:
        # Found thinking tags - extract answer after </think>
        answer_section = match.group(1).strip()

        # Check for <answer>...</answer> tags (ChemDFM-R style)
        answer_tag_match = re.search(r'<answer>(.*?)</answer>', answer_section, re.DOTALL | re.IGNORECASE)
        if answer_tag_match:
            return response, answer_tag_match.group(1).strip()

        # Now look for "Final answer:" in the answer section (not the thinking block)
        # Use .* to capture everything after "Final answer:", not just first line
        final_match = re.search(r'Final answer:\s*(.*)', answer_section, re.IGNORECASE | re.DOTALL)
        if final_match:
            answer = final_match.group(1).strip()
            return response, answer

        # No "Final answer:" marker, use entire section after </think>
        return response, answer_section

    # SECOND: Try to find "Final answer:" marker (case-insensitive)
    # Only if no </think> tag was found
    match = re.search(r'Final answer:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
    if match:
        answer = match.group(1).strip()
        return response, answer

    # THIRD: Try to find Qwen-style "Thinking Process:" format
    # The answer is the last non-empty line
    if "Thinking Process:" in response or "Thought Process:" in response:
        lines = response.strip().split('\n')
        # Get last non-empty line - this is the final answer
        for line in reversed(lines):
            line = line.strip()
            if line:
                return response, line

    # No thinking markers found - treat entire response as answer
    return response, response.strip()


def strip_thinking_tags(response: str) -> str:
    """
    Remove all <think>...</think> tags from a response, keeping only the answer.

    Args:
        response: Model response that may contain thinking tags

    Returns:
        Response with thinking tags removed
    """
    # Remove everything between <think> and </think> including the tags
    cleaned = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
    return cleaned.strip()


# ============================================================================
# Whitespace and Formatting Cleanup
# ============================================================================

def clean_response(response: str) -> str:
    """
    Clean up common formatting issues in model responses.

    Args:
        response: Raw model output

    Returns:
        Cleaned response string
    """
    # Strip leading/trailing whitespace
    cleaned = response.strip()

    # Remove markdown code fences
    cleaned = re.sub(r'^```[a-z]*\n', '', cleaned)
    cleaned = re.sub(r'\n```$', '', cleaned)

    # Remove backticks
    cleaned = cleaned.replace('`', '')

    # Normalize whitespace (collapse multiple spaces/newlines)
    cleaned = re.sub(r'\s+', ' ', cleaned)

    return cleaned.strip()


# ============================================================================
# Integer Extraction (Benchmark 1: Atom Counting)
# ============================================================================

def extract_integer(response: str) -> Optional[int]:
    """
    Extract the first integer from a response string.

    Supports multiple formats:
    1. \\boxed{42} - LaTeX boxed format (Qwen thinking models)
    2. "Final answer: 42" - explicit format
    3. Plain integer in text

    Args:
        response: Model response

    Returns:
        Extracted integer, or None if no valid integer found
    """
    # First, clean the response
    cleaned = clean_response(response)

    # Try to extract from \boxed{} format first (highest priority)
    boxed_match = re.search(r'\\boxed\{(-?\d+)\}', cleaned)
    if boxed_match:
        try:
            return int(boxed_match.group(1))
        except ValueError:
            pass

    # Try to find after "Final answer:" marker
    final_match = re.search(r'Final answer:\s*(-?\d+)', cleaned, re.IGNORECASE)
    if final_match:
        try:
            return int(final_match.group(1))
        except ValueError:
            pass

    # Fallback: Try to find the first integer (possibly negative)
    match = re.search(r'-?\d+', cleaned)
    if match:
        try:
            return int(match.group())
        except ValueError:
            return None

    return None


# ============================================================================
# Float Extraction (Benchmark 3: Property Estimation)
# ============================================================================

def extract_float(response: str) -> Optional[float]:
    """
    Extract the first float or integer from a response string.

    Supports multiple formats:
    1. \\boxed{3.14} - LaTeX boxed format (Qwen thinking models)
    2. "Final answer: 3.14" - explicit format
    3. Plain number in text

    Args:
        response: Model response

    Returns:
        Extracted float, or None if no valid number found
    """
    # First, clean the response
    cleaned = clean_response(response)

    # Try to extract from \boxed{} format first (highest priority)
    boxed_match = re.search(r'\\boxed\{(-?\d+\.?\d*)\}', cleaned)
    if boxed_match:
        try:
            return float(boxed_match.group(1))
        except ValueError:
            pass

    # Try to find after "Final answer:" marker
    final_match = re.search(r'Final answer:\s*(-?\d+\.?\d*)', cleaned, re.IGNORECASE)
    if final_match:
        try:
            return float(final_match.group(1))
        except ValueError:
            pass

    # Fallback: Try to find the first number (possibly negative, possibly with decimal point)
    match = re.search(r'-?\d+\.?\d*', cleaned)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None

    return None


# ============================================================================
# Yes/No Extraction (Benchmarks 2 and 5)
# ============================================================================

def extract_yes_no(response: str) -> Optional[bool]:
    """
    Extract a Yes/No answer from a response.

    Supports multiple formats:
    1. \\boxed{Yes} or \\boxed{No} - LaTeX boxed format (Qwen thinking models)
    2. "Final answer: Yes/No" - explicit format
    3. Plain yes/no in text

    Args:
        response: Model response

    Returns:
        True for "Yes", False for "No", None if ambiguous or invalid
    """
    # Clean and normalize
    cleaned = clean_response(response).lower()

    # Try to extract from \boxed{} format first (highest priority)
    boxed_match = re.search(r'\\boxed\{(yes|no)\}', cleaned, re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1).lower() == "yes"

    # Try to find after "Final answer:" marker
    final_match = re.search(r'Final answer:\s*(yes|no)\b', cleaned, re.IGNORECASE)
    if final_match:
        return final_match.group(1).lower() == "yes"

    # Check for explicit yes/no at the start or as the only word
    if re.match(r'^yes\b', cleaned) or cleaned == "yes":
        return True
    elif re.match(r'^no\b', cleaned) or cleaned == "no":
        return False

    # Look for the LAST occurrence of yes/no (to get final answer after thinking)
    # Find all matches with their positions
    yes_matches = list(re.finditer(r'\byes\b', cleaned))
    no_matches = list(re.finditer(r'\bno\b', cleaned))

    if yes_matches and no_matches:
        # Both appear - use the last one
        last_yes = yes_matches[-1].start() if yes_matches else -1
        last_no = no_matches[-1].start() if no_matches else -1
        return True if last_yes > last_no else False
    elif yes_matches:
        return True
    elif no_matches:
        return False

    # Ambiguous or invalid
    return None


# ============================================================================
# Multiple Choice Extraction (Benchmark 4: Retrieval)
# ============================================================================

def extract_multiple_choice(response: str, valid_choices: Optional[list] = None) -> Optional[str]:
    """
    Extract a multiple choice answer (A, B, C, D) from a response.

    Supports multiple formats:
    1. \\boxed{A} - LaTeX boxed format (Qwen thinking models)
    2. "Final answer: A" - explicit format
    3. Plain letter in text

    Args:
        response: Model response
        valid_choices: List of valid choice letters (default: ["A", "B", "C", "D"])

    Returns:
        Extracted choice letter (uppercase), or None if invalid
    """
    if valid_choices is None:
        valid_choices = ["A", "B", "C", "D"]

    # Clean the response
    cleaned = clean_response(response).upper()

    # Try to extract from \boxed{} format first (highest priority)
    boxed_match = re.search(r'\\boxed\{([A-D])\}', cleaned, re.IGNORECASE)
    if boxed_match:
        choice = boxed_match.group(1).upper()
        if choice in valid_choices:
            return choice

    # Try to find a single letter choice
    # Look for patterns like "A", "A.", "A)", "(A)", "Answer: A", etc.
    patterns = [
        r'^([A-D])\b',  # Letter at start
        r'^([A-D])[.)]',  # Letter followed by period or paren
        r'^\(([A-D])\)',  # Letter in parentheses
        r'answer[:\s]+([A-D])\b',  # "Answer: A" or similar
        r'Final answer[:\s]+([A-D])\b',  # "Final answer: A"
        r'choice[:\s]+([A-D])\b',  # "Choice: A" or similar
        r'\b([A-D])\b',  # Any single letter
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            choice = match.group(1).upper()
            if choice in valid_choices:
                return choice

    return None


# ============================================================================
# Molecular String Extraction (Benchmarks 6 and 7)
# ============================================================================

def extract_molecule_string(response: str, representation: str) -> Optional[str]:
    """
    Extract a molecular string from a generation response.

    Handles multiple output formats:
    - Pure molecule string (direct output or after thinking tags)
    - {"molecule": "VALUE"} JSON, inline or inside a ```json``` code block
    - MolJSON: {"atoms": ..., "bonds": ...} object in a code block
    - Prose reasoning with the answer embedded at the end (phi-4 style)

    Args:
        response: Model response (may be full prose or just the answer)
        representation: Expected representation type (for validation)

    Returns:
        Extracted molecule string, or None if invalid
    """
    if not response:
        return None

    # --- 1. Try {"molecule": "VALUE"} JSON extraction (works for all non-moljson reps) ---
    # Handles both inline JSON and ```json ... ``` code blocks
    if representation != "moljson":
        mol_json_match = re.search(r'\{[^{}]*"molecule"\s*:\s*"([^"]+)"[^{}]*\}', response, re.DOTALL)
        if mol_json_match:
            return mol_json_match.group(1).strip()

    # --- 2. For moljson, extract {"atoms": ..., "bonds": ...} object from a code block ---
    if representation == "moljson":
        # Look for a fenced code block containing JSON with atoms/bonds
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if code_block_match:
            candidate = code_block_match.group(1).strip()
            if '"atoms"' in candidate and '"bonds"' in candidate:
                return candidate
        # Also try bare JSON object with atoms/bonds
        try:
            import json as _json
            bare_match = re.search(r'\{[^{}]*"atoms".*?"bonds".*?\}', response, re.DOTALL)
            if bare_match:
                _json.loads(bare_match.group(0))  # validate
                return bare_match.group(0).strip()
        except Exception:
            pass

    # --- 3. Clean and apply prefix stripping for the first-line fallback ---
    cleaned = clean_response(response)

    prefixes = [
        r'^molecule[:\s]+',
        r'^answer[:\s]+',
        r'^complete[:\s]+',
        r'^result[:\s]+',
        r'^output[:\s]+',
    ]
    for prefix in prefixes:
        cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE)

    # --- 4. For SELFIES, scan for the longest [token]-style sequence ---
    if representation == "selfies":
        # Find all [...] sequences and return the longest (most likely to be the real answer)
        matches = re.findall(r'(\[(?:[^\[\]]*)\](?:\[(?:[^\[\]]*)\])*)', cleaned)
        if matches:
            return max(matches, key=len).strip()

    # --- 5. First non-empty line fallback ---
    for line in cleaned.split('\n'):
        line = line.strip()
        if line:
            return line

    return None


# ============================================================================
# Response Validation
# ============================================================================

def validate_response_for_benchmark(
    response: str,
    benchmark_num: int,
    representation: Optional[str] = None
) -> Tuple[bool, Optional[Union[int, float, bool, str]]]:
    """
    Validate and parse a response for a specific benchmark.

    Args:
        response: Model response
        benchmark_num: Benchmark number (1-7)
        representation: Molecular representation (needed for benchmarks 6-7)

    Returns:
        Tuple of (is_valid, parsed_value)
        - is_valid: Whether the response could be parsed
        - parsed_value: The extracted value (type depends on benchmark)
    """
    # First extract answer from thinking tags if present
    _, answer = extract_answer_from_thinking_response(response)

    if benchmark_num == 1:
        # Atom counting - expect integer
        value = extract_integer(answer)
        return (value is not None, value)

    elif benchmark_num == 2:
        # Functional groups - expect yes/no
        value = extract_yes_no(answer)
        return (value is not None, value)

    elif benchmark_num == 3:
        # Property estimation - expect float
        value = extract_float(answer)
        return (value is not None, value)

    elif benchmark_num == 4:
        # Retrieval - expect A/B/C/D
        value = extract_multiple_choice(answer)
        return (value is not None, value)

    elif benchmark_num == 5:
        # Isomer discrimination - expect yes/no
        value = extract_yes_no(answer)
        return (value is not None, value)

    elif benchmark_num == 6 or benchmark_num == 7:
        # Generation/completion - expect molecule string
        if representation is None:
            raise ValueError("Representation required for benchmarks 6-7")
        value = extract_molecule_string(answer, representation)
        return (value is not None, value)

    else:
        raise ValueError(f"Unknown benchmark number: {benchmark_num}")


# ============================================================================
# Batch Parsing
# ============================================================================

def parse_responses_batch(
    responses: list,
    benchmark_num: int,
    representation: Optional[str] = None
) -> Tuple[list, list]:
    """
    Parse a batch of responses.

    Args:
        responses: List of model responses
        benchmark_num: Benchmark number
        representation: Molecular representation (needed for benchmarks 6-7)

    Returns:
        Tuple of (valid_flags, parsed_values)
        - valid_flags: List of booleans indicating parse success
        - parsed_values: List of parsed values (None if parse failed)
    """
    valid_flags = []
    parsed_values = []

    for response in responses:
        is_valid, value = validate_response_for_benchmark(response, benchmark_num, representation)
        valid_flags.append(is_valid)
        parsed_values.append(value)

    return valid_flags, parsed_values


# ============================================================================
# Testing and Debugging
# ============================================================================

def test_parsing():
    """Test parsing functions with example responses."""
    print("Testing Response Parsing\n" + "=" * 50)

    # Test integer extraction
    print("\n1. Integer Extraction:")
    test_cases = [
        "42",
        "The answer is 42",
        "There are 42 carbon atoms.",
        "```\n42\n```",
        "I think it's 42.",
    ]
    for test in test_cases:
        result = extract_integer(test)
        print(f"  '{test[:30]}...' → {result}")

    # Test float extraction
    print("\n2. Float Extraction:")
    test_cases = [
        "3.14",
        "The LogP is 3.14",
        "approximately 3.14",
        "-2.5",
    ]
    for test in test_cases:
        result = extract_float(test)
        print(f"  '{test}' → {result}")

    # Test yes/no extraction
    print("\n3. Yes/No Extraction:")
    test_cases = [
        "Yes",
        "No",
        "Yes, it contains a hydroxyl group.",
        "No, there is no carboxylic acid.",
        "Maybe",
        "I think yes",
    ]
    for test in test_cases:
        result = extract_yes_no(test)
        print(f"  '{test}' → {result}")

    # Test multiple choice extraction
    print("\n4. Multiple Choice Extraction:")
    test_cases = [
        "A",
        "B.",
        "(C)",
        "The answer is D",
        "I choose option B",
        "Answer: A",
    ]
    for test in test_cases:
        result = extract_multiple_choice(test)
        print(f"  '{test}' → {result}")

    # Test thinking tag extraction
    print("\n5. Thinking Tag Extraction:")
    test_cases = [
        "<think>Let me count... 1, 2, 3</think>The answer is 3",
        "Just 3, no thinking here",
    ]
    for test in test_cases:
        full, answer = extract_answer_from_thinking_response(test)
        print(f"  Full: '{full[:40]}...'")
        print(f"  Answer: '{answer}'")

    # Test molecule string extraction
    print("\n6. Molecule String Extraction:")
    test_cases = [
        ("CCO", "canonical_smiles"),
        ("Molecule: CC(C)O", "canonical_smiles"),
        ("[C][C][O]", "selfies"),
    ]
    for test, rep in test_cases:
        result = extract_molecule_string(test, rep)
        print(f"  '{test}' ({rep}) → '{result}'")

    print("\n✓ Parsing tests completed!")


if __name__ == "__main__":
    test_parsing()
