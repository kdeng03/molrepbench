"""
02_run_inference.py - Run inference for all benchmarks

This script runs model inference across all benchmark tasks. It supports:
- Resumable checkpoints (JSONL format)
- Parallel execution per model/representation/thinking condition
- Dry-run mode for prompt verification
- CLI control over which benchmarks/models/representations to run

Architecture:
- Generic call_model() handles all vLLM API calls
- Per-benchmark runner functions (run_benchmark_1, run_benchmark_2, etc.)
- Incremental JSONL saving for crash recovery
- Final CSV consolidation step
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    BENCHMARKS,
    BENCHMARK_NUM_TO_NAME,
    BENCHMARK_TOKEN_BUDGETS,
    DATA_DIR,
    ELEMENTS,
    GENERATION_SAMPLE_SIZE,
    MODEL_IDS,
    MODELS,
    REPRESENTATIONS,
    RESULTS_DIR,
    SEED,
    STOP_SEQUENCES,
)
from utils import chemistry, parsing, prompts, representations

# Global variable for direct vLLM model (used with --direct-vllm flag)
VLLM_MODEL = None

# Setup logging - force flush and use stderr to avoid tqdm interference
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
    force=True,  # Override any existing configuration (e.g., from vLLM)
)
logger = logging.getLogger(__name__)

# Ensure stderr is unbuffered for immediate output
sys.stderr.reconfigure(line_buffering=True)


# ============================================================================
# Direct vLLM Model Loading (Optimization #4)
# ============================================================================

# Global tokenizer for chat template application (avoid reloading)
VLLM_TOKENIZER = None


def initialize_vllm_model(model_config: Dict[str, Any], tensor_parallel_size: int = 1):
    """
    Initialize vLLM model for direct inference (no API server needed).

    Args:
        model_config: Model configuration from config.MODELS
        tensor_parallel_size: Number of GPUs to use for tensor parallelism

    Returns:
        vLLM LLM object
    """
    global VLLM_TOKENIZER

    try:
        from vllm import LLM
        from transformers import AutoTokenizer

        logger.info(f"Loading model {model_config['hf_id']} directly with vLLM...")
        logger.info(f"Tensor parallel size: {tensor_parallel_size}")

        # Check if this is a thinking model (has thinking_on_config)
        is_thinking_model = "thinking_on_config" in model_config

        # Determine max_model_len based on model config
        # For Qwen3-4B-Thinking: 8192, for Qwen3-30B-A3B: 32768
        # max_len = 32768
        # if "qwen3-30b" in model_config["id"].lower():
        #     max_len = 32768

        llm_kwargs = {
            "model": model_config["hf_id"],
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": True,
            "dtype": "auto",
            "gpu_memory_utilization": 0.90,
            "max_model_len": model_config.get("max_model_len", 32768),
            "attention_backend": "FLASH_ATTN",  # Use FlashAttention instead of FlashInfer
        }

        # Add reasoning parser for thinking models
        if is_thinking_model:
            # Use model-specific reasoning parser if specified, otherwise default to qwen3
            # None means no reasoning parser (e.g. phi-4 base has thinking_on_config but no <think> tokens)
            reasoning_parser = model_config.get("reasoning_parser", "qwen3")
            if reasoning_parser is not None:
                llm_kwargs["reasoning_parser"] = reasoning_parser
                logger.info(f"Enabling reasoning parser: {reasoning_parser}")
                # Allow structured output after reasoning (don't suppress <think> tags)
                from vllm.config import StructuredOutputsConfig
                llm_kwargs["structured_outputs_config"] = StructuredOutputsConfig(
                    reasoning_parser=reasoning_parser,
                    enable_in_reasoning=True,
                )
                logger.info("Enabled structured output with enable_in_reasoning=True")

        llm = LLM(**llm_kwargs)

        # Load tokenizer once globally for batch template application
        logger.info("Loading tokenizer for chat template application...")
        VLLM_TOKENIZER = AutoTokenizer.from_pretrained(
            model_config["hf_id"],
            trust_remote_code=True
        )

        logger.info(f"Model and tokenizer loaded successfully!")
        return llm
    except ImportError:
        logger.error("vLLM not installed. Install with: pip install vllm")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)


def call_model_direct_vllm(
    messages: List[Dict[str, str]],
    model_config: Dict[str, Any],
    thinking: bool,
    generation_params: Dict[str, Any],
    stop_sequences: Optional[List[str]] = None,
    json_schema: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Call vLLM model directly (no API server) for faster inference.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model_config: Model configuration from config.MODELS
        thinking: Whether thinking is enabled
        generation_params: Dict with temperature, max_tokens, top_p, and optionally top_k
        stop_sequences: Optional list of stop sequences
        json_schema: Optional JSON schema dict for structured output

    Returns:
        Dict with raw_response, parsed_answer, model, thinking, latency_ms
    """
    global VLLM_MODEL

    if VLLM_MODEL is None:
        logger.error("Direct vLLM model not initialized. Call initialize_vllm_model() first.")
        return None

    try:
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        # Skip stop sequences for models that reason in plain text with thinking off
        skip_stop = not thinking and model_config.get("skip_stop_on_thinking_off", False)

        # Setup sampling params
        sampling_params = SamplingParams(
            temperature=generation_params["temperature"],
            max_tokens=generation_params["max_tokens"],
            top_p=generation_params["top_p"],
            stop=stop_sequences if stop_sequences and not thinking and not skip_stop else None,
        )

        if "top_k" in generation_params:
            sampling_params.top_k = generation_params["top_k"]

        # For thinking models, set thinking token budget
        if thinking and "thinking_max_tokens" in generation_params:
            thinking_budget = generation_params["thinking_max_tokens"]
            sampling_params.thinking_token_budget = thinking_budget

        # Resolve thinking config for current mode
        if thinking and "thinking_on_config" in model_config:
            _thinking_cfg = model_config["thinking_on_config"]
        elif not thinking and "thinking_off_config" in model_config:
            _thinking_cfg = model_config["thinking_off_config"]
        else:
            _thinking_cfg = {}

        chat_kwargs = _thinking_cfg.get("chat_template_kwargs", {})

        # Add structured output for JSON schema (same logic as API mode)
        if json_schema is not None:
            is_qwen_model = "chat_template_kwargs" in _thinking_cfg
            if is_qwen_model or not thinking:
                sampling_params.structured_outputs = StructuredOutputsParams(json=json_schema["schema"])
                logger.info(f"✓ Direct vLLM: using structured output with schema: {json_schema['name']}")
            else:
                logger.info(f"Direct vLLM: thinking mode (non-Qwen), relying on prompt for JSON format (schema: {json_schema['name']})")

        # Inject system message if defined for this thinking mode
        _system_msg = _thinking_cfg.get("system_message")
        if _system_msg:
            if messages and messages[0]["role"] == "system":
                messages = [{"role": "system", "content": _system_msg}] + messages[1:]
            else:
                messages = [{"role": "system", "content": _system_msg}] + list(messages)

        # Use chat() method instead of generate() for proper chat template support
        start_time = time.time()
        outputs = VLLM_MODEL.chat(
            messages,
            sampling_params,
            chat_template_kwargs=chat_kwargs if chat_kwargs else None
        )
        latency_ms = (time.time() - start_time) * 1000

        # Extract response - with reasoning parser, output may have 'reasoning' field
        output = outputs[0].outputs[0]

        # Check if reasoning output is available (from reasoning parser)
        if hasattr(output, 'reasoning') and output.reasoning:
            reasoning = output.reasoning
            content = output.text if hasattr(output, 'text') else ""
            raw_response = f"<think>{reasoning}</think>\n{content}" if reasoning else content
        else:
            raw_response = output.text

        # Ensure raw_response is a string (some models/parsers return a list)
        if isinstance(raw_response, list):
            raw_response = "".join(str(part) for part in raw_response)
        elif not isinstance(raw_response, str):
            raw_response = str(raw_response) if raw_response is not None else ""

        _, parsed_answer = parsing.extract_answer_from_thinking_response(raw_response)

        # Handle JSON schema extraction
        if json_schema is not None:
            json_to_parse = parsed_answer if thinking else raw_response
            try:
                json_obj = json.loads(json_to_parse)
                if "molecule" in json_obj:
                    parsed_answer = json_obj["molecule"]
                elif "atoms" in json_obj and "bonds" in json_obj:
                    parsed_answer = json.dumps(json_obj)
            except json.JSONDecodeError:
                pass

        return {
            "raw_response": raw_response,
            "parsed_answer": parsed_answer,
            "model": model_config["id"],
            "thinking": thinking,
            "latency_ms": latency_ms,
        }

    except Exception as e:
        logger.error(f"Direct vLLM inference failed: {e}")
        return None


def call_model_direct_vllm_batch(
    messages_list: List[List[Dict[str, str]]],
    model_config: Dict[str, Any],
    thinking: bool,
    generation_params: Dict[str, Any],
    stop_sequences: Optional[List[str]] = None,
    json_schema: Optional[Dict[str, Any]] = None,
) -> List[Optional[Dict[str, Any]]]:
    """
    Call vLLM model directly with batch of prompts (FASTEST option).

    Uses .generate() with manually applied chat templates for true parallel batching.

    Args:
        messages_list: List of message lists
        model_config: Model configuration
        thinking: Whether thinking is enabled
        generation_params: Generation parameters
        stop_sequences: Optional stop sequences
        json_schema: Optional JSON schema

    Returns:
        List of result dicts
    """
    global VLLM_MODEL

    if VLLM_MODEL is None:
        logger.error("Direct vLLM model not initialized.")
        return [None] * len(messages_list)

    global VLLM_TOKENIZER

    if VLLM_TOKENIZER is None:
        logger.error("Tokenizer not initialized. Call initialize_vllm_model() first.")
        return [None] * len(messages_list)

    try:
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        # Skip stop sequences for models that reason in plain text with thinking off
        skip_stop = not thinking and model_config.get("skip_stop_on_thinking_off", False)

        # Setup sampling params
        sampling_params = SamplingParams(
            temperature=generation_params["temperature"],
            max_tokens=generation_params["max_tokens"],
            top_p=generation_params["top_p"],
            stop=stop_sequences if stop_sequences and not thinking and not skip_stop else None,
        )

        if "top_k" in generation_params:
            sampling_params.top_k = generation_params["top_k"]

        # For thinking models, set thinking token budget
        if thinking and "thinking_max_tokens" in generation_params:
            thinking_budget = generation_params["thinking_max_tokens"]
            sampling_params.thinking_token_budget = thinking_budget

        # Resolve thinking config for current mode
        if thinking and "thinking_on_config" in model_config:
            _thinking_cfg = model_config["thinking_on_config"]
        elif not thinking and "thinking_off_config" in model_config:
            _thinking_cfg = model_config["thinking_off_config"]
        else:
            _thinking_cfg = {}

        # Add structured output for JSON schema (same logic as API mode)
        if json_schema is not None:
            is_qwen_model = "chat_template_kwargs" in _thinking_cfg
            if is_qwen_model or not thinking:
                sampling_params.structured_outputs = StructuredOutputsParams(json=json_schema["schema"])
                logger.info(f"✓ Direct vLLM batch: using structured output with schema: {json_schema['name']}")
            else:
                logger.info(f"Direct vLLM batch: thinking mode (non-Qwen), relying on prompt for JSON format (schema: {json_schema['name']})")

        chat_template_kwargs = _thinking_cfg.get("chat_template_kwargs", {})
        _system_msg = _thinking_cfg.get("system_message")

        # Manually apply chat template to each conversation (using global tokenizer)
        prompts = []
        for messages in messages_list:
            msgs = list(messages)
            if _system_msg:
                if msgs and msgs[0]["role"] == "system":
                    msgs = [{"role": "system", "content": _system_msg}] + msgs[1:]
                else:
                    msgs = [{"role": "system", "content": _system_msg}] + msgs
            prompt = VLLM_TOKENIZER.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs=chat_template_kwargs if chat_template_kwargs else None
            )
            prompts.append(prompt)

        # Batch generate using generate() method for TRUE parallel batching
        logger.info(f"Batching {len(prompts)} prompts with .generate() for parallel processing")
        start_time = time.time()
        outputs = VLLM_MODEL.generate(prompts, sampling_params, use_tqdm=False) # TODO
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Batch generation took {latency_ms/1000:.2f}s for {len(prompts)} prompts ({latency_ms/len(prompts):.0f}ms per prompt)")

        # Process results
        results = []
        for output in outputs:
            output_data = output.outputs[0]

            # Check if reasoning output is available (from reasoning parser)
            if hasattr(output_data, 'reasoning') and output_data.reasoning:
                reasoning = output_data.reasoning
                content = output_data.text if hasattr(output_data, 'text') else ""
                raw_response = f"<think>{reasoning}</think>\n{content}" if reasoning else content
            else:
                raw_response = output_data.text

            _, parsed_answer = parsing.extract_answer_from_thinking_response(raw_response)

            # Handle JSON schema extraction
            if json_schema is not None:
                json_to_parse = parsed_answer if thinking else raw_response
                try:
                    json_obj = json.loads(json_to_parse)
                    if "molecule" in json_obj:
                        parsed_answer = json_obj["molecule"]
                    elif "atoms" in json_obj and "bonds" in json_obj:
                        parsed_answer = json.dumps(json_obj)
                except json.JSONDecodeError:
                    pass

            results.append({
                "raw_response": raw_response,
                "parsed_answer": parsed_answer,
                "model": model_config["id"],
                "thinking": thinking,
                "latency_ms": latency_ms / len(messages_list),  # Average latency per message
            })

        return results

    except Exception as e:
        logger.error(f"Direct vLLM batch inference failed: {e}")
        import traceback
        traceback.print_exc()
        return [None] * len(messages_list)

# ============================================================================
# JSON Schema Helper for Structured Output (Benchmarks 6 & 7)
# ============================================================================


def get_json_schema_for_representation(repr_name: str) -> Optional[Dict[str, Any]]:
    """
    Get JSON schema for structured output based on representation.

    Args:
        repr_name: Representation name (e.g., "smiles", "moljson", etc.)

    Returns:
        Dict with "name" and "schema" keys, or None if schema not needed
    """
    # For MolJSON, use the MolJSON schema
    if repr_name == "moljson":
        try:
            # Try to import MolJSON schema
            import sys
            from pathlib import Path
            moljson_dir = Path(__file__).resolve().parents[1] / "moljson"
            if str(moljson_dir) not in sys.path:
                sys.path.insert(0, str(moljson_dir))
            from schema import GetSchema

            return {
                "name": "MolJSON",
                "schema": GetSchema()
            }
        except Exception as e:
            logger.warning(f"Could not load MolJSON schema: {e}")
            return None

    # For string-based formats (SMILES, IUPAC, SELFIES, etc.)
    # Create a simple schema that forces the output format
    if repr_name in ["smiles", "canonical_smiles", "isomeric_smiles", "randomized_smiles",
                     "deepsmiles", "iupac", "selfies", "inchi", "cml"]:
        # Normalize representation name for schema
        field_name = repr_name.replace("_", "")  # e.g., "canonical_smiles" -> "canonicalsmiles"

        return {
            "name": f"{field_name}_answer",
            "schema": {
                "type": "object",
                "properties": {
                    "molecule": {
                        "type": "string",
                        "description": f"Molecule written as {repr_name} ONLY. Do not ask clarifying questions. Do not write any comments."
                    }
                },
                "required": ["molecule"],
                "additionalProperties": False
            }
        }

    # No schema for other formats
    return None


# ============================================================================
# Helper Functions
# ============================================================================


def get_generation_params(
    model_config: Dict[str, Any],
    benchmark_name: str,
    thinking: bool
) -> Dict[str, Any]:
    """
    Get generation parameters for a model, benchmark, and thinking mode.

    Args:
        model_config: Model configuration from config.MODELS
        benchmark_name: Benchmark name (e.g., "atom_counting", "functional_groups")
        thinking: Whether thinking mode is enabled

    Returns:
        Dict with generation parameters (temperature, max_tokens, top_p, top_k, thinking_max_tokens)

    Raises:
        ValueError: If parameters are not configured
    """
    if "generation_params" not in model_config:
        model_id = model_config["id"]
        logger.error(
            f"ERROR: Model '{model_id}' does not have 'generation_params' configured in config.py."
        )
        raise ValueError(f"Missing generation_params for model '{model_id}'")

    # Get model-specific sampling parameters (temperature, top_p, top_k)
    model_params = model_config["generation_params"].copy()

    # Get benchmark-specific token budgets
    # if benchmark_name not in BENCHMARK_TOKEN_BUDGETS:
    #     logger.error(
    #         f"ERROR: Benchmark '{benchmark_name}' does not have token budgets configured in BENCHMARK_TOKEN_BUDGETS."
    #     )
    #     raise ValueError(f"Missing token budgets for benchmark '{benchmark_name}'")

    # thinking_mode = "thinking_on" if thinking else "thinking_off"
    # token_budgets = BENCHMARK_TOKEN_BUDGETS[benchmark_name][thinking_mode]

    # Use model params directly (max_tokens=32768 set per model in config)
    params = model_params

    return params


# ============================================================================
# Core Model Call Function
# ============================================================================


def call_model_batch(
    messages_list: List[List[Dict[str, str]]],
    model_config: Dict[str, Any],
    thinking: bool,
    vllm_url: str,
    generation_params: Dict[str, Any],
    stop_sequences: Optional[List[str]] = None,
    json_schema: Optional[Dict[str, Any]] = None,
) -> List[Optional[Dict[str, Any]]]:
    """
    Call vLLM API with batch of prompts for faster inference.

    Args:
        messages_list: List of message lists (each item is a conversation)
        model_config: Model configuration from config.MODELS
        thinking: Whether thinking is enabled
        vllm_url: Base URL for vLLM server
        generation_params: Dict with temperature, max_tokens, top_p, and optionally top_k
        stop_sequences: Optional list of stop sequences
        json_schema: Optional JSON schema dict with 'name' and 'schema' keys for structured output

    Returns:
        List of result dicts (same format as call_model), None for failed requests
    """
    if not messages_list:
        return []

    model_id = model_config["id"]
    thinking_config = (
        model_config["thinking_on_config"]
        if thinking
        else model_config["thinking_off_config"]
    )

    # Build batch request data
    batch_requests = []
    for messages in messages_list:
        request_data = {
            "model": model_config["hf_id"],
            "messages": messages,
            "temperature": generation_params["temperature"],
            "max_tokens": generation_params["max_tokens"],
            "top_p": generation_params["top_p"],
        }

        if "top_k" in generation_params:
            request_data["top_k"] = generation_params["top_k"]

        if json_schema is not None:
            is_qwen_model = "chat_template_kwargs" in thinking_config
            if is_qwen_model or not thinking:
                request_data["guided_json"] = json_schema["schema"]

        skip_stop = not thinking and model_config.get("skip_stop_on_thinking_off", False)
        if stop_sequences and not thinking and not skip_stop:
            request_data["stop"] = stop_sequences

        if "chat_template_kwargs" in thinking_config:
            if "extra_body" not in request_data:
                request_data["extra_body"] = {}
            request_data["extra_body"]["chat_template_kwargs"] = thinking_config["chat_template_kwargs"]

        if "system_message" in thinking_config:
            system_msg = {"role": "system", "content": thinking_config["system_message"]}
            if messages[0]["role"] == "system":
                messages[0] = system_msg
            else:
                messages = [system_msg] + messages

        if thinking and "thinking_max_tokens" in generation_params:
            thinking_max_tokens = generation_params["thinking_max_tokens"]
            request_data["max_tokens"] = thinking_max_tokens

        batch_requests.append(request_data)

    # Send batch request
    api_url = f"{vllm_url}/chat/completions"
    results = []
    start_time = time.time()

    # Process as individual requests for now (vLLM batches them internally)
    # Can be optimized with async requests later
    for request_data in batch_requests:
        try:
            response = requests.post(api_url, json=request_data, timeout=120)
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code != 200:
                logger.error(f"API error {response.status_code}: {response.text[:200]}")
                results.append(None)
                continue

            response_json = response.json()
            message = response_json["choices"][0]["message"]

            # Handle vLLM reasoning parser output format
            # With --reasoning-parser, response has separate "reasoning" and "content" fields
            if "reasoning" in message and message["reasoning"]:
                # New format: reasoning is separate from content
                reasoning = message["reasoning"]
                content = message.get("content", "")
                raw_response = f"<think>{reasoning}</think>\n{content}" if reasoning else content
            else:
                # Fallback: content only (thinking disabled or old vLLM version)
                raw_response = message["content"]

            _, parsed_answer = parsing.extract_answer_from_thinking_response(raw_response)

            # Handle JSON schema extraction
            if json_schema is not None:
                json_to_parse = parsed_answer if thinking else raw_response
                try:
                    json_obj = json.loads(json_to_parse)
                    if "molecule" in json_obj:
                        parsed_answer = json_obj["molecule"]
                    elif "atoms" in json_obj and "bonds" in json_obj:
                        parsed_answer = json.dumps(json_obj)
                except json.JSONDecodeError:
                    import re
                    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', json_to_parse, re.DOTALL)
                    if json_match:
                        try:
                            json_obj = json.loads(json_match.group(0))
                            if "molecule" in json_obj:
                                parsed_answer = json_obj["molecule"]
                            elif "atoms" in json_obj and "bonds" in json_obj:
                                parsed_answer = json.dumps(json_obj)
                        except json.JSONDecodeError:
                            pass

            results.append({
                "raw_response": raw_response,
                "parsed_answer": parsed_answer,
                "model": model_id,
                "thinking": thinking,
                "latency_ms": latency_ms,
            })

        except requests.exceptions.Timeout:
            logger.error(f"API timeout after 120s")
            results.append(None)
        except Exception as e:
            logger.error(f"API call failed: {e}")
            results.append(None)

    return results


def call_model(
    messages: List[Dict[str, str]],
    model_config: Dict[str, Any],
    thinking: bool,
    vllm_url: str,
    generation_params: Dict[str, Any],
    stop_sequences: Optional[List[str]] = None,
    json_schema: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Call the vLLM API with OpenAI-compatible chat completions endpoint.
    If VLLM_MODEL is initialized (direct mode), uses direct inference instead.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model_config: Model configuration from config.MODELS
        thinking: Whether thinking is enabled
        vllm_url: Base URL for vLLM server (ignored if direct mode)
        generation_params: Dict with temperature, max_tokens, top_p, and optionally top_k
        stop_sequences: Optional list of stop sequences
        json_schema: Optional JSON schema dict with 'name' and 'schema' keys for structured output
        dry_run: If True, print prompt and return None

    Returns:
        Dict with:
            - raw_response: Full model output (str)
            - parsed_answer: Answer after thinking extraction (str)
            - model: Model ID (str)
            - thinking: Whether thinking was enabled (bool)
            - latency_ms: API call latency in milliseconds (float)
        Returns None if API call fails.
    """
    global VLLM_MODEL

    # If direct vLLM mode is active, use direct inference
    if VLLM_MODEL is not None and not dry_run:
        return call_model_direct_vllm(
            messages=messages,
            model_config=model_config,
            thinking=thinking,
            generation_params=generation_params,
            stop_sequences=stop_sequences,
            json_schema=json_schema,
        )
    model_id = model_config["id"]
    model_name = model_config["name"]

    # Validate that model has generation_params configured
    if "generation_params" not in model_config:
        logger.error(
            f"ERROR: Model '{model_id}' does not have 'generation_params' configured in config.py. "
            f"Please add generation parameters for this model."
        )
        raise ValueError(f"Missing generation_params for model '{model_id}'")

    # Apply thinking configuration
    thinking_config = (
        model_config["thinking_on_config"]
        if thinking
        else model_config["thinking_off_config"]
    )

    # For dry run, print prompt and return
    if dry_run:
        print(f"\n{'='*80}")
        print(f"Model: {model_name} | Thinking: {thinking}")
        print(f"Thinking Config: {thinking_config}")
        print(f"Messages:")
        for msg in messages:
            print(f"  [{msg['role']}]: {msg['content'][:500]}...")
        print(f"Generation Params: {generation_params}")
        print(f"Stop Sequences: {stop_sequences}")
        if json_schema is not None:
            print(f"\nJSON Schema (Structured Output):")
            print(f"  Name: {json_schema['name']}")
            print(f"  Schema: {json.dumps(json_schema['schema'], indent=2)[:500]}...")
            print(f"\nExpected Output Format:")
            if "molecule" in str(json_schema['schema']):
                print(f'  {{"molecule": "<complete_molecule_string>"}}')
            else:
                print(f'  {{"atoms": [...], "bonds": [...], ...}}')
        print(f"{'='*80}\n")
        sys.stdout.flush()
        return None

    # Build API request
    request_data = {
        "model": model_config["hf_id"],
        "messages": messages,
        "temperature": generation_params["temperature"],
        "max_tokens": generation_params["max_tokens"],
        "top_p": generation_params["top_p"],
    }

    # Add top_k if present in generation_params (model-specific)
    if "top_k" in generation_params:
        request_data["top_k"] = generation_params["top_k"]

    # Add JSON schema for structured output (if provided)
    # vLLM supports guided_json parameter for structured output
    # See: https://docs.vllm.ai/en/latest/features/structured_outputs/
    # For Qwen models with thinking enabled, vLLM supports BOTH reasoning and structured output:
    # https://docs.vllm.ai/en/latest/features/reasoning_outputs/
    # The thinking process goes inside <think> tags, followed by structured JSON output
    if json_schema is not None:
        # Check if this is a Qwen model (has chat_template_kwargs)
        is_qwen_model = "chat_template_kwargs" in thinking_config
        print(f"DEBUG: JSON schema={json_schema['name']}, is_qwen={is_qwen_model}, thinking={thinking}", flush=True)

        # For Qwen models: always enable guided_json (works with thinking mode)
        # For other models: only enable when thinking is OFF
        if is_qwen_model or not thinking:
            # vLLM uses "guided_json" as a top-level parameter for structured output
            request_data["guided_json"] = json_schema["schema"]
            print(f"DEBUG: ✓ Added guided_json to request_data", flush=True)
            logger.info(f"✓ Using structured output (guided_json) with schema: {json_schema['name']}")
        else:
            # Non-Qwen models with thinking: rely on prompt for JSON format
            logger.info(f"Thinking mode (non-Qwen): relying on prompt for JSON format (schema: {json_schema['name']})")

    # Add stop sequences if provided (but NOT when thinking is enabled)
    # Thinking mode outputs often contain blank lines which would trigger stop sequences
    skip_stop = not thinking and model_config.get("skip_stop_on_thinking_off", False)
    if stop_sequences and not thinking and not skip_stop:
        request_data["stop"] = stop_sequences

    # Apply thinking config
    # For Qwen models: chat_template_kwargs
    if "chat_template_kwargs" in thinking_config:
        if "extra_body" not in request_data:
            request_data["extra_body"] = {}
        request_data["extra_body"]["chat_template_kwargs"] = thinking_config["chat_template_kwargs"]

    # For Nemotron: system message
    if "system_message" in thinking_config:
        # Prepend or modify system message in messages
        system_msg = {"role": "system", "content": thinking_config["system_message"]}
        if messages[0]["role"] == "system":
            messages[0] = system_msg
        else:
            messages = [system_msg] + messages

    # If thinking is on, increase max_tokens
    if thinking and "thinking_max_tokens" in generation_params:
        thinking_max_tokens = generation_params["thinking_max_tokens"]
        request_data["max_tokens"] = thinking_max_tokens

    # Make API call
    api_url = f"{vllm_url}/chat/completions"
    start_time = time.time()

    try:
        response = requests.post(api_url, json=request_data, timeout=120)
        latency_ms = (time.time() - start_time) * 1000

        if response.status_code != 200:
            logger.error(
                f"API error {response.status_code}: {response.text[:200]}"
            )
            return None

        response_json = response.json()
        raw_response = response_json["choices"][0]["message"]["content"]

        # Parse thinking tags if present
        _, parsed_answer = parsing.extract_answer_from_thinking_response(raw_response)

        # If JSON schema was used, the response should be valid JSON
        # Try to extract the molecule field if present
        if json_schema is not None:
            # For thinking mode, the JSON might come AFTER the thinking tags
            # Try to parse the parsed_answer (which has thinking stripped)
            json_to_parse = parsed_answer if thinking else raw_response

            try:
                json_obj = json.loads(json_to_parse)
                # Extract the molecule field from structured output
                if "molecule" in json_obj:
                    parsed_answer = json_obj["molecule"]
                # For MolJSON, the entire object is the molecule
                elif "atoms" in json_obj and "bonds" in json_obj:
                    parsed_answer = json.dumps(json_obj)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to find JSON in the response
                # Look for a JSON object pattern
                import re
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', json_to_parse, re.DOTALL)
                if json_match:
                    try:
                        json_obj = json.loads(json_match.group(0))
                        if "molecule" in json_obj:
                            parsed_answer = json_obj["molecule"]
                        elif "atoms" in json_obj and "bonds" in json_obj:
                            parsed_answer = json.dumps(json_obj)
                    except json.JSONDecodeError:
                        logger.warning(f"JSON schema enabled but could not parse JSON from response: {json_to_parse[:100]}")
                else:
                    logger.warning(f"JSON schema enabled but no valid JSON found in response: {json_to_parse[:100]}")

        return {
            "raw_response": raw_response,
            "parsed_answer": parsed_answer,
            "model": model_id,
            "thinking": thinking,
            "latency_ms": latency_ms,
        }

    except requests.exceptions.Timeout:
        logger.error(f"API timeout after 120s")
        return None
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return None


# ============================================================================
# Dataset Helper Functions
# ============================================================================


def load_test_dataset(dataset_path: Path):
    """
    Load test dataset and return DataFrame with standardized column names.

    Returns:
        Tuple of (DataFrame, id_column_name, smiles_column_name)
    """
    df = pd.read_csv(dataset_path)

    # Determine ID column
    if "CID" in df.columns:
        id_col = "CID"
    elif "molecule_id" in df.columns:
        id_col = "molecule_id"
    else:
        raise ValueError(f"Dataset must have 'CID' or 'molecule_id' column. Found: {df.columns.tolist()}")

    # Determine SMILES column
    if "smiles" in df.columns:
        smiles_col = "smiles"
    elif "SMILES" in df.columns:
        smiles_col = "SMILES"
    else:
        raise ValueError(f"Dataset must have 'smiles' or 'SMILES' column. Found: {df.columns.tolist()}")

    return df, id_col, smiles_col


# ============================================================================
# Checkpoint Management
# ============================================================================


def load_checkpoint(checkpoint_path: Path, compound_key_field: str = None) -> Set[Tuple[str, str]]:
    """
    Load completed (molecule_id, representation) pairs from JSONL checkpoint.

    Args:
        checkpoint_path: Path to JSONL checkpoint file
        compound_key_field: If set (e.g. "group" or "property"), build keys as
            (molecule_id, f"{field_value}_{representation}") to match the
            skip-check format used by B2 and B3.

    Returns:
        Set of (molecule_id, key) tuples
    """
    completed = set()
    if not checkpoint_path.exists():
        return completed

    with open(checkpoint_path, "r") as f:
        for line in f:
            record = json.loads(line)
            if compound_key_field and compound_key_field in record:
                key = f"{record[compound_key_field]}_{record['representation']}"
            else:
                key = record["representation"]
            completed.add((record["molecule_id"], key))

    logger.info(f"Loaded {len(completed)} completed items from {checkpoint_path}")
    return completed


def load_checkpoint_benchmark_5(checkpoint_path: Path) -> Set[Tuple[int, str]]:
    """
    Load completed (pair_id, representation) pairs from JSONL checkpoint for Benchmark 5.

    Args:
        checkpoint_path: Path to JSONL checkpoint file

    Returns:
        Set of (pair_id, representation) tuples
    """
    completed = set()
    if not checkpoint_path.exists():
        return completed

    with open(checkpoint_path, "r") as f:
        for line in f:
            record = json.loads(line)
            completed.add((record["pair_id"], record["representation"]))

    logger.info(f"Loaded {len(completed)} completed items from {checkpoint_path}")
    return completed


def save_to_checkpoint(record: Dict[str, Any], checkpoint_path: Path, run_id: int = 1):
    """
    Append a record to the JSONL checkpoint file.

    Args:
        record: Dict to save as JSON line
        checkpoint_path: Path to JSONL checkpoint file
        run_id: Run number for error-bar replication (default: 1)
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps({**record, "run_id": run_id}) + "\n")


# ============================================================================
# Benchmark 1: Atom Counting
# ============================================================================


def run_benchmark_1(
    model_config: Dict[str, Any],
    thinking: bool,
    vllm_url: str,
    num_samples: Optional[int] = None,
    dry_run: bool = False,
    use_direct_vllm: bool = False,
    batch_size: int = 50,
    run_id: int = 1,
):
    """
    Run Benchmark 1: Atom Counting

    For each molecule in comprehension subset:
    - Pick one element present in the molecule (randomly, seeded by molecule_id)
    - For each representation:
        - Convert molecule to representation
        - Build prompt
        - Call model
        - Parse response as integer
        - Save result

    Args:
        model_config: Model configuration dict
        thinking: Whether thinking is enabled
        vllm_url: vLLM API URL
        num_samples: If provided, limit to first N molecules
        dry_run: If True, print prompts without calling API
    """
    benchmark_name = "atom_counting"
    model_id = model_config["id"]
    thinking_str = "thinking_on" if thinking else "thinking_off"

    logger.info(f"\n{'='*80}")
    logger.info(f"Starting Benchmark 1: Atom Counting")
    logger.info(f"Model: {model_id} | Thinking: {thinking_str}")
    logger.info(f"{'='*80}\n")

    # Setup checkpoint
    checkpoint_dir = RESULTS_DIR / "raw"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"benchmark_1_{model_id}_{thinking_str}_run{run_id}.jsonl"

    # Load completed items
    completed = load_checkpoint(checkpoint_path)

    # Load data
    subset_df = pd.read_csv(DATA_DIR / "comprehension_subset_ids.csv")
    test_df, id_col, smiles_col = load_test_dataset(DATA_DIR / "chebi20_test.csv")

    # Get molecules in comprehension subset
    molecule_ids = subset_df["molecule_id"].tolist()
    if num_samples:
        molecule_ids = molecule_ids[:num_samples]

    logger.info(f"Processing {len(molecule_ids)} molecules")

    # Counter for dry run
    dry_run_count = 0
    dry_run_limit = 3

    # Phase 1: Pre-collect all pending tasks across all molecules
    from rdkit import Chem
    logger.info("Pre-collecting tasks...")
    all_tasks = []
    for mol_id in molecule_ids:
        mol_row = test_df[test_df[id_col] == mol_id].iloc[0]
        smiles = mol_row[smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES for {mol_id}: {smiles}")
            continue
        element_counts = chemistry.get_atom_counts(mol)
        available_elements = [e for e in ELEMENTS if element_counts.get(e, 0) > 0]
        if not available_elements:
            logger.warning(f"No target elements in {mol_id}")
            continue
        random.seed(hash(mol_id) % (2**32))
        element = random.choice(available_elements)
        ground_truth = element_counts[element]

        for repr_name in REPRESENTATIONS:
            if (mol_id, repr_name) in completed:
                continue
            if repr_name == "iupac":
                repr_string = mol_row["iupac"] if "iupac" in mol_row and pd.notna(mol_row["iupac"]) else None
            elif repr_name == "selfies":
                repr_string = mol_row["selfies"] if "selfies" in mol_row and pd.notna(mol_row["selfies"]) else (
                    mol_row["SELFIES"] if "SELFIES" in mol_row and pd.notna(mol_row["SELFIES"]) else None)
            else:
                repr_string = representations.convert_to_representation(smiles, repr_name)
            if repr_string is None:
                logger.warning(f"Conversion failed: {mol_id} -> {repr_name}")
                continue
            prompt_text = prompts.create_atom_counting_prompt(repr_string, element, repr_name)
            all_tasks.append({
                "messages": [{"role": "user", "content": prompt_text}],
                "mol_id": mol_id,
                "repr_name": repr_name,
                "element": element,
                "ground_truth": ground_truth,
            })

    logger.info(f"Total pending tasks: {len(all_tasks)}")

    if dry_run:
        gen_params = get_generation_params(model_config, "atom_counting", thinking)
        stop_seqs = STOP_SEQUENCES.get("comprehension", [])
        for task in all_tasks[:dry_run_limit]:
            call_model(messages=task["messages"], model_config=model_config, thinking=thinking,
                       vllm_url=vllm_url, generation_params=gen_params, stop_sequences=stop_seqs, dry_run=True)
        logger.info(f"\nDry run limit reached ({dry_run_limit} prompts)")
        return

    # Phase 2: Process in batches
    gen_params = get_generation_params(model_config, "atom_counting", thinking)
    stop_seqs = STOP_SEQUENCES.get("comprehension", [])

    for batch_start in tqdm(range(0, len(all_tasks), batch_size), desc="Batches"):
        batch = all_tasks[batch_start:batch_start + batch_size]
        messages_list = [t["messages"] for t in batch]

        if use_direct_vllm:
            results = call_model_direct_vllm_batch(
                messages_list=messages_list, model_config=model_config, thinking=thinking,
                generation_params=gen_params, stop_sequences=stop_seqs,
            )
        else:
            with ThreadPoolExecutor(max_workers=min(len(batch), 32)) as executor:
                future_to_idx = {
                    executor.submit(call_model, t["messages"], model_config, thinking,
                                    vllm_url, gen_params, stop_seqs): i
                    for i, t in enumerate(batch)
                }
                results = [None] * len(batch)
                for future in as_completed(future_to_idx):
                    results[future_to_idx[future]] = future.result()

        for task, result in zip(batch, results):
            mol_id = task["mol_id"]
            repr_name = task["repr_name"]
            element = task["element"]
            ground_truth = task["ground_truth"]

            if result is None:
                logger.warning(f"API call failed: {mol_id} | {repr_name} | {model_id} | {thinking_str}")
                record = {
                    "molecule_id": mol_id, "representation": repr_name, "model": model_id,
                    "thinking": thinking, "element": element, "raw_response": None,
                    "parsed_answer": None, "predicted": None, "ground_truth": ground_truth,
                    "latency_ms": None,
                }
            else:
                predicted = parsing.extract_integer(result["parsed_answer"])
                record = {
                    "molecule_id": mol_id, "representation": repr_name, "model": model_id,
                    "thinking": thinking, "element": element, "raw_response": result["raw_response"],
                    "parsed_answer": result["parsed_answer"], "predicted": predicted,
                    "ground_truth": ground_truth, "latency_ms": result["latency_ms"],
                }
            save_to_checkpoint(record, checkpoint_path, run_id)
            completed.add((mol_id, repr_name))

    if not dry_run:
        logger.info(f"\nBenchmark 1 complete for {model_id} | {thinking_str}")
        logger.info(f"Results saved to {checkpoint_path}")


# ============================================================================
# Benchmark Stubs (2-7)
# ============================================================================


def run_benchmark_2(
    model_config: Dict[str, Any],
    thinking: bool,
    vllm_url: str,
    num_samples: Optional[int] = None,
    dry_run: bool = False,
    use_direct_vllm: bool = False,
    batch_size: int = 50,
    run_id: int = 1,
):
    """
    Run Benchmark 2: Functional Group Identification

    For each molecule in comprehension subset:
    - For each of 10 functional groups:
        - Build prompt, call model, parse yes/no
        - Save result

    Args:
        model_config: Model configuration dict
        thinking: Whether thinking is enabled
        vllm_url: vLLM API URL
        num_samples: If provided, limit to first N molecules
        dry_run: If True, print prompts without calling API
    """
    from config import FUNCTIONAL_GROUPS

    benchmark_name = "functional_groups"
    model_id = model_config["id"]
    thinking_str = "thinking_on" if thinking else "thinking_off"

    logger.info(f"\n{'='*80}")
    logger.info(f"Starting Benchmark 2: Functional Group Identification")
    logger.info(f"Model: {model_id} | Thinking: {thinking_str}")
    logger.info(f"{'='*80}\n")

    # Setup checkpoint
    checkpoint_dir = RESULTS_DIR / "raw"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"benchmark_2_{model_id}_{thinking_str}_run{run_id}.jsonl"

    # Load completed items
    completed = load_checkpoint(checkpoint_path, compound_key_field="group")

    # Load data
    subset_df = pd.read_csv(DATA_DIR / "comprehension_subset_ids.csv")
    test_df, id_col, smiles_col = load_test_dataset(DATA_DIR / "chebi20_test.csv")

    # Get molecules in comprehension subset
    molecule_ids = subset_df["molecule_id"].tolist()
    if num_samples:
        molecule_ids = molecule_ids[:num_samples]

    logger.info(f"Processing {len(molecule_ids)} molecules")

    # Counter for dry run
    dry_run_count = 0
    dry_run_limit = 3

    # Phase 1: Pre-collect all pending tasks
    from rdkit import Chem
    logger.info("Pre-collecting tasks...")
    all_tasks = []
    for mol_id in molecule_ids:
        mol_row = test_df[test_df[id_col] == mol_id].iloc[0]
        smiles = mol_row[smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES for {mol_id}: {smiles}")
            continue
        ground_truths = {fg_key: chemistry.has_substructure(mol, fg_config["smarts"])
                         for fg_key, fg_config in FUNCTIONAL_GROUPS.items()}
        for fg_key, fg_config in FUNCTIONAL_GROUPS.items():
            fg_name = fg_config["name"]
            ground_truth = ground_truths[fg_key]
            for repr_name in REPRESENTATIONS:
                if (mol_id, f"{fg_key}_{repr_name}") in completed:
                    continue
                if repr_name == "iupac":
                    repr_string = mol_row["iupac"] if "iupac" in mol_row and pd.notna(mol_row["iupac"]) else None
                elif repr_name == "selfies":
                    repr_string = mol_row["selfies"] if "selfies" in mol_row and pd.notna(mol_row["selfies"]) else (
                        mol_row["SELFIES"] if "SELFIES" in mol_row and pd.notna(mol_row["SELFIES"]) else None)
                else:
                    repr_string = representations.convert_to_representation(smiles, repr_name)
                if repr_string is None:
                    logger.warning(f"Conversion failed: {mol_id} -> {repr_name}")
                    continue
                prompt_text = prompts.create_functional_group_prompt(repr_string, fg_name, repr_name)
                all_tasks.append({
                    "messages": [{"role": "user", "content": prompt_text}],
                    "mol_id": mol_id, "repr_name": repr_name,
                    "fg_key": fg_key, "ground_truth": ground_truth,
                })

    logger.info(f"Total pending tasks: {len(all_tasks)}")

    gen_params = get_generation_params(model_config, "functional_groups", thinking)
    stop_seqs = STOP_SEQUENCES.get("comprehension", [])

    if dry_run:
        for task in all_tasks[:dry_run_limit]:
            call_model(messages=task["messages"], model_config=model_config, thinking=thinking,
                       vllm_url=vllm_url, generation_params=gen_params, stop_sequences=stop_seqs, dry_run=True)
        logger.info(f"\nDry run limit reached ({dry_run_limit} prompts)")
        return

    # Phase 2: Process in batches
    for batch_start in tqdm(range(0, len(all_tasks), batch_size), desc="Batches"):
        batch = all_tasks[batch_start:batch_start + batch_size]
        messages_list = [t["messages"] for t in batch]

        if use_direct_vllm:
            results = call_model_direct_vllm_batch(
                messages_list=messages_list, model_config=model_config, thinking=thinking,
                generation_params=gen_params, stop_sequences=stop_seqs,
            )
        else:
            with ThreadPoolExecutor(max_workers=min(len(batch), 32)) as executor:
                future_to_idx = {
                    executor.submit(call_model, t["messages"], model_config, thinking,
                                    vllm_url, gen_params, stop_seqs): i
                    for i, t in enumerate(batch)
                }
                results = [None] * len(batch)
                for future in as_completed(future_to_idx):
                    results[future_to_idx[future]] = future.result()

        for task, result in zip(batch, results):
            mol_id, repr_name = task["mol_id"], task["repr_name"]
            fg_key, ground_truth = task["fg_key"], task["ground_truth"]
            if result is None:
                logger.warning(f"API call failed: {mol_id} | {fg_key} | {repr_name} | {model_id} | {thinking_str}")
                record = {
                    "molecule_id": mol_id, "representation": repr_name, "model": model_id,
                    "thinking": thinking, "group": fg_key, "raw_response": None,
                    "parsed_answer": None, "predicted": None, "ground_truth": ground_truth, "latency_ms": None,
                }
            else:
                predicted = parsing.extract_yes_no(result["parsed_answer"])
                record = {
                    "molecule_id": mol_id, "representation": repr_name, "model": model_id,
                    "thinking": thinking, "group": fg_key, "raw_response": result["raw_response"],
                    "parsed_answer": result["parsed_answer"], "predicted": predicted,
                    "ground_truth": ground_truth, "latency_ms": result["latency_ms"],
                }
            save_to_checkpoint(record, checkpoint_path, run_id)
            completed.add((mol_id, f"{fg_key}_{repr_name}"))

    if not dry_run:
        logger.info(f"\nBenchmark 2 complete for {model_id} | {thinking_str}")
        logger.info(f"Results saved to {checkpoint_path}")


def run_benchmark_3(
    model_config: Dict[str, Any],
    thinking: bool,
    vllm_url: str,
    num_samples: Optional[int] = None,
    dry_run: bool = False,
    use_direct_vllm: bool = False,
    batch_size: int = 50,
    run_id: int = 1,
):
    """
    Run Benchmark 3: Molecular Property Estimation

    For each molecule in comprehension subset:
    - For each of 4 properties (LogP, TPSA, HBD, HBA):
        - Build prompt, call model, parse number
        - Save result

    Args:
        model_config: Model configuration dict
        thinking: Whether thinking is enabled
        vllm_url: vLLM API URL
        num_samples: If provided, limit to first N molecules
        dry_run: If True, print prompts without calling API
    """
    from config import PROPERTIES

    benchmark_name = "property_estimation"
    model_id = model_config["id"]
    thinking_str = "thinking_on" if thinking else "thinking_off"

    logger.info(f"\n{'='*80}")
    logger.info(f"Starting Benchmark 3: Molecular Property Estimation")
    logger.info(f"Model: {model_id} | Thinking: {thinking_str}")
    logger.info(f"{'='*80}\n")

    # Setup checkpoint
    checkpoint_dir = RESULTS_DIR / "raw"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"benchmark_3_{model_id}_{thinking_str}_run{run_id}.jsonl"

    # Load completed items
    completed = load_checkpoint(checkpoint_path, compound_key_field="property")

    # Load data
    subset_df = pd.read_csv(DATA_DIR / "comprehension_subset_ids.csv")
    test_df, id_col, smiles_col = load_test_dataset(DATA_DIR / "chebi20_test.csv")

    # Get molecules in comprehension subset
    molecule_ids = subset_df["molecule_id"].tolist()
    if num_samples:
        molecule_ids = molecule_ids[:num_samples]

    logger.info(f"Processing {len(molecule_ids)} molecules")

    dry_run_limit = 3

    # Phase 1: Pre-collect all pending tasks
    from rdkit import Chem
    logger.info("Pre-collecting tasks...")
    all_tasks = []
    for mol_id in molecule_ids:
        mol_row = test_df[test_df[id_col] == mol_id].iloc[0]
        smiles = mol_row[smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES for {mol_id}: {smiles}")
            continue
        props = chemistry.calculate_all_properties(mol)
        for prop_key, prop_config in PROPERTIES.items():
            prop_name = prop_config["name"]
            ground_truth = props.get(prop_key)
            if ground_truth is None:
                logger.warning(f"Failed to calculate {prop_key} for {mol_id}")
                continue
            for repr_name in REPRESENTATIONS:
                if (mol_id, f"{prop_key}_{repr_name}") in completed:
                    continue
                if repr_name == "iupac":
                    repr_string = mol_row["iupac"] if "iupac" in mol_row and pd.notna(mol_row["iupac"]) else None
                elif repr_name == "selfies":
                    repr_string = mol_row["selfies"] if "selfies" in mol_row and pd.notna(mol_row["selfies"]) else (
                        mol_row["SELFIES"] if "SELFIES" in mol_row and pd.notna(mol_row["SELFIES"]) else None)
                else:
                    repr_string = representations.convert_to_representation(smiles, repr_name)
                if repr_string is None:
                    logger.warning(f"Conversion failed: {mol_id} -> {repr_name}")
                    continue
                prompt_text = prompts.create_property_estimation_prompt(repr_string, prop_name, repr_name)
                all_tasks.append({
                    "messages": [{"role": "user", "content": prompt_text}],
                    "mol_id": mol_id, "repr_name": repr_name,
                    "prop_key": prop_key, "ground_truth": ground_truth,
                })

    logger.info(f"Total pending tasks: {len(all_tasks)}")

    gen_params = get_generation_params(model_config, "property_estimation", thinking)
    stop_seqs = STOP_SEQUENCES.get("comprehension", [])

    if dry_run:
        for task in all_tasks[:dry_run_limit]:
            call_model(messages=task["messages"], model_config=model_config, thinking=thinking,
                       vllm_url=vllm_url, generation_params=gen_params, stop_sequences=stop_seqs, dry_run=True)
        logger.info(f"\nDry run limit reached ({dry_run_limit} prompts)")
        return

    # Phase 2: Process in batches
    for batch_start in tqdm(range(0, len(all_tasks), batch_size), desc="Batches"):
        batch = all_tasks[batch_start:batch_start + batch_size]
        messages_list = [t["messages"] for t in batch]

        if use_direct_vllm:
            results = call_model_direct_vllm_batch(
                messages_list=messages_list, model_config=model_config, thinking=thinking,
                generation_params=gen_params, stop_sequences=stop_seqs,
            )
        else:
            with ThreadPoolExecutor(max_workers=min(len(batch), 32)) as executor:
                future_to_idx = {
                    executor.submit(call_model, t["messages"], model_config, thinking,
                                    vllm_url, gen_params, stop_seqs): i
                    for i, t in enumerate(batch)
                }
                results = [None] * len(batch)
                for future in as_completed(future_to_idx):
                    results[future_to_idx[future]] = future.result()

        for task, result in zip(batch, results):
            mol_id, repr_name = task["mol_id"], task["repr_name"]
            prop_key, ground_truth = task["prop_key"], task["ground_truth"]
            if result is None:
                logger.warning(f"API call failed: {mol_id} | {prop_key} | {repr_name} | {model_id} | {thinking_str}")
                record = {
                    "molecule_id": mol_id, "representation": repr_name, "model": model_id,
                    "thinking": thinking, "property": prop_key, "raw_response": None,
                    "parsed_answer": None, "predicted": None, "ground_truth": ground_truth, "latency_ms": None,
                }
            else:
                if prop_key in ["logp", "tpsa"]:
                    predicted = parsing.extract_float(result["parsed_answer"])
                else:
                    predicted = parsing.extract_integer(result["parsed_answer"])
                record = {
                    "molecule_id": mol_id, "representation": repr_name, "model": model_id,
                    "thinking": thinking, "property": prop_key, "raw_response": result["raw_response"],
                    "parsed_answer": result["parsed_answer"], "predicted": predicted,
                    "ground_truth": ground_truth, "latency_ms": result["latency_ms"],
                }
            save_to_checkpoint(record, checkpoint_path, run_id)
            completed.add((mol_id, f"{prop_key}_{repr_name}"))

    if not dry_run:
        logger.info(f"\nBenchmark 3 complete for {model_id} | {thinking_str}")
        logger.info(f"Results saved to {checkpoint_path}")


def run_benchmark_4(
    model_config: Dict[str, Any],
    thinking: bool,
    vllm_url: str,
    num_samples: Optional[int] = None,
    dry_run: bool = False,
    use_direct_vllm: bool = False,
    batch_size: int = 50,
    run_id: int = 1,
):
    """
    Run Benchmark 4: Molecule Retrieval / Discrimination

    For each molecule in comprehension subset:
    - Get 3 distractors, shuffle into A-D positions
    - Build 4-choice prompt, call model, parse letter
    - Save result

    Args:
        model_config: Model configuration dict
        thinking: Whether thinking is enabled
        vllm_url: vLLM API URL
        num_samples: If provided, limit to first N molecules
        dry_run: If True, print prompts without calling API
    """
    benchmark_name = "retrieval"
    model_id = model_config["id"]
    thinking_str = "thinking_on" if thinking else "thinking_off"

    logger.info(f"\n{'='*80}")
    logger.info(f"Starting Benchmark 4: Molecule Retrieval")
    logger.info(f"Model: {model_id} | Thinking: {thinking_str}")
    logger.info(f"{'='*80}\n")

    # Setup checkpoint
    checkpoint_dir = RESULTS_DIR / "raw"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"benchmark_4_{model_id}_{thinking_str}_run{run_id}.jsonl"

    # Load completed items
    completed = load_checkpoint(checkpoint_path)

    # Load data
    subset_df = pd.read_csv(DATA_DIR / "comprehension_subset_ids.csv")
    test_df, id_col, smiles_col = load_test_dataset(DATA_DIR / "chebi20_test.csv")
    distractors_df = pd.read_csv(DATA_DIR / "retrieval_distractors.csv")

    # Get molecules in comprehension subset
    molecule_ids = subset_df["molecule_id"].tolist()
    if num_samples:
        molecule_ids = molecule_ids[:num_samples]

    logger.info(f"Processing {len(molecule_ids)} molecules")

    dry_run_limit = 3

    # Phase 1: Pre-collect all pending tasks
    logger.info("Pre-collecting tasks...")
    all_tasks = []
    for mol_id in molecule_ids:
        mol_row = test_df[test_df[id_col] == mol_id].iloc[0]
        description = mol_row["description"]
        correct_smiles = mol_row[smiles_col]
        distractor_rows = distractors_df[distractors_df["target_molecule_id"] == mol_id]
        if len(distractor_rows) == 0:
            logger.warning(f"No distractors found for {mol_id}")
            continue
        distractor_ids = distractor_rows["distractor_molecule_id"].tolist()
        distractor_mol_rows = []
        for dist_id in distractor_ids:
            dist_row = test_df[test_df[id_col] == dist_id]
            if len(dist_row) == 0:
                logger.warning(f"Distractor {dist_id} not found in dataset")
                break
            distractor_mol_rows.append(dist_row.iloc[0])
        if len(distractor_mol_rows) != 3:
            logger.warning(f"Expected 3 distractors for {mol_id}, got {len(distractor_mol_rows)}")
            continue

        def get_repr_from_row(row, repr_name, smiles):
            """Look up pre-computed representation from a dataset row, falling back to conversion."""
            if repr_name == "iupac":
                return row["iupac"] if "iupac" in row and pd.notna(row["iupac"]) else None
            elif repr_name == "selfies":
                if "selfies" in row and pd.notna(row["selfies"]):
                    return row["selfies"]
                elif "SELFIES" in row and pd.notna(row["SELFIES"]):
                    return row["SELFIES"]
                return None
            else:
                return representations.convert_to_representation(smiles, repr_name)

        for repr_name in REPRESENTATIONS:
            if (mol_id, repr_name) in completed:
                continue
            correct_repr = get_repr_from_row(mol_row, repr_name, correct_smiles)
            if correct_repr is None:
                logger.warning(f"Conversion failed: {mol_id} -> {repr_name}")
                continue
            distractor_reprs = []
            for dist_row in distractor_mol_rows:
                dist_repr = get_repr_from_row(dist_row, repr_name, dist_row[smiles_col])
                if dist_repr is None:
                    break
                distractor_reprs.append(dist_repr)
            if len(distractor_reprs) != 3:
                logger.warning(f"Failed to convert distractors for {mol_id} -> {repr_name}")
                continue
            choices = [correct_repr] + distractor_reprs
            random.seed(hash(f"{mol_id}_{repr_name}") % (2**32))
            random.shuffle(choices)
            correct_letter = "ABCD"[choices.index(correct_repr)]
            molecules_dict = {letter: mol for letter, mol in zip("ABCD", choices)}
            prompt_text = prompts.create_retrieval_prompt(description, molecules_dict, repr_name)
            all_tasks.append({
                "messages": [{"role": "user", "content": prompt_text}],
                "mol_id": mol_id, "repr_name": repr_name, "correct_letter": correct_letter,
            })

    logger.info(f"Total pending tasks: {len(all_tasks)}")

    gen_params = get_generation_params(model_config, "retrieval", thinking)
    stop_seqs = STOP_SEQUENCES.get("retrieval", [])

    if dry_run:
        for task in all_tasks[:dry_run_limit]:
            call_model(messages=task["messages"], model_config=model_config, thinking=thinking,
                       vllm_url=vllm_url, generation_params=gen_params, stop_sequences=stop_seqs, dry_run=True)
        logger.info(f"\nDry run limit reached ({dry_run_limit} prompts)")
        return

    # Phase 2: Process in batches
    for batch_start in tqdm(range(0, len(all_tasks), batch_size), desc="Batches"):
        batch = all_tasks[batch_start:batch_start + batch_size]
        messages_list = [t["messages"] for t in batch]

        if use_direct_vllm:
            results = call_model_direct_vllm_batch(
                messages_list=messages_list, model_config=model_config, thinking=thinking,
                generation_params=gen_params, stop_sequences=stop_seqs,
            )
        else:
            with ThreadPoolExecutor(max_workers=min(len(batch), 32)) as executor:
                future_to_idx = {
                    executor.submit(call_model, t["messages"], model_config, thinking,
                                    vllm_url, gen_params, stop_seqs): i
                    for i, t in enumerate(batch)
                }
                results = [None] * len(batch)
                for future in as_completed(future_to_idx):
                    results[future_to_idx[future]] = future.result()

        for task, result in zip(batch, results):
            mol_id, repr_name = task["mol_id"], task["repr_name"]
            correct_letter = task["correct_letter"]
            if result is None:
                logger.warning(f"API call failed: {mol_id} | {repr_name} | {model_id} | {thinking_str}")
                record = {
                    "molecule_id": mol_id, "representation": repr_name, "model": model_id,
                    "thinking": thinking, "raw_response": None, "parsed_answer": None,
                    "predicted_letter": None, "correct_letter": correct_letter, "latency_ms": None,
                }
            else:
                predicted_letter = parsing.extract_multiple_choice(result["parsed_answer"])
                record = {
                    "molecule_id": mol_id, "representation": repr_name, "model": model_id,
                    "thinking": thinking, "raw_response": result["raw_response"],
                    "parsed_answer": result["parsed_answer"], "predicted_letter": predicted_letter,
                    "correct_letter": correct_letter, "latency_ms": result["latency_ms"],
                }
            save_to_checkpoint(record, checkpoint_path, run_id)
            completed.add((mol_id, repr_name))

    if not dry_run:
        logger.info(f"\nBenchmark 4 complete for {model_id} | {thinking_str}")
        logger.info(f"Results saved to {checkpoint_path}")


def run_benchmark_5(
    model_config: Dict[str, Any],
    thinking: bool,
    vllm_url: str,
    num_samples: Optional[int] = None,
    dry_run: bool = False,
    use_direct_vllm: bool = False,
    batch_size: int = 50,
    run_id: int = 1,
):
    """
    Run Benchmark 5: Isomer / Identity Discrimination

    For each pair in isomer_pairs.csv:
    - Convert both molecules to target representation
    - Build prompt, call model, parse yes/no
    - Save result

    Args:
        model_config: Model configuration dict
        thinking: Whether thinking is enabled
        vllm_url: vLLM API URL
        num_samples: If provided, limit to first N pairs
        dry_run: If True, print prompts without calling API
    """
    benchmark_name = "isomer_discrimination"
    model_id = model_config["id"]
    thinking_str = "thinking_on" if thinking else "thinking_off"

    logger.info(f"\n{'='*80}")
    logger.info(f"Starting Benchmark 5: Isomer Discrimination")
    logger.info(f"Model: {model_id} | Thinking: {thinking_str}")
    logger.info(f"{'='*80}\n")

    # Setup checkpoint
    checkpoint_dir = RESULTS_DIR / "raw"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"benchmark_5_{model_id}_{thinking_str}_run{run_id}.jsonl"

    # Load completed items
    completed = load_checkpoint_benchmark_5(checkpoint_path)

    # Load data
    pairs_df = pd.read_csv(DATA_DIR / "isomer_pairs.csv")
    test_df, id_col, smiles_col = load_test_dataset(DATA_DIR / "chebi20_test.csv")

    # Limit samples if requested
    if num_samples:
        pairs_df = pairs_df.head(num_samples)

    logger.info(f"Processing {len(pairs_df)} pairs")

    dry_run_limit = 3

    # Phase 1: Pre-collect all pending tasks
    logger.info("Pre-collecting tasks...")
    all_tasks = []
    for idx, row in pairs_df.iterrows():
        pair_id = idx
        mol_id_1 = row["molecule_id_1"]
        mol_id_2 = row["molecule_id_2"]
        smiles1 = row["smiles_1"]
        smiles2 = row["smiles_2"]
        pair_type = row["type"] if "type" in row else row.get("pair_type", "unknown")
        ground_truth = pair_type in ("natural", "stereoisomer")
        mol_row_1 = test_df[test_df[id_col] == mol_id_1]
        mol_row_2 = test_df[test_df[id_col] == mol_id_2]

        for repr_name in REPRESENTATIONS:
            if (pair_id, repr_name) in completed:
                continue
            if repr_name == "iupac":
                # Use pre-computed iupac_1/iupac_2 columns (populated by script 01 from
                # ChEBI-20's native iupacname column). Fall back to test_df lookup for
                # older CSVs that predate this column. Never call PubChem at inference time.
                repr1 = row.get("iupac_1") if pd.notna(row.get("iupac_1", None)) else None
                repr2 = row.get("iupac_2") if pd.notna(row.get("iupac_2", None)) else None
                if repr1 is None and len(mol_row_1) > 0:
                    r1 = mol_row_1.iloc[0]
                    repr1 = r1.get("iupac") if pd.notna(r1.get("iupac", None)) else None
                if repr2 is None and len(mol_row_2) > 0:
                    r2 = mol_row_2.iloc[0]
                    repr2 = r2.get("iupac") if pd.notna(r2.get("iupac", None)) else None
            elif repr_name == "selfies":
                repr1 = row.get("selfies_1") if pd.notna(row.get("selfies_1", None)) else None
                repr2 = row.get("selfies_2") if pd.notna(row.get("selfies_2", None)) else None
                if repr1 is None and len(mol_row_1) > 0:
                    r1 = mol_row_1.iloc[0]
                    repr1 = r1.get("selfies") if pd.notna(r1.get("selfies", None)) else (
                        r1.get("SELFIES") if pd.notna(r1.get("SELFIES", None)) else None)
                if repr2 is None and len(mol_row_2) > 0:
                    r2 = mol_row_2.iloc[0]
                    repr2 = r2.get("selfies") if pd.notna(r2.get("selfies", None)) else (
                        r2.get("SELFIES") if pd.notna(r2.get("SELFIES", None)) else None)
            else:
                repr1 = representations.convert_to_representation(smiles1, repr_name)
                repr2 = representations.convert_to_representation(smiles2, repr_name)

            if repr1 is None or repr2 is None:
                logger.warning(f"Conversion failed for pair {pair_id} -> {repr_name}")
                continue
            prompt_text = prompts.create_isomer_discrimination_prompt(repr1, repr2, repr_name)
            all_tasks.append({
                "messages": [{"role": "user", "content": prompt_text}],
                "pair_id": pair_id, "repr_name": repr_name,
                "pair_type": pair_type, "ground_truth": ground_truth,
            })

    logger.info(f"Total pending tasks: {len(all_tasks)}")

    gen_params = get_generation_params(model_config, "isomer_discrimination", thinking)
    stop_seqs = STOP_SEQUENCES.get("comprehension", [])

    if dry_run:
        for task in all_tasks[:dry_run_limit]:
            call_model(messages=task["messages"], model_config=model_config, thinking=thinking,
                       vllm_url=vllm_url, generation_params=gen_params, stop_sequences=stop_seqs, dry_run=True)
        logger.info(f"\nDry run limit reached ({dry_run_limit} prompts)")
        return

    # Phase 2: Process in batches
    for batch_start in tqdm(range(0, len(all_tasks), batch_size), desc="Batches"):
        batch = all_tasks[batch_start:batch_start + batch_size]
        messages_list = [t["messages"] for t in batch]

        if use_direct_vllm:
            results = call_model_direct_vllm_batch(
                messages_list=messages_list, model_config=model_config, thinking=thinking,
                generation_params=gen_params, stop_sequences=stop_seqs,
            )
        else:
            with ThreadPoolExecutor(max_workers=min(len(batch), 32)) as executor:
                future_to_idx = {
                    executor.submit(call_model, t["messages"], model_config, thinking,
                                    vllm_url, gen_params, stop_seqs): i
                    for i, t in enumerate(batch)
                }
                results = [None] * len(batch)
                for future in as_completed(future_to_idx):
                    results[future_to_idx[future]] = future.result()

        for task, result in zip(batch, results):
            pair_id, repr_name = task["pair_id"], task["repr_name"]
            pair_type, ground_truth = task["pair_type"], task["ground_truth"]
            if result is None:
                logger.warning(f"API call failed: pair {pair_id} | {repr_name} | {model_id} | {thinking_str}")
                record = {
                    "pair_id": pair_id, "representation": repr_name, "model": model_id,
                    "thinking": thinking, "pair_type": pair_type, "raw_response": None,
                    "parsed_answer": None, "predicted": None, "ground_truth": ground_truth, "latency_ms": None,
                }
            else:
                predicted = parsing.extract_yes_no(result["parsed_answer"])
                record = {
                    "pair_id": pair_id, "representation": repr_name, "model": model_id,
                    "thinking": thinking, "pair_type": pair_type, "raw_response": result["raw_response"],
                    "parsed_answer": result["parsed_answer"], "predicted": predicted,
                    "ground_truth": ground_truth, "latency_ms": result["latency_ms"],
                }
            save_to_checkpoint(record, checkpoint_path, run_id)
            completed.add((pair_id, repr_name))

    if not dry_run:
        logger.info(f"\nBenchmark 5 complete for {model_id} | {thinking_str}")
        logger.info(f"Results saved to {checkpoint_path}")


def run_benchmark_6(
    model_config: Dict[str, Any],
    thinking: bool,
    vllm_url: str,
    num_samples: Optional[int] = None,
    dry_run: bool = False,
    use_direct_vllm: bool = False,
    batch_size: int = 50,
    run_id: int = 1,
):
    """
    Run Benchmark 6: Caption-to-Molecule Generation

    For each molecule in FULL test set:
    - Get 2-shot examples from fewshot_examples.csv
    - Build generation prompt, call model
    - Parse molecule string from output
    - Save result

    Args:
        model_config: Model configuration dict
        thinking: Whether thinking is enabled
        vllm_url: vLLM API URL
        num_samples: If provided, limit to first N molecules
        dry_run: If True, print prompts without calling API
    """
    benchmark_name = "generation"
    model_id = model_config["id"]
    thinking_str = "thinking_on" if thinking else "thinking_off"

    logger.info(f"\n{'='*80}")
    logger.info(f"Starting Benchmark 6: Caption-to-Molecule Generation")
    logger.info(f"Model: {model_id} | Thinking: {thinking_str}")
    logger.info(f"{'='*80}\n")

    # Setup checkpoint
    checkpoint_dir = RESULTS_DIR / "raw"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"benchmark_6_{model_id}_{thinking_str}_run{run_id}.jsonl"

    # Load completed items
    completed = load_checkpoint(checkpoint_path)

    # Load data - USE CHEBI DATASET for generation (has real descriptions)
    test_df, id_col, smiles_col = load_test_dataset(DATA_DIR / "chebi20_test.csv")
    # fewshot_df = pd.read_csv(DATA_DIR / "fewshot_examples.csv")

    # Apply cap: explicit --num_samples overrides, else use GENERATION_SAMPLE_SIZE from config
    cap = num_samples or GENERATION_SAMPLE_SIZE
    if cap:
        test_df = test_df.head(cap)

    logger.info(f"Processing {len(test_df)} molecules")

    dry_run_limit = 3

    # Phase 1: Pre-collect all pending tasks
    logger.info("Pre-collecting tasks...")
    all_tasks = []
    for idx, row in test_df.iterrows():
        mol_id = row[id_col]
        description = row["description"]
        for repr_name in REPRESENTATIONS:
            if (mol_id, repr_name) in completed:
                continue
            json_schema = get_json_schema_for_representation(repr_name)
            prompt_text = prompts.create_generation_prompt(
                description, repr_name, few_shot_examples=None,
                use_json_format=(json_schema is not None)
            )
            all_tasks.append({
                "messages": [{"role": "user", "content": prompt_text}],
                "mol_id": mol_id, "repr_name": repr_name,
                "description": description, "json_schema": json_schema,
            })

    logger.info(f"Total pending tasks: {len(all_tasks)}")

    gen_params = get_generation_params(model_config, "generation", thinking)
    stop_seqs = STOP_SEQUENCES.get("generation", [])

    if dry_run:
        for task in all_tasks[:dry_run_limit]:
            call_model(messages=task["messages"], model_config=model_config, thinking=thinking,
                       vllm_url=vllm_url, generation_params=gen_params, stop_sequences=stop_seqs,
                       json_schema=task["json_schema"], dry_run=True)
        logger.info(f"\nDry run limit reached ({dry_run_limit} prompts)")
        return

    # Phase 2: Process in batches
    for batch_start in tqdm(range(0, len(all_tasks), batch_size), desc="Batches"):
        batch = all_tasks[batch_start:batch_start + batch_size]
        messages_list = [t["messages"] for t in batch]

        if use_direct_vllm:
            # Sub-batch by json_schema so structured output is applied per group
            schema_groups = {}
            for i, t in enumerate(batch):
                key = t["json_schema"]["name"] if t["json_schema"] else None
                schema_groups.setdefault(key, []).append((i, t))

            results = [None] * len(batch)
            for key, group in schema_groups.items():
                group_indices, group_tasks = zip(*group)
                group_schema = group_tasks[0]["json_schema"]
                group_messages = [t["messages"] for t in group_tasks]
                group_results = call_model_direct_vllm_batch(
                    messages_list=group_messages, model_config=model_config, thinking=thinking,
                    generation_params=gen_params, stop_sequences=stop_seqs,
                    json_schema=group_schema,
                )
                for idx, res in zip(group_indices, group_results):
                    results[idx] = res
        else:
            with ThreadPoolExecutor(max_workers=min(len(batch), 32)) as executor:
                future_to_idx = {
                    executor.submit(call_model, t["messages"], model_config, thinking,
                                    vllm_url, gen_params, stop_seqs, t["json_schema"]): i
                    for i, t in enumerate(batch)
                }
                results = [None] * len(batch)
                for future in as_completed(future_to_idx):
                    results[future_to_idx[future]] = future.result()

        for task, result in zip(batch, results):
            mol_id, repr_name = task["mol_id"], task["repr_name"]
            description = task["description"]
            if result is None:
                logger.warning(f"API call failed: {mol_id} | {repr_name} | {model_id} | {thinking_str}")
                record = {
                    "molecule_id": mol_id, "representation": repr_name, "model": model_id,
                    "thinking": thinking, "description": description, "raw_response": None,
                    "parsed_answer": None, "generated_string": None, "latency_ms": None,
                }
            else:
                # Apply per-task json_schema parsing if needed
                parsed_answer = result["parsed_answer"]
                json_schema = task["json_schema"]
                if json_schema is not None:
                    try:
                        json_obj = json.loads(parsed_answer)
                        if "molecule" in json_obj:
                            parsed_answer = json_obj["molecule"]
                        elif "atoms" in json_obj and "bonds" in json_obj:
                            parsed_answer = json.dumps(json_obj)
                    except (json.JSONDecodeError, TypeError):
                        pass
                generated_string = parsing.extract_molecule_string(parsed_answer, repr_name)
                record = {
                    "molecule_id": mol_id, "representation": repr_name, "model": model_id,
                    "thinking": thinking, "description": description,
                    "raw_response": result["raw_response"], "parsed_answer": parsed_answer,
                    "generated_string": generated_string, "latency_ms": result["latency_ms"],
                }
            save_to_checkpoint(record, checkpoint_path, run_id)
            completed.add((mol_id, repr_name))

    if not dry_run:
        logger.info(f"\nBenchmark 6 complete for {model_id} | {thinking_str}")
        logger.info(f"Results saved to {checkpoint_path}")


def run_benchmark_7(
    model_config: Dict[str, Any],
    thinking: bool,
    vllm_url: str,
    num_samples: Optional[int] = None,
    dry_run: bool = False,
    use_direct_vllm: bool = False,
    batch_size: int = 50,
    run_id: int = 1,
):
    """
    Run Benchmark 7: Molecular Completion / Infilling

    For each molecule in FULL test set:
    - Get partial string (first 50%) from completion_partials.csv
    - Build completion prompt, call model
    - Parse completed molecule string
    - Save result

    Args:
        model_config: Model configuration dict
        thinking: Whether thinking is enabled
        vllm_url: vLLM API URL
        num_samples: If provided, limit to first N molecules
        dry_run: If True, print prompts without calling API
    """
    benchmark_name = "completion"
    model_id = model_config["id"]
    thinking_str = "thinking_on" if thinking else "thinking_off"

    logger.info(f"\n{'='*80}")
    logger.info(f"Starting Benchmark 7: Molecular Completion")
    logger.info(f"Model: {model_id} | Thinking: {thinking_str}")
    logger.info(f"{'='*80}\n")

    # Setup checkpoint
    checkpoint_dir = RESULTS_DIR / "raw"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"benchmark_7_{model_id}_{thinking_str}_run{run_id}.jsonl"

    # Load completed items
    completed = load_checkpoint(checkpoint_path)

    # Load data - USE ZINC DATASET for completion (has matching partials)
    test_df, id_col, smiles_col = load_test_dataset(DATA_DIR / "chebi20_test.csv")
    partials_df = pd.read_csv(DATA_DIR / "completion_partials.csv")

    # Apply cap: explicit --num_samples overrides, else use GENERATION_SAMPLE_SIZE from config
    cap = num_samples or GENERATION_SAMPLE_SIZE
    if cap:
        test_df = test_df.head(cap)

    logger.info(f"Processing {len(test_df)} molecules")

    dry_run_limit = 3

    # Phase 1: Pre-collect all pending tasks
    logger.info("Pre-collecting tasks...")
    all_tasks = []
    # CML is excluded from completion — XML cannot be meaningfully split at 50%
    b7_representations = [r for r in REPRESENTATIONS if r != "cml"]
    for idx, row in test_df.iterrows():
        mol_id = row[id_col]
        for repr_name in b7_representations:
            if (mol_id, repr_name) in completed:
                continue
            partial_row = partials_df[
                (partials_df["molecule_id"] == mol_id) &
                (partials_df["representation"] == repr_name)
            ]
            if len(partial_row) == 0:
                logger.warning(f"No partial string for {mol_id} -> {repr_name}")
                continue
            partial_string = partial_row.iloc[0]["partial_string"]
            json_schema = get_json_schema_for_representation(repr_name)
            prompt_text = prompts.create_completion_prompt(
                partial_string, repr_name, use_json_format=(json_schema is not None)
            )
            all_tasks.append({
                "messages": [{"role": "user", "content": prompt_text}],
                "mol_id": mol_id, "repr_name": repr_name,
                "partial_string": partial_string, "json_schema": json_schema,
            })

    logger.info(f"Total pending tasks: {len(all_tasks)}")

    gen_params = get_generation_params(model_config, "completion", thinking)
    stop_seqs = STOP_SEQUENCES.get("completion", [])

    if dry_run:
        for task in all_tasks[:dry_run_limit]:
            call_model(messages=task["messages"], model_config=model_config, thinking=thinking,
                       vllm_url=vllm_url, generation_params=gen_params, stop_sequences=stop_seqs,
                       json_schema=task["json_schema"], dry_run=True)
        logger.info(f"\nDry run limit reached ({dry_run_limit} prompts)")
        return

    # Phase 2: Process in batches
    for batch_start in tqdm(range(0, len(all_tasks), batch_size), desc="Batches"):
        batch = all_tasks[batch_start:batch_start + batch_size]
        messages_list = [t["messages"] for t in batch]

        if use_direct_vllm:
            results = call_model_direct_vllm_batch(
                messages_list=messages_list, model_config=model_config, thinking=thinking,
                generation_params=gen_params, stop_sequences=stop_seqs,
            )
        else:
            with ThreadPoolExecutor(max_workers=min(len(batch), 32)) as executor:
                future_to_idx = {
                    executor.submit(call_model, t["messages"], model_config, thinking,
                                    vllm_url, gen_params, stop_seqs, t["json_schema"]): i
                    for i, t in enumerate(batch)
                }
                results = [None] * len(batch)
                for future in as_completed(future_to_idx):
                    results[future_to_idx[future]] = future.result()

        for task, result in zip(batch, results):
            mol_id, repr_name = task["mol_id"], task["repr_name"]
            partial_string = task["partial_string"]
            if result is None:
                logger.warning(f"API call failed: {mol_id} | {repr_name} | {model_id} | {thinking_str}")
                record = {
                    "molecule_id": mol_id, "representation": repr_name, "model": model_id,
                    "thinking": thinking, "partial_string": partial_string, "raw_response": None,
                    "parsed_answer": None, "generated_string": None, "latency_ms": None,
                }
            else:
                parsed_answer = result["parsed_answer"]
                json_schema = task["json_schema"]
                if json_schema is not None:
                    try:
                        json_obj = json.loads(parsed_answer)
                        if "molecule" in json_obj:
                            parsed_answer = json_obj["molecule"]
                        elif "atoms" in json_obj and "bonds" in json_obj:
                            parsed_answer = json.dumps(json_obj)
                    except (json.JSONDecodeError, TypeError):
                        pass
                generated_string = parsing.extract_molecule_string(parsed_answer, repr_name)
                record = {
                    "molecule_id": mol_id, "representation": repr_name, "model": model_id,
                    "thinking": thinking, "partial_string": partial_string,
                    "raw_response": result["raw_response"], "parsed_answer": parsed_answer,
                    "generated_string": generated_string, "latency_ms": result["latency_ms"],
                }
            save_to_checkpoint(record, checkpoint_path, run_id)
            completed.add((mol_id, repr_name))

    if not dry_run:
        logger.info(f"\nBenchmark 7 complete for {model_id} | {thinking_str}")
        logger.info(f"Results saved to {checkpoint_path}")


# ============================================================================
# Benchmark 9: Tautomer Recognition
# ============================================================================


def run_benchmark_9(
    model_config: Dict[str, Any],
    thinking: bool,
    vllm_url: str,
    num_samples: Optional[int] = None,
    dry_run: bool = False,
    use_direct_vllm: bool = False,
    batch_size: int = 50,
    run_id: int = 1,
):
    """
    Run Benchmark 9: Tautomer Recognition

    For each pair in tautomer_pairs250.csv:
    - Convert both molecules to target representation
    - Build prompt, call model, parse yes/no
    - Save result

    Args:
        model_config: Model configuration dict
        thinking: Whether thinking is enabled
        vllm_url: vLLM API URL
        num_samples: If provided, limit to first N pairs
        dry_run: If True, print prompts without calling API
    """
    benchmark_name = "tautomer_recognition"
    model_id = model_config["id"]
    thinking_str = "thinking_on" if thinking else "thinking_off"

    logger.info(f"\n{'='*80}")
    logger.info(f"Starting Benchmark 9: Tautomer Recognition")
    logger.info(f"Model: {model_id} | Thinking: {thinking_str}")
    logger.info(f"{'='*80}\n")

    # Setup checkpoint
    checkpoint_dir = RESULTS_DIR / "raw"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"benchmark_9_{model_id}_{thinking_str}_run{run_id}.jsonl"

    # Load completed items (same structure as B5: keyed by (pair_id, representation))
    completed = load_checkpoint_benchmark_5(checkpoint_path)

    # Load data
    pairs_path = DATA_DIR / "tautomer_pairs250.csv"
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"TautomerPairs-250 not found at {pairs_path}. "
            "Run 01_prepare_dataset.py first (steps 10b)."
        )
    pairs_df = pd.read_csv(pairs_path)

    if num_samples:
        pairs_df = pairs_df.head(num_samples)

    logger.info(f"Processing {len(pairs_df)} pairs")

    dry_run_limit = 3

    # Phase 1: Pre-collect all pending tasks
    logger.info("Pre-collecting tasks...")
    all_tasks = []
    for idx, row in pairs_df.iterrows():
        pair_id = int(row["pair_id"])
        smiles1 = row["smiles_1"]
        smiles2 = row["smiles_2"]
        pair_type = row.get("pair_type", "unknown")
        tautomer_class = row.get("tautomer_class", "unknown")
        ground_truth = str(row["ground_truth"]).strip().lower() == "yes"

        for repr_name in REPRESENTATIONS:
            if (pair_id, repr_name) in completed:
                continue

            # Representation conversion — use precomputed columns where available
            if repr_name == "randomized_smiles":
                from rdkit import Chem as _Chem
                mol1 = _Chem.MolFromSmiles(smiles1)
                mol2 = _Chem.MolFromSmiles(smiles2)
                repr1 = _Chem.MolToSmiles(mol1, doRandom=True) if mol1 else None
                repr2 = _Chem.MolToSmiles(mol2, doRandom=True) if mol2 else None
            elif repr_name == "iupac":
                repr1 = row.get("iupac_1") if pd.notna(row.get("iupac_1", None)) else None
                repr2 = row.get("iupac_2") if pd.notna(row.get("iupac_2", None)) else None
            elif repr_name == "selfies":
                repr1 = row.get("selfies_1") if pd.notna(row.get("selfies_1", None)) else None
                repr2 = row.get("selfies_2") if pd.notna(row.get("selfies_2", None)) else None
            else:
                repr1 = representations.convert_to_representation(smiles1, repr_name)
                repr2 = representations.convert_to_representation(smiles2, repr_name)

            if repr1 is None or repr2 is None:
                logger.warning(f"Conversion failed for pair {pair_id} -> {repr_name}")
                continue

            prompt_text = prompts.create_tautomer_recognition_prompt(repr1, repr2, repr_name)
            all_tasks.append({
                "messages": [{"role": "user", "content": prompt_text}],
                "pair_id": pair_id,
                "repr_name": repr_name,
                "pair_type": pair_type,
                "tautomer_class": tautomer_class,
                "ground_truth": ground_truth,
            })

    logger.info(f"Total pending tasks: {len(all_tasks)}")

    gen_params = get_generation_params(model_config, "tautomer_recognition", thinking)
    stop_seqs = STOP_SEQUENCES.get("comprehension", [])

    if dry_run:
        for task in all_tasks[:dry_run_limit]:
            call_model(
                messages=task["messages"], model_config=model_config, thinking=thinking,
                vllm_url=vllm_url, generation_params=gen_params,
                stop_sequences=stop_seqs, dry_run=True,
            )
        logger.info(f"\nDry run limit reached ({dry_run_limit} prompts)")
        return

    # Phase 2: Process in batches
    for batch_start in tqdm(range(0, len(all_tasks), batch_size), desc="Batches"):
        batch = all_tasks[batch_start: batch_start + batch_size]
        messages_list = [t["messages"] for t in batch]

        if use_direct_vllm:
            results = call_model_direct_vllm_batch(
                messages_list=messages_list, model_config=model_config, thinking=thinking,
                generation_params=gen_params, stop_sequences=stop_seqs,
            )
        else:
            with ThreadPoolExecutor(max_workers=min(len(batch), 32)) as executor:
                future_to_idx = {
                    executor.submit(
                        call_model, t["messages"], model_config, thinking,
                        vllm_url, gen_params, stop_seqs,
                    ): i
                    for i, t in enumerate(batch)
                }
                results = [None] * len(batch)
                for future in as_completed(future_to_idx):
                    results[future_to_idx[future]] = future.result()

        for task, result in zip(batch, results):
            pair_id, repr_name = task["pair_id"], task["repr_name"]
            pair_type = task["pair_type"]
            tautomer_class = task["tautomer_class"]
            ground_truth = task["ground_truth"]

            if result is None:
                logger.warning(f"API call failed: pair {pair_id} | {repr_name} | {model_id} | {thinking_str}")
                record = {
                    "pair_id": pair_id, "representation": repr_name, "model": model_id,
                    "thinking": thinking, "pair_type": pair_type, "tautomer_class": tautomer_class,
                    "raw_response": None, "parsed_answer": None, "predicted": None,
                    "ground_truth": ground_truth, "latency_ms": None,
                }
            else:
                predicted = parsing.extract_yes_no(result["parsed_answer"])
                record = {
                    "pair_id": pair_id, "representation": repr_name, "model": model_id,
                    "thinking": thinking, "pair_type": pair_type, "tautomer_class": tautomer_class,
                    "raw_response": result["raw_response"], "parsed_answer": result["parsed_answer"],
                    "predicted": predicted, "ground_truth": ground_truth,
                    "latency_ms": result["latency_ms"],
                }
            save_to_checkpoint(record, checkpoint_path, run_id)
            completed.add((pair_id, repr_name))

    if not dry_run:
        logger.info(f"\nBenchmark 9 complete for {model_id} | {thinking_str}")
        logger.info(f"Results saved to {checkpoint_path}")


# ============================================================================
# Benchmark 10: Protonation State Recognition
# ============================================================================


def run_benchmark_10(
    model_config: Dict[str, Any],
    thinking: bool,
    vllm_url: str,
    num_samples: Optional[int] = None,
    dry_run: bool = False,
    use_direct_vllm: bool = False,
    batch_size: int = 50,
    run_id: int = 1,
):
    """
    Run Benchmark 10: Protonation State Recognition

    For each pair in protonation_pairs250.csv:
    - Convert both molecules to target representation
    - Build prompt, call model, parse yes/no
    - Save result

    Args:
        model_config: Model configuration dict
        thinking: Whether thinking is enabled
        vllm_url: vLLM API URL
        num_samples: If provided, limit to first N pairs
        dry_run: If True, print prompts without calling API
    """
    benchmark_name = "protonation_recognition"
    model_id = model_config["id"]
    thinking_str = "thinking_on" if thinking else "thinking_off"

    logger.info(f"\n{'='*80}")
    logger.info(f"Starting Benchmark 10: Protonation State Recognition")
    logger.info(f"Model: {model_id} | Thinking: {thinking_str}")
    logger.info(f"{'='*80}\n")

    # Setup checkpoint
    checkpoint_dir = RESULTS_DIR / "raw"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"benchmark_10_{model_id}_{thinking_str}_run{run_id}.jsonl"

    completed = load_checkpoint_benchmark_5(checkpoint_path)

    # Load data
    pairs_path = DATA_DIR / "protonation_pairs250.csv"
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"ProtonationPairs-250 not found at {pairs_path}. "
            "Run 01_prepare_dataset.py first (step 10c)."
        )
    pairs_df = pd.read_csv(pairs_path)

    if num_samples:
        pairs_df = pairs_df.head(num_samples)

    logger.info(f"Processing {len(pairs_df)} pairs")

    dry_run_limit = 3

    # Phase 1: Pre-collect all pending tasks
    logger.info("Pre-collecting tasks...")
    all_tasks = []
    for idx, row in pairs_df.iterrows():
        pair_id = int(row["pair_id"])
        smiles1 = row["smiles_1"]
        smiles2 = row["smiles_2"]
        pair_type = row.get("pair_type", "unknown")
        ionizable_group = row.get("ionizable_group", "none")
        charge_1 = row.get("charge_1", 0)
        charge_2 = row.get("charge_2", 0)
        ground_truth = str(row["ground_truth"]).strip().lower() == "yes"

        for repr_name in REPRESENTATIONS:
            if (pair_id, repr_name) in completed:
                continue

            if repr_name == "randomized_smiles":
                from rdkit import Chem as _Chem
                mol1 = _Chem.MolFromSmiles(smiles1)
                mol2 = _Chem.MolFromSmiles(smiles2)
                repr1 = _Chem.MolToSmiles(mol1, doRandom=True) if mol1 else None
                repr2 = _Chem.MolToSmiles(mol2, doRandom=True) if mol2 else None
            elif repr_name == "iupac":
                repr1 = row.get("iupac_1") if pd.notna(row.get("iupac_1", None)) else None
                repr2 = row.get("iupac_2") if pd.notna(row.get("iupac_2", None)) else None
            elif repr_name == "selfies":
                repr1 = row.get("selfies_1") if pd.notna(row.get("selfies_1", None)) else None
                repr2 = row.get("selfies_2") if pd.notna(row.get("selfies_2", None)) else None
            else:
                repr1 = representations.convert_to_representation(smiles1, repr_name)
                repr2 = representations.convert_to_representation(smiles2, repr_name)

            if repr1 is None or repr2 is None:
                logger.warning(f"Conversion failed for pair {pair_id} -> {repr_name}")
                continue

            prompt_text = prompts.create_protonation_recognition_prompt(repr1, repr2, repr_name)
            all_tasks.append({
                "messages": [{"role": "user", "content": prompt_text}],
                "pair_id": pair_id,
                "repr_name": repr_name,
                "pair_type": pair_type,
                "ionizable_group": ionizable_group,
                "charge_1": charge_1,
                "charge_2": charge_2,
                "ground_truth": ground_truth,
            })

    logger.info(f"Total pending tasks: {len(all_tasks)}")

    gen_params = get_generation_params(model_config, "protonation_recognition", thinking)
    stop_seqs = STOP_SEQUENCES.get("comprehension", [])

    if dry_run:
        for task in all_tasks[:dry_run_limit]:
            call_model(
                messages=task["messages"], model_config=model_config, thinking=thinking,
                vllm_url=vllm_url, generation_params=gen_params,
                stop_sequences=stop_seqs, dry_run=True,
            )
        logger.info(f"\nDry run limit reached ({dry_run_limit} prompts)")
        return

    # Phase 2: Process in batches
    for batch_start in tqdm(range(0, len(all_tasks), batch_size), desc="Batches"):
        batch = all_tasks[batch_start: batch_start + batch_size]
        messages_list = [t["messages"] for t in batch]

        if use_direct_vllm:
            results = call_model_direct_vllm_batch(
                messages_list=messages_list, model_config=model_config, thinking=thinking,
                generation_params=gen_params, stop_sequences=stop_seqs,
            )
        else:
            with ThreadPoolExecutor(max_workers=min(len(batch), 32)) as executor:
                future_to_idx = {
                    executor.submit(
                        call_model, t["messages"], model_config, thinking,
                        vllm_url, gen_params, stop_seqs,
                    ): i
                    for i, t in enumerate(batch)
                }
                results = [None] * len(batch)
                for future in as_completed(future_to_idx):
                    results[future_to_idx[future]] = future.result()

        for task, result in zip(batch, results):
            pair_id, repr_name = task["pair_id"], task["repr_name"]
            pair_type = task["pair_type"]
            ionizable_group = task["ionizable_group"]
            charge_1 = task["charge_1"]
            charge_2 = task["charge_2"]
            ground_truth = task["ground_truth"]

            if result is None:
                logger.warning(f"API call failed: pair {pair_id} | {repr_name} | {model_id} | {thinking_str}")
                record = {
                    "pair_id": pair_id, "representation": repr_name, "model": model_id,
                    "thinking": thinking, "pair_type": pair_type,
                    "ionizable_group": ionizable_group, "charge_1": charge_1, "charge_2": charge_2,
                    "raw_response": None, "parsed_answer": None, "predicted": None,
                    "ground_truth": ground_truth, "latency_ms": None,
                }
            else:
                predicted = parsing.extract_yes_no(result["parsed_answer"])
                record = {
                    "pair_id": pair_id, "representation": repr_name, "model": model_id,
                    "thinking": thinking, "pair_type": pair_type,
                    "ionizable_group": ionizable_group, "charge_1": charge_1, "charge_2": charge_2,
                    "raw_response": result["raw_response"], "parsed_answer": result["parsed_answer"],
                    "predicted": predicted, "ground_truth": ground_truth,
                    "latency_ms": result["latency_ms"],
                }
            save_to_checkpoint(record, checkpoint_path, run_id)
            completed.add((pair_id, repr_name))

    if not dry_run:
        logger.info(f"\nBenchmark 10 complete for {model_id} | {thinking_str}")
        logger.info(f"Results saved to {checkpoint_path}")


# ============================================================================
# Consolidation
# ============================================================================


def consolidate_benchmark_results(benchmark_num: int):
    """
    Consolidate all JSONL checkpoints for a benchmark into a single CSV.

    Args:
        benchmark_num: Benchmark number (1-7)
    """
    from config import BENCHMARK_NUM_TO_NAME, get_result_filename

    benchmark_name = BENCHMARK_NUM_TO_NAME[benchmark_num]
    checkpoint_dir = RESULTS_DIR / "raw"

    # Find all checkpoint files for this benchmark
    pattern = f"benchmark_{benchmark_num}_*.jsonl"
    checkpoint_files = list(checkpoint_dir.glob(pattern))

    if not checkpoint_files:
        logger.warning(f"No checkpoint files found for benchmark {benchmark_num}")
        return

    logger.info(
        f"Consolidating {len(checkpoint_files)} checkpoint files for {benchmark_name}"
    )

    # Read all records
    all_records = []
    for checkpoint_file in checkpoint_files:
        with open(checkpoint_file, "r") as f:
            for line in f:
                all_records.append(json.loads(line))

    # Convert to DataFrame
    df = pd.DataFrame(all_records)

    # Save to CSV
    output_path = get_result_filename(benchmark_name, suffix="_raw")
    df.to_csv(output_path, index=False)

    logger.info(f"Consolidated results saved to {output_path}")
    logger.info(f"Total records: {len(df)}")


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference for molecular representation benchmarks"
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        help='Benchmark to run (1-7 or "all")',
    )

    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help=f'Model to use ({", ".join(MODEL_IDS)} or "all")',
    )

    parser.add_argument(
        "--representation",
        type=str,
        default="all",
        help=f'Representation to use ({", ".join(REPRESENTATIONS)} or "all")',
    )

    parser.add_argument(
        "--thinking",
        type=str,
        default="both",
        choices=["on", "off", "both"],
        help="Thinking mode (on, off, or both)",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)",
    )

    parser.add_argument(
        "--vllm_url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API URL",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print first 3 prompts per representation without calling API",
    )

    parser.add_argument(
        "--consolidate",
        type=int,
        default=None,
        help="Consolidate checkpoint files for benchmark N into CSV",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override data directory (default: data/)",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Override results directory (default: results/)",
    )

    parser.add_argument(
        "--direct_vllm",
        action="store_true",
        help="Use direct vLLM model loading instead of API server (faster, requires GPU)",
    )

    parser.add_argument(
        "--tensor_parallel",
        type=int,
        default=1,
        help="Tensor parallel size for direct vLLM (number of GPUs)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Batch size for parallel inference (default: 50 prompts per .generate() call)",
    )

    parser.add_argument(
        "--run_id",
        type=int,
        default=1,
        help="Run number for error-bar replication (default: 1). Use 1/2/3 for three independent runs.",
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Override max output tokens (default: use model config value)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Override DATA_DIR and RESULTS_DIR if specified
    global DATA_DIR, RESULTS_DIR, VLLM_MODEL
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
        logger.info(f"Using custom data directory: {DATA_DIR}")
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)
        logger.info(f"Using custom results directory: {RESULTS_DIR}")

    # Handle consolidation
    if args.consolidate is not None:
        consolidate_benchmark_results(args.consolidate)
        return

    # Parse model selection first (needed for direct vLLM)
    if args.model == "all":
        if args.direct_vllm:
            logger.error("ERROR: --direct_vllm requires a single model (use --model <model_id>)")
            sys.exit(1)
        model_configs = MODELS
    else:
        model_configs = [m for m in MODELS if m["id"] == args.model]
        if not model_configs:
            logger.error(f"Unknown model: {args.model}")
            sys.exit(1)

    # Override max_tokens if specified
    if args.max_tokens is not None:
        for mc in model_configs:
            mc["generation_params"]["max_tokens"] = args.max_tokens
            logger.info(f"Overriding max_tokens to {args.max_tokens} for {mc['id']}")

    # Initialize direct vLLM if requested
    if args.direct_vllm:
        if len(model_configs) != 1:
            logger.error("ERROR: --direct_vllm requires exactly one model")
            sys.exit(1)

        logger.info("=" * 80)
        logger.info("DIRECT vLLM MODE ENABLED")
        logger.info("=" * 80)
        logger.info("This mode loads the model directly into GPU memory.")
        logger.info("Benefits: 2-3x faster than API mode (no network overhead)")
        logger.info("Requirements: GPU with enough memory for the model")
        logger.info("=" * 80)

        VLLM_MODEL = initialize_vllm_model(
            model_configs[0],
            tensor_parallel_size=args.tensor_parallel
        )

    # Parse benchmark selection
    if args.benchmark == "all":
        benchmark_nums = list(range(1, 7)) + [9, 10]
    else:
        benchmark_nums = [int(args.benchmark)]

    # Filter representations if specified
    global REPRESENTATIONS
    if args.representation != "all":
        if args.representation not in REPRESENTATIONS:
            logger.error(f"Unknown representation: {args.representation}. Valid: {REPRESENTATIONS}")
            sys.exit(1)
        REPRESENTATIONS = [args.representation]
        logger.info(f"Filtering to representation: {args.representation}")

    # Parse thinking selection
    if args.thinking == "both":
        thinking_modes = [True, False]
    elif args.thinking == "on":
        thinking_modes = [True]
    else:
        thinking_modes = [False]

    # Map benchmark numbers to runner functions
    benchmark_runners = {
        1: run_benchmark_1,
        2: run_benchmark_2,
        3: run_benchmark_3,
        4: run_benchmark_4,
        5: run_benchmark_5,
        6: run_benchmark_6,
        # 7: run_benchmark_7,
        9: run_benchmark_9,
        10: run_benchmark_10,
    }

    # Run benchmarks
    for benchmark_num in benchmark_nums:
        runner = benchmark_runners[benchmark_num]
        benchmark_name = BENCHMARK_NUM_TO_NAME[benchmark_num]

        logger.info(f"{'='*60}")
        logger.info(f"BENCHMARK {benchmark_num} STARTED: {benchmark_name}")
        logger.info(f"{'='*60}")

        for model_config in model_configs:
            for thinking in thinking_modes:
                try:
                    runner(
                        model_config=model_config,
                        thinking=thinking,
                        vllm_url=args.vllm_url,
                        num_samples=args.num_samples,
                        dry_run=args.dry_run,
                        use_direct_vllm=args.direct_vllm,
                        batch_size=args.batch_size,
                        run_id=args.run_id,
                    )
                except NotImplementedError as e:
                    logger.warning(str(e))
                    continue
                except Exception as e:
                    logger.error(f"Error in benchmark {benchmark_num}: {e}")
                    continue

        logger.info(f"{'='*60}")
        logger.info(f"BENCHMARK {benchmark_num} DONE: {benchmark_name}")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
