"""
BFCL v3 schema adapter for Seer tool-calling eval.

BFCL v3 data format:
- Test files: JSONL with {id, question: [[{role,content}]], function: [{name,desc,parameters}]}
- Answer files: JSONL with {id, ground_truth: [{"func_name": {"param": [acceptable_vals]}}]}
- question is doubly-nested: [[messages]] (outer=list of turns, inner=list of messages)
- ground_truth args are wrapped in arrays (multiple acceptable values)
- parameter type is "dict" not "object" in some files

Scoring: AST accuracy = exact name match + at least one acceptable arg value present.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DATA_DIR = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Seer → BFCL conversion
# ---------------------------------------------------------------------------

def seer_tools_to_bfcl_functions(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Seer tool catalog to BFCL function array format.

    Strips x-resource-picker, required_scopes, output_schema, etc.
    """
    bfcl_functions = []
    for tool in tools:
        params = deepcopy(tool.get("parameters", {}))
        _strip_nonstandard(params)
        # Normalize "object" type (Seer uses standard JSON Schema)
        if params.get("type") == "object":
            pass  # Already correct
        bfcl_functions.append({
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": params,
        })
    return bfcl_functions


def _strip_nonstandard(schema: Any) -> None:
    """Recursively remove non-standard JSON Schema keys."""
    if isinstance(schema, dict):
        keys_to_remove = [k for k in schema if k.startswith("x-")]
        for k in keys_to_remove:
            del schema[k]
        for v in schema.values():
            _strip_nonstandard(v)
    elif isinstance(schema, list):
        for item in schema:
            _strip_nonstandard(item)


# ---------------------------------------------------------------------------
# BFCL ground truth parsing
# ---------------------------------------------------------------------------

def parse_ground_truth(gt_entry: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse BFCL ground_truth format to normalized [{name, arguments}].

    BFCL format: [{"func_name": {"param": [val1, val2], "param2": [val3]}}]
    Empty list means no function should be called (irrelevance).
    """
    if not gt_entry:
        return []

    calls = []
    for item in gt_entry:
        for func_name, args_dict in item.items():
            # Normalize args: unwrap single-element arrays, keep first value
            normalized_args = {}
            for param_name, values in args_dict.items():
                if isinstance(values, list):
                    # BFCL stores acceptable values as arrays; use first non-empty
                    if values:
                        val = values[0]
                        # Empty string means optional param not provided
                        if val != "":
                            normalized_args[param_name] = val
                    # If all values are empty strings, omit the param
                else:
                    normalized_args[param_name] = values
            calls.append({"name": func_name, "arguments": normalized_args})

    return calls


def get_acceptable_args(gt_entry: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get all acceptable argument value sets for each ground truth call.

    Returns list of {name, acceptable_args} where acceptable_args maps
    param names to lists of acceptable values.
    """
    if not gt_entry:
        return []

    calls = []
    for item in gt_entry:
        for func_name, args_dict in item.items():
            acceptable = {}
            for param_name, values in args_dict.items():
                if isinstance(values, list):
                    # Filter out empty strings (optional params)
                    acceptable[param_name] = [v for v in values if v != ""]
                else:
                    acceptable[param_name] = [values]
            calls.append({"name": func_name, "acceptable_args": acceptable})
    return calls


# ---------------------------------------------------------------------------
# AST comparison (BFCL scoring rules)
# ---------------------------------------------------------------------------

def compare_tool_calls(
    predicted: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    acceptable_args: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Compare predicted tool calls against ground truth.

    Args:
        predicted: [{name, arguments}] from our pipeline
        ground_truth: normalized [{name, arguments}] from parse_ground_truth
        acceptable_args: optional per-call acceptable arg lists from get_acceptable_args

    Returns (is_correct, details).
    """
    # Irrelevance: no calls expected
    if not ground_truth and not predicted:
        return True, {"reason": "both empty (irrelevant query handled correctly)"}

    if not ground_truth and predicted:
        return False, {"reason": "ground truth is empty but prediction made calls"}

    if ground_truth and not predicted:
        return False, {"reason": "ground truth has calls but prediction is empty"}

    # Normalize both sides
    pred_calls = _normalize_calls(predicted)
    gt_calls = _normalize_calls(ground_truth)

    if len(pred_calls) != len(gt_calls):
        return False, {
            "reason": f"call count mismatch: predicted {len(pred_calls)}, expected {len(gt_calls)}",
        }

    # Single call
    if len(gt_calls) == 1:
        match, detail = _compare_single_call(
            pred_calls[0], gt_calls[0],
            acceptable=acceptable_args[0] if acceptable_args else None,
        )
        return match, {"calls": [detail]}

    # Parallel: order-independent matching
    return _compare_parallel_calls(pred_calls, gt_calls, all_acceptable=acceptable_args)


def _normalize_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize a list of tool calls for comparison."""
    normalized = []
    for call in calls:
        name = call.get("name", "")
        args = call.get("arguments", call.get("args", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        normalized.append({"name": name, "arguments": args})
    return normalized


def _compare_single_call(
    predicted: Dict[str, Any],
    ground_truth: Dict[str, Any],
    acceptable: Optional[Dict[str, Any]] = None,
    required_params: Optional[List[str]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Compare a single predicted call against ground truth."""
    detail: Dict[str, Any] = {
        "expected_name": ground_truth["name"],
        "predicted_name": predicted["name"],
    }

    if predicted["name"] != ground_truth["name"]:
        detail["reason"] = "name mismatch"
        return False, detail

    pred_args = predicted.get("arguments", {})
    gt_args = ground_truth.get("arguments", {})

    if acceptable and "acceptable_args" in acceptable:
        args_match, arg_details = _compare_args_acceptable(pred_args, acceptable["acceptable_args"])
    else:
        args_match, arg_details = _compare_arguments(pred_args, gt_args, required_params=required_params)

    detail["arguments"] = arg_details
    if not args_match:
        detail["reason"] = "argument mismatch"
        return False, detail

    detail["reason"] = "exact match"
    return True, detail


def _compare_arguments(
    predicted: Dict[str, Any],
    ground_truth: Dict[str, Any],
    required_params: Optional[List[str]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Type-aware argument comparison. Extra predicted args are OK.

    If required_params is provided, only missing required params count as failures.
    Optional params missing from prediction are OK if the GT value matches the default.
    """
    details: Dict[str, Any] = {"matching": [], "missing": [], "type_mismatch": []}
    all_match = True

    for key, gt_val in ground_truth.items():
        if key not in predicted:
            # If this param is optional, it's OK to omit
            if required_params and key not in required_params:
                details["matching"].append(f"{key} (optional, omitted)")
                continue
            details["missing"].append(key)
            all_match = False
            continue

        pred_val = predicted[key]
        if _values_match(pred_val, gt_val):
            details["matching"].append(key)
        else:
            details["type_mismatch"].append({
                "key": key, "expected": gt_val, "predicted": pred_val,
            })
            all_match = False

    return all_match, details


def _compare_args_acceptable(
    predicted: Dict[str, Any],
    acceptable: Dict[str, List[Any]],
) -> Tuple[bool, Dict[str, Any]]:
    """Compare predicted args against list of acceptable values per param."""
    details: Dict[str, Any] = {"matching": [], "missing": [], "no_acceptable_match": []}
    all_match = True

    for key, acceptable_vals in acceptable.items():
        if not acceptable_vals:
            continue  # Optional param with no required value

        if key not in predicted:
            details["missing"].append(key)
            all_match = False
            continue

        pred_val = predicted[key]
        if any(_values_match(pred_val, v) for v in acceptable_vals):
            details["matching"].append(key)
        else:
            details["no_acceptable_match"].append({
                "key": key, "predicted": pred_val, "acceptable": acceptable_vals,
            })
            all_match = False

    return all_match, details


def _values_match(predicted: Any, ground_truth: Any) -> bool:
    """Type-aware value comparison."""
    if type(predicted) != type(ground_truth):
        # Allow numeric comparison across int/float
        if isinstance(predicted, (int, float)) and isinstance(ground_truth, (int, float)):
            return predicted == ground_truth
        return False

    if isinstance(ground_truth, dict):
        if set(predicted.keys()) != set(ground_truth.keys()):
            return False
        return all(_values_match(predicted[k], ground_truth[k]) for k in ground_truth)

    if isinstance(ground_truth, list):
        if len(predicted) != len(ground_truth):
            return False
        return all(_values_match(p, g) for p, g in zip(predicted, ground_truth))

    return predicted == ground_truth


def _compare_parallel_calls(
    predicted: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    all_acceptable: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Order-independent matching for parallel calls."""
    from itertools import permutations

    gt_indices = list(range(len(ground_truth)))
    for perm in permutations(gt_indices):
        all_match = True
        details = []
        for i, gt_idx in enumerate(perm):
            acc = all_acceptable[gt_idx] if all_acceptable else None
            match, detail = _compare_single_call(predicted[i], ground_truth[gt_idx], acceptable=acc)
            details.append(detail)
            if not match:
                all_match = False
                break
        if all_match:
            return True, {"calls": details, "reason": "parallel match (order-independent)"}

    return False, {"reason": "no matching permutation found for parallel calls"}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_bfcl_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load a BFCL v3 JSONL file."""
    cases = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return cases


def load_answers(category: str) -> Dict[str, Any]:
    """Load possible_answer file for a category. Returns {id: ground_truth}."""
    answer_file = DATA_DIR / f"answers_{category}.json"
    if not answer_file.exists():
        return {}

    answers = {}
    with open(str(answer_file), "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                answers[d.get("id", "")] = d.get("ground_truth", [])
            except json.JSONDecodeError:
                continue
    return answers


def extract_query(test_case: Dict[str, Any]) -> str:
    """Extract the user query from a BFCL test case.

    Handles doubly-nested question format: [[{role, content}]]
    """
    messages = test_case.get("question", [])
    # Handle [[messages]] nesting
    if messages and isinstance(messages[0], list):
        messages = messages[0]

    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def extract_functions(test_case: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract function definitions from a BFCL test case."""
    return test_case.get("function", [])


def get_ground_truth_for_case(
    test_case: Dict[str, Any],
    answers: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Get ground truth for a test case.

    Checks case's own ground_truth first, then looks in answers dict.
    """
    # Some files have inline ground_truth
    if "ground_truth" in test_case:
        gt_raw = test_case["ground_truth"]
        if isinstance(gt_raw, list) and gt_raw:
            if isinstance(gt_raw[0], dict):
                return parse_ground_truth(gt_raw)

    # Look up from answers file
    case_id = test_case.get("id", "")
    if case_id in answers:
        return parse_ground_truth(answers[case_id])

    return []
