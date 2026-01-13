"""
V2 Workflow Schema Validator

Validates workflow specs against V2 schema with:
- Structure validation (JSON Schema)
- DAG validation (cycles, orphans, connectivity)
- Reference validation (node IDs, template expressions)
"""
# pylint: disable=too-many-branches  # Validation logic requires multiple conditional checks

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import networkx as nx
from jsonschema import ValidationError as JSONSchemaValidationError
from jsonschema import validate


class ValidationError(Exception):
    """Workflow validation error"""

    def __init__(self, message: str, path: str | None = None):
        self.message = message
        self.path = path
        super().__init__(f"{path}: {message}" if path else message)


class WorkflowValidator:
    """Validates V2 workflow specifications"""

    def __init__(self, schema: Dict[str, Any]):
        """Initialize with JSON Schema"""
        self.schema = schema

    def validate(self, workflow: Dict[str, Any]) -> List[ValidationError]:
        """Validate workflow and return all errors"""
        errors: List[ValidationError] = []

        # 1. JSON Schema validation
        try:
            validate(instance=workflow, schema=self.schema)
        except JSONSchemaValidationError as e:
            errors.append(
                ValidationError(
                    f"Schema validation failed: {e.message}",
                    path=".".join(str(p) for p in e.path) if e.path else None,
                )
            )
            return errors  # Don't continue if structure is invalid

        spec = workflow.get("spec", {})

        # 2. DAG validation
        errors.extend(self._validate_dag(spec.get("nodes", []), spec.get("edges", [])))

        # 3. Node reference validation
        errors.extend(self._validate_node_references(spec.get("nodes", []), spec.get("edges", [])))

        # 4. Template expression validation (basic syntax)
        errors.extend(self._validate_expressions(spec))

        return errors

    # pylint: disable=too-complex  # DAG validation requires multiple conditional checks
    def _validate_dag(
        self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
    ) -> List[ValidationError]:
        """Validate DAG properties: no cycles, connectivity"""
        errors: List[ValidationError] = []

        if not edges:
            # Empty edges is OK (fallback to sequential)
            return errors

        # Build node ID set
        node_ids = {n["id"] for n in nodes}
        node_ids.add("_start")
        node_ids.add("_end")

        # Build graph
        graph = nx.DiGraph()

        for edge in edges:
            from_id = edge["from"]
            to_id = edge["to"]

            # Check edge nodes exist
            if from_id not in node_ids:
                errors.append(
                    ValidationError(
                        f"Edge references unknown node: '{from_id}'",
                        path="spec.edges",
                    )
                )
            if to_id not in node_ids:
                errors.append(
                    ValidationError(
                        f"Edge references unknown node: '{to_id}'",
                        path="spec.edges",
                    )
                )

            graph.add_edge(from_id, to_id)

        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(graph))
            if cycles:
                for cycle in cycles:
                    errors.append(
                        ValidationError(
                            f"DAG contains cycle: {' -> '.join(cycle)}",
                            path="spec.edges",
                        )
                    )
        except (nx.NetworkXError, RuntimeError) as e:  # NetworkX-specific errors
            errors.append(
                ValidationError(
                    f"Failed to check for cycles: {e}",
                    path="spec.edges",
                )
            )

        # Check for orphaned nodes (no incoming or outgoing edges)
        for node_id in node_ids:
            if node_id in ("_start", "_end"):
                continue
            if node_id not in graph:
                errors.append(
                    ValidationError(
                        f"Node '{node_id}' has no incoming or outgoing edges (orphaned)",
                        path="spec.nodes",
                    )
                )

        # Check connectivity (all nodes reachable from _start)
        if "_start" in graph:
            reachable = nx.descendants(graph, "_start")
            reachable.add("_start")
            for node_id in node_ids:
                if node_id not in reachable and node_id != "_end":
                    errors.append(
                        ValidationError(
                            f"Node '{node_id}' is not reachable from _start",
                            path="spec.nodes",
                        )
                    )

        return errors

    def _validate_node_references(
        self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
    ) -> List[ValidationError]:
        """Validate that all nodes in edges exist"""
        errors: List[ValidationError] = []
        node_ids = {n["id"] for n in nodes}

        for edge in edges:
            from_id = edge["from"]
            to_id = edge["to"]

            if from_id not in node_ids and from_id not in ("_start", "_end"):
                errors.append(
                    ValidationError(
                        f"Edge 'from' references non-existent node: '{from_id}'",
                        path="spec.edges",
                    )
                )

            if to_id not in node_ids and to_id not in ("_start", "_end"):
                errors.append(
                    ValidationError(
                        f"Edge 'to' references non-existent node: '{to_id}'",
                        path="spec.edges",
                    )
                )

        return errors

    # pylint: disable=too-complex  # Expression validation requires nested recursive checks
    def _validate_expressions(self, spec: Dict[str, Any]) -> List[ValidationError]:
        """Validate template expressions (basic syntax check)"""
        errors: List[ValidationError] = []

        # Check for balanced ${ } braces
        def check_expr(text: str, path: str) -> None:
            if not isinstance(text, str):
                return
            open_count = text.count("${")
            close_count = text.count("}")
            if open_count != close_count:
                errors.append(
                    ValidationError(
                        "Unbalanced template expression braces",
                        path=path,
                    )
                )

        # Check all string fields recursively
        def check_dict(d: Dict[str, Any], path: str) -> None:
            for key, value in d.items():
                new_path = f"{path}.{key}" if path else key
                if isinstance(value, str):
                    check_expr(value, new_path)
                elif isinstance(value, dict):
                    check_dict(value, new_path)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            check_dict(item, f"{new_path}[{i}]")
                        elif isinstance(item, str):
                            check_expr(item, f"{new_path}[{i}]")

        check_dict(spec, "spec")

        return errors


def validate_workflow_v2(workflow: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a V2 workflow.

    Returns:
        (is_valid, errors)
    """
    validator = WorkflowValidator(schema)
    errors = validator.validate(workflow)
    return len(errors) == 0, [str(e) for e in errors]
