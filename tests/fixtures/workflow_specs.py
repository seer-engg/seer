"""
Workflow specification builders for testing.

Provides fluent builders for creating workflow specs with various configurations.
Makes test setup cleaner and more maintainable.

Example:
    spec = (
        WorkflowSpecBuilder()
        .add_trigger("t1", "test.trigger")
        .add_task_node("n1", "test.tool", {"param": "value"})
        .add_edge("t1", "n1")
        .build()
    )
"""
from typing import Any, Dict, List, Optional


class WorkflowSpecBuilder:
    """
    Fluent builder for creating workflow specifications in tests.

    Provides a clean API for building workflow specs with various node types,
    edges, and configurations.
    """

    def __init__(self, version: str = "2"):
        """
        Initialize builder with default values.

        Args:
            version: Workflow spec version (default: "2")
        """
        self.version = version
        self.triggers: List[Dict[str, Any]] = []
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self.variables: Dict[str, Any] = {}

    def add_trigger(
        self,
        trigger_id: str,
        key: str,
        label: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> "WorkflowSpecBuilder":
        """
        Add a trigger to the workflow.

        Args:
            trigger_id: Unique trigger identifier
            key: Trigger key (e.g., "gmail.new_email")
            label: Human-readable label
            config: Trigger configuration

        Returns:
            Self for chaining
        """
        self.triggers.append(
            {
                "id": trigger_id,
                "key": key,
                "label": label or key,
                "config": config or {},
            }
        )
        return self

    def add_condition_node(
        self,
        node_id: str,
        condition: str,
        label: Optional[str] = None,
    ) -> "WorkflowSpecBuilder":
        """
        Add a condition (branching) node to the workflow.

        Args:
            node_id: Unique node identifier
            condition: Condition expression (e.g., "${task_1.result.success}")
            label: Human-readable label

        Returns:
            Self for chaining
        """
        self.nodes.append(
            {
                "id": node_id,
                "type": "condition",
                "label": label or "Condition",
                "config": {
                    "condition": condition,
                },
            }
        )
        return self

    def add_parallel_node(
        self,
        node_id: str,
        branch_count: int = 2,
        label: Optional[str] = None,
    ) -> "WorkflowSpecBuilder":
        """
        Add a parallel execution node to the workflow.

        Args:
            node_id: Unique node identifier
            branch_count: Number of parallel branches
            label: Human-readable label

        Returns:
            Self for chaining
        """
        self.nodes.append(
            {
                "id": node_id,
                "type": "parallel",
                "label": label or "Parallel",
                "config": {
                    "branch_count": branch_count,
                },
            }
        )
        return self

    def add_loop_node(
        self,
        node_id: str,
        collection_expr: str,
        label: Optional[str] = None,
    ) -> "WorkflowSpecBuilder":
        """
        Add a loop node to the workflow.

        Args:
            node_id: Unique node identifier
            collection_expr: Expression evaluating to collection (e.g., "${trigger.data.items}")
            label: Human-readable label

        Returns:
            Self for chaining
        """
        self.nodes.append(
            {
                "id": node_id,
                "type": "loop",
                "label": label or "Loop",
                "config": {
                    "collection": collection_expr,
                },
            }
        )
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        label: Optional[str] = None,
        condition: Optional[str] = None,
    ) -> "WorkflowSpecBuilder":
        """
        Add an edge connecting two nodes.

        Args:
            source: Source node ID
            target: Target node ID
            label: Edge label (for conditional branches, e.g., "true", "false")
            condition: Optional condition expression

        Returns:
            Self for chaining
        """
        edge: Dict[str, Any] = {
            "source": source,
            "target": target,
        }

        if label:
            edge["label"] = label

        if condition:
            edge["condition"] = condition

        self.edges.append(edge)
        return self

    def add_variable(self, name: str, value: Any) -> "WorkflowSpecBuilder":
        """
        Add a workflow variable.

        Args:
            name: Variable name
            value: Variable value

        Returns:
            Self for chaining
        """
        self.variables[name] = value
        return self

    def build(self) -> Dict[str, Any]:
        """
        Build the final workflow specification.

        Returns:
            Complete workflow spec dictionary
        """
        spec: Dict[str, Any] = {
            "version": self.version,
            "triggers": self.triggers,
            "nodes": self.nodes,
            "edges": self.edges,
        }

        if self.variables:
            spec["variables"] = self.variables

        return spec

    @classmethod
    def minimal(cls) -> Dict[str, Any]:
        """
        Create a minimal valid workflow spec.

        Returns:
            Minimal workflow spec with one trigger, one task, one edge
        """
        return (
            cls()
            .add_trigger("t1", "test.trigger")
            .add_task_node("n1", "test.tool")
            .add_edge("t1", "n1")
            .build()
        )

    @classmethod
    def with_branching(cls) -> Dict[str, Any]:
        """
        Create workflow spec with conditional branching.

        Returns:
            Workflow spec with condition node and multiple branches
        """
        return (
            cls()
            .add_trigger("t1", "test.trigger")
            .add_task_node("task_1", "test.check", {}, "Check Condition")
            .add_condition_node("cond_1", "${task_1.result.success}", "Branch")
            .add_task_node("task_success", "test.on_success", {}, "Success Path")
            .add_task_node("task_failure", "test.on_failure", {}, "Failure Path")
            .add_edge("t1", "task_1")
            .add_edge("task_1", "cond_1")
            .add_edge("cond_1", "task_success", label="true")
            .add_edge("cond_1", "task_failure", label="false")
            .build()
        )

    @classmethod
    def with_loop(cls) -> Dict[str, Any]:
        """
        Create workflow spec with loop node.

        Returns:
            Workflow spec with loop iterating over collection
        """
        return (
            cls()
            .add_trigger("t1", "test.trigger")
            .add_loop_node("loop_1", "${t1.data.items}", "Process Items")
            .add_task_node("task_1", "test.process_item", {"item": "${loop.current}"}, "Process")
            .add_edge("t1", "loop_1")
            .add_edge("loop_1", "task_1")
            .build()
        )

    @classmethod
    def with_parallel(cls) -> Dict[str, Any]:
        """
        Create workflow spec with parallel execution.

        Returns:
            Workflow spec with parallel node executing multiple tasks concurrently
        """
        return (
            cls()
            .add_trigger("t1", "test.trigger")
            .add_parallel_node("parallel_1", branch_count=2, label="Parallel Tasks")
            .add_task_node("task_a", "test.task_a", {}, "Task A")
            .add_task_node("task_b", "test.task_b", {}, "Task B")
            .add_task_node("task_merge", "test.merge", {}, "Merge Results")
            .add_edge("t1", "parallel_1")
            .add_edge("parallel_1", "task_a")
            .add_edge("parallel_1", "task_b")
            .add_edge("task_a", "task_merge")
            .add_edge("task_b", "task_merge")
            .build()
        )


def create_invalid_spec_missing_version() -> Dict[str, Any]:
    """Create invalid spec missing version field."""
    return {
        "triggers": [],
        "nodes": [],
        "edges": [],
    }


def create_invalid_spec_invalid_version() -> Dict[str, Any]:
    """Create invalid spec with unsupported version."""
    return {
        "version": "99",
        "triggers": [],
        "nodes": [],
        "edges": [],
    }


def create_invalid_spec_missing_triggers() -> Dict[str, Any]:
    """Create invalid spec missing triggers."""
    return {
        "version": "2",
        "nodes": [],
        "edges": [],
    }


def create_invalid_spec_cyclic_edges() -> Dict[str, Any]:
    """Create invalid spec with cyclic edges."""
    return {
        "version": "2",
        "triggers": [{"id": "t1", "key": "test.trigger", "label": "Test", "config": {}}],
        "nodes": [
            {"id": "n1", "type": "tool", "label": "Task 1", "config": {"tool_call": {"tool_id": "test.tool", "parameters": {}}}},
            {"id": "n2", "type": "tool", "label": "Task 2", "config": {"tool_call": {"tool_id": "test.tool", "parameters": {}}}},
        ],
        "edges": [
            {"source": "t1", "target": "n1"},
            {"source": "n1", "target": "n2"},
            {"source": "n2", "target": "n1"},  # Cycle!
        ],
    }
