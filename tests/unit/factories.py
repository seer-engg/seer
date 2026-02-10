"""
Test data factories for creating model instances.

Uses a simple factory pattern to create test data with sensible defaults
that can be customized. This reduces duplication of test data creation
across test files.

Usage:
    from tests.unit.factories import WorkflowSpecFactory, TriggerSpecFactory

    # Create minimal spec
    spec = WorkflowSpecFactory.minimal()

    # Create spec with a tool node
    spec = WorkflowSpecFactory.with_tool_node(node_id="my_tool", tool="gmail.send_email")

    # Create webhook trigger
    trigger = TriggerSpecFactory.webhook()
"""
from typing import Any, Dict, List, Optional

from seer.core.schema.models import (
    Edge,
    EdgeType,
    ToolNode,
    TriggerSpec,
    WorkflowSpec,
)


class WorkflowSpecFactory:
    """Factory for creating WorkflowSpec instances with sensible defaults."""

    @staticmethod
    def create(
        version: str = "2",
        nodes: Optional[List] = None,
        edges: Optional[List] = None,
        triggers: Optional[List] = None,
    ) -> WorkflowSpec:
        """Create a WorkflowSpec with defaults."""
        return WorkflowSpec(
            version=version,
            nodes=nodes or [],
            edges=edges or [],
            triggers=triggers or [],
        )

    @staticmethod
    def minimal() -> WorkflowSpec:
        """Create a minimal valid WorkflowSpec (empty nodes and edges)."""
        return WorkflowSpec(version="2", nodes=[], edges=[])

    @staticmethod
    def with_tool_node(
        node_id: str = "tool_1",
        tool: str = "test.tool",
        inputs: Optional[Dict[str, Any]] = None,
    ) -> WorkflowSpec:
        """Create a WorkflowSpec with a single tool node."""
        return WorkflowSpec(
            version="2",
            nodes=[ToolNode(id=node_id, tool=tool, inputs=inputs or {})],
            edges=[],
        )

    @staticmethod
    def with_chain(node_count: int = 2, tool_prefix: str = "tool") -> WorkflowSpec:
        """
        Create a WorkflowSpec with a chain of tool nodes.

        Args:
            node_count: Number of nodes in the chain (default 2)
            tool_prefix: Prefix for tool IDs (default "tool")

        Returns:
            WorkflowSpec with nodes connected in sequence
        """
        nodes = [
            ToolNode(id=f"n{i}", tool=f"{tool_prefix}_{i}", inputs={})
            for i in range(1, node_count + 1)
        ]
        edges = [
            Edge(source=f"n{i}", target=f"n{i+1}", type=EdgeType.default)
            for i in range(1, node_count)
        ]
        return WorkflowSpec(version="2", nodes=nodes, edges=edges)

    @staticmethod
    def with_trigger(
        trigger_id: str = "trigger_1",
        trigger_key: str = "webhook.generic",
        mode: str = "webhook",
    ) -> WorkflowSpec:
        """Create a WorkflowSpec with a single trigger."""
        return WorkflowSpec(
            version="2",
            nodes=[],
            edges=[],
            triggers=[
                TriggerSpec(id=trigger_id, key=trigger_key, mode=mode)
            ],
        )


class TriggerSpecFactory:
    """Factory for creating TriggerSpec instances with sensible defaults."""

    @staticmethod
    def create(
        id: str = "trigger_1",  # pylint: disable=redefined-builtin  # Reason: Matches TriggerSpec field name
        key: str = "webhook.generic",
        mode: str = "webhook",
        event_schema: Optional[Dict] = None,
        ui_meta: Optional[Dict] = None,
        provider_config: Optional[Dict] = None,
        meta: Optional[Dict] = None,
    ) -> TriggerSpec:
        """Create a TriggerSpec with defaults."""
        kwargs: Dict[str, Any] = {
            "id": id,
            "key": key,
            "mode": mode,
        }
        if event_schema is not None:
            kwargs["event_schema"] = event_schema
        if ui_meta is not None:
            kwargs["ui_meta"] = ui_meta
        if provider_config is not None:
            kwargs["provider_config"] = provider_config
        if meta is not None:
            kwargs["meta"] = meta
        return TriggerSpec(**kwargs)

    @staticmethod
    def webhook(
        id: str = "webhook_1",  # pylint: disable=redefined-builtin  # Reason: Matches TriggerSpec field name
        key: str = "webhook.generic",
    ) -> TriggerSpec:
        """Create a webhook TriggerSpec."""
        return TriggerSpec(id=id, key=key, mode="webhook")

    @staticmethod
    def polling(
        id: str = "poll_1",  # pylint: disable=redefined-builtin  # Reason: Matches TriggerSpec field name
        provider: str = "gmail",
        key: Optional[str] = None,
        connection_id: int = 123,
    ) -> TriggerSpec:
        """Create a polling TriggerSpec."""
        return TriggerSpec(
            id=id,
            key=key or f"poll.{provider}.new_item",
            mode="polling",
            provider_config={"provider_connection_id": connection_id},
        )

    @staticmethod
    def with_sample_event(
        id: str = "trigger_1",  # pylint: disable=redefined-builtin  # Reason: Matches TriggerSpec field name
        key: str = "webhook.generic",
        sample_data: Optional[Dict] = None,
    ) -> TriggerSpec:
        """Create a TriggerSpec with sample event data for testing."""
        return TriggerSpec(
            id=id,
            key=key,
            mode="webhook",
            meta={"sample_event": {"data": sample_data or {"message": "test"}}},
        )
