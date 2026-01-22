"""
Pydantic models describing the workflow specification.

These definitions are copied verbatim from the shared design doc so that the
compiler stage can rely on a strongly-typed representation of the workflow
JSON payload.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# -----------------------------
# JSON-ish values
# -----------------------------
# NOTE: Pydantic struggles with recursive type aliases when generating schemas,
# so we approximate JSONValue using non-recursive containers to avoid
# RecursionError during workflow parsing while still keeping loose typing.
JSONPrimitive = Union[str, int, float, bool, None]
JSONValue = Union[JSONPrimitive, Dict[str, Any], List[Any]]
JsonSchema = Dict[str, Any]  # draft-07/2020-12 style dict


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )


# -----------------------------
# Schema references (type safety contract)
# -----------------------------
class SchemaRef(StrictModel):
    """
    A reference to a schema known to the engine (or shared with clients).

    Examples:
      - "tools.github.search_issues@v1.output"
      - "schemas.IssueSummary@v2"
    """

    id: str = Field(min_length=1)


class InlineSchema(StrictModel):
    """
    Inline JSON Schema (clients can supply it since they know tool schemas).
    """

    json_schema: JsonSchema = Field(..., alias="schema")


SchemaSpec = Union[SchemaRef, InlineSchema]


class OutputMode(str, Enum):
    text = "text"
    json = "json"


class OutputContract(StrictModel):
    """
    Declares what a node writes to state[out].

    - mode=text -> a string
    - mode=json -> validated object per schema
    """

    mode: OutputMode = OutputMode.json
    schema: Optional[SchemaSpec] = None  # required if mode=json

    @model_validator(mode="after")
    def _check_schema_when_json(self) -> "OutputContract":
        if self.mode == OutputMode.json and self.schema is None:
            raise ValueError('OutputContract: schema is required when mode="json"')
        if self.mode == OutputMode.text and self.schema is not None:
            raise ValueError('OutputContract: schema must be omitted when mode="text"')
        return self


# -----------------------------
# Edges
# -----------------------------
class EdgeType(str, Enum):
    """Types of edges in the workflow graph."""
    default = "default"                      # Sequential flow
    conditional_true = "conditional_true"    # If condition true branch
    conditional_false = "conditional_false"  # If condition false branch
    loop_body = "loop_body"                  # For-each loop body entry
    loop_exit = "loop_exit"                  # For-each loop exit
    trigger = "trigger"                      # Trigger to node entry point


class Edge(StrictModel):
    """
    Explicit edge connecting two nodes in the workflow graph.
    """
    source: str = Field(min_length=1)  # Source node ID
    target: str = Field(min_length=1)  # Target node ID
    type: EdgeType = EdgeType.default
    ui: Dict[str, JSONValue] = Field(default_factory=dict)


# -----------------------------
# Nodes
# -----------------------------
class NodeBase(StrictModel):
    id: str = Field(min_length=1)
    type: str
    ui: Dict[str, JSONValue] = Field(default_factory=dict)


class TaskKind(str, Enum):
    set = "set"


class TaskNode(NodeBase):
    type: Literal["task"] = "task"
    kind: TaskKind
    value: Optional[JSONValue] = None
    inputs: Dict[str, JSONValue] = Field(default_factory=dict)

    # Optional: declare output contract for tasks (esp for kind=set)
    outputs: Optional[OutputContract] = None

    @model_validator(mode="after")
    def _validate_set(self) -> "TaskNode":
        if self.kind == TaskKind.set and self.value is None:
            raise ValueError('task kind="set" requires "value"')
        return self


class ToolNode(NodeBase):
    type: Literal["tool"] = "tool"
    tool: str = Field(min_length=1)
    inputs: Dict[str, JSONValue] = Field(default_factory=dict)

    # Usually derived from ToolRegistry at compile time.
    # But allow client to assert expected schema (optional safety/version check).
    expect_outputs: Optional[OutputContract] = None


class LLMNode(NodeBase):
    type: Literal["llm"] = "llm"
    inputs: Dict[str, JSONValue] = Field(default_factory=dict)

    # Key addition: explicitly declare response mode + schema for structured outputs
    outputs: OutputContract = Field(default_factory=lambda: OutputContract(mode=OutputMode.text))

    @model_validator(mode="after")
    def _validate_llm_inputs(self) -> "LLMNode":
        required = ["model", "prompt"]
        missing = [k for k in required if k not in self.inputs]
        if missing:
            raise ValueError(f'LLMNode requires {", ".join(missing)} in inputs')
        return self


class IfNode(NodeBase):
    """
    Conditional node that routes to different branches based on condition.

    Branch targets are defined by edges with type=conditional_true/conditional_false.
    """
    type: Literal["if"] = "if"
    condition: str = Field(min_length=1)


class ForEachNode(NodeBase):
    """
    Loop node that iterates over a list.

    Loop body is defined by edges with type=loop_body.
    Loop exit is defined by edges with type=loop_exit.
    The ForEachNode writes item_var and index_var to state for body nodes.
    """
    type: Literal["for_each"] = "for_each"
    items: str = Field(min_length=1)  # expression resolving to list
    item_var: str = "item"
    index_var: str = "index"

    # Optional aggregation contract for what out holds after loop
    # (e.g. list of item results / reduce object)
    outputs: Optional[OutputContract] = None


Node = Annotated[
    Union[TaskNode, ToolNode, LLMNode, IfNode, ForEachNode],
    Field(discriminator="type"),
]


class TriggerSchemas(StrictModel):
    """Schema definitions for trigger validation and configuration."""
    # event schema is the schema of the event that is received from the trigger
    event: JsonSchema = Field(default_factory=dict)
    filter: Optional[JsonSchema] = None

    # config schema is the schema of the configuration that is used to configure the trigger
    config: Optional[JsonSchema] = None


class TriggerMetadata(StrictModel):
    """Metadata and defaults for trigger configuration."""
    sample_event: Optional[Dict[str, Any]] = None
    requires_connection: bool = True


class TriggerDefinition(StrictModel):
    """Complete trigger definition with identity, schemas, and metadata."""
    key: str
    title: str
    provider: str
    mode: str
    description: Optional[str] = None
    schemas: TriggerSchemas = Field(default_factory=TriggerSchemas)
    meta: TriggerMetadata = Field(default_factory=TriggerMetadata)

class TriggerSpec(StrictModel):
    """
    Declarative trigger configuration embedded in the workflow spec.

    Frontend supplies this alongside nodes so triggers can be versioned with the workflow.
    No longer inherits from TriggerDefinition - only includes fields needed for workflow execution.
    """

    # Unique instance identifier (allows multiple triggers of same type)
    id: str = Field(min_length=1)

    # Trigger type and mode
    key: str  # Trigger type key (e.g., "gmail.new_email")
    mode: str  # "polling", "webhook", etc.

    # Event schema (flattened from schemas.event)
    event_schema: JsonSchema = Field(default_factory=dict)

    # Metadata
    meta: TriggerMetadata = Field(default_factory=TriggerMetadata)

    # Configuration
    filters: Dict[str, JSONValue] = Field(default_factory=dict)
    provider_config: Dict[str, JSONValue] = Field(default_factory=dict)
    ui_meta: Dict[str, JSONValue] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_fields(cls, data: Any) -> Any:
        """Support legacy TriggerSpec format during migration."""
        if not isinstance(data, dict):
            return data

        # Migrate schemas.event → event_schema
        if "schemas" in data and "event_schema" not in data:
            schemas = data.pop("schemas")
            if isinstance(schemas, dict):
                data["event_schema"] = schemas.get("event", {})
            else:
                data["event_schema"] = schemas.event if hasattr(schemas, "event") else {}

        # Migrate provider_connection_id → provider_config
        if "provider_connection_id" in data:
            conn_id = data.pop("provider_connection_id")
            if conn_id is not None:
                data.setdefault("provider_config", {})["provider_connection_id"] = conn_id

        # Remove fields no longer part of spec
        data.pop("title", None)
        data.pop("description", None)
        data.pop("provider", None)
        data.pop("enabled", None)

        return data


class WorkflowSpec(StrictModel):
    version: str = Field(default="2")

    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    triggers: List[TriggerSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_workflow(self) -> "WorkflowSpec":
        # Validate unique trigger IDs (allow duplicate keys for same type)
        seen_trigger_ids = set()
        duplicate_trigger_ids = []
        for trigger in self.triggers or []:
            if trigger.id in seen_trigger_ids:
                duplicate_trigger_ids.append(trigger.id)
            seen_trigger_ids.add(trigger.id)
        if duplicate_trigger_ids:
            dup_list = ", ".join(sorted(set(duplicate_trigger_ids)))
            raise ValueError(f"Duplicate trigger id values are not allowed: {dup_list}")

        # Collect valid identifiers
        node_ids = {n.id for n in self.nodes}
        trigger_ids = {t.id for t in self.triggers}

        # Validate edge source/target references
        for edge in self.edges:
            if edge.type == EdgeType.trigger:
                # Trigger edges: source must be a trigger id, target must be a node
                if edge.source not in trigger_ids:
                    raise ValueError(f"Trigger edge with source '{edge.source}' and target '{edge.target}' not found in triggers")
                if edge.target not in node_ids:
                    raise ValueError(f"Trigger edge with source '{edge.source}' and target '{edge.target}' not found in nodes")
            else:
                # Regular edges: source and target must be nodes
                if edge.source not in node_ids:
                    raise ValueError(f"Edge with source '{edge.source}' and target '{edge.target}' source not found in nodes")
                if edge.target not in node_ids:
                    raise ValueError(f"Edge with source '{edge.source}' and target '{edge.target}' target not found in nodes")

        # Validate unique node IDs
        seen_nodes = set()
        duplicate_nodes = []
        for node in self.nodes:
            if node.id in seen_nodes:
                duplicate_nodes.append(node.id)
            seen_nodes.add(node.id)
        if duplicate_nodes:
            dup_list = ", ".join(sorted(set(duplicate_nodes)))
            raise ValueError(f"Duplicate node id values are not allowed: {dup_list}")

        return self
