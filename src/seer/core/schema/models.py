# pylint: disable=too-many-lines  # schema models are intentionally comprehensive in a single module
"""
Pydantic models describing the workflow specification.

These definitions are copied verbatim from the shared design doc so that the
compiler stage can rely on a strongly-typed representation of the workflow
JSON payload.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from seer.core.registry.default_models import DEFAULT_AGENT_MODEL_IDS

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
    text = "text"  # pylint: disable=invalid-name  # Reason: Enum value matches JSON spec format
    json = "json"  # pylint: disable=invalid-name  # Reason: Enum value matches JSON spec format


def enforce_all_properties_required(schema: JsonSchema, path: str = "$") -> None:
    """
    Recursively assert that every JSON Schema object has all its `properties`
    keys listed in `required`. Raises ValueError with a descriptive path if not.

    This is intentionally NOT called from OutputContract itself — it is called
    during compilation (register_type_sync) only for node types that drive LLM
    structured output (AgentNode, BrowserNode), where the schema must be strict
    so the LLM cannot silently omit fields.
    """
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object" and "properties" in schema:
        declared = set(schema["properties"].keys())
        required = set(schema.get("required") or [])
        missing = declared - required
        if missing:
            raise ValueError(
                f"OutputContract: all properties must be in 'required' for json output schema. "
                f"Missing at '{path}': {sorted(missing)}. Add them to the 'required' array."
            )
        # Recurse into each property's sub-schema
        for prop_name, prop_schema in schema["properties"].items():
            enforce_all_properties_required(prop_schema, path=f"{path}.{prop_name}")
    # Recurse into array items
    if "items" in schema and isinstance(schema["items"], dict):
        enforce_all_properties_required(schema["items"], path=f"{path}[]")


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
    default = "default"  # pylint: disable=invalid-name  # Reason: Enum value matches JSON spec format
    conditional_true = "conditional_true"  # pylint: disable=invalid-name  # Reason: Enum value matches JSON spec format
    conditional_false = "conditional_false"  # pylint: disable=invalid-name  # Reason: Enum value matches JSON spec format
    loop_body = "loop_body"  # pylint: disable=invalid-name  # Reason: Enum value matches JSON spec format
    loop_exit = "loop_exit"  # pylint: disable=invalid-name  # Reason: Enum value matches JSON spec format
    trigger = "trigger"  # pylint: disable=invalid-name  # Reason: Enum value matches JSON spec format


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


class ToolNode(NodeBase):
    type: Literal["tool"] = "tool"
    tool: str = Field(min_length=1)
    inputs: Dict[str, JSONValue] = Field(default_factory=dict)

    # Usually derived from ToolRegistry at compile time.
    # But allow client to assert expected schema (optional safety/version check).
    expect_outputs: Optional[OutputContract] = None

    # Explicit OAuth connection ID for multi-account scenarios.
    # When user has multiple accounts for the same provider (e.g., personal + work Gmail),
    # this field specifies which account to use. If omitted and only one account exists,
    # it's auto-selected. If omitted with multiple accounts, validation fails.
    connection_id: Optional[int] = None


class MCPNode(NodeBase):
    """
    MCP (Model Context Protocol) node for invoking tools from external MCP servers.

    Supports both HTTP and stdio MCP servers with optional authentication.
    Auth expressions like ${secrets.api_key} are resolved at runtime.
    """
    type: Literal["mcp"] = "mcp"
    server: str = Field()
    server_type: Literal["http", "stdio"] = "http"
    auth: Optional[Dict[str, Any]] = None
    tool: str = Field()
    inputs: Dict[str, JSONValue] = Field(default_factory=dict)

    # Optional: declare expected output contract for validation
    expect_outputs: Optional[OutputContract] = None


class IfNode(NodeBase):
    """
    Conditional node that routes to different branches based on condition.

    Branch targets are defined by edges with type=conditional_true/conditional_false.
    """
    type: Literal["if"] = "if"
    condition: str = ""


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


# -----------------------------
# HITL (Human-In-The-Loop) Node
# -----------------------------
class DeliveryChannelType(str, Enum):
    """Types of delivery channels for HITL notifications."""
    platform = "platform"  # pylint: disable=invalid-name  # Reason: Enum value matches JSON spec format
    gmail = "gmail"  # pylint: disable=invalid-name  # Reason: Enum value matches JSON spec format


class GmailDeliveryConfig(StrictModel):
    """Configuration for Gmail delivery channel."""
    recipient_email: str = Field(description="Email address to send HITL form link to")


class HITLDeliveryChannel(StrictModel):
    """
    Delivery channel configuration for HITL notifications.

    Supports multiple delivery methods:
    - platform: Default web-based HITL (existing behavior)
    - gmail: Send email with form link via Gmail
    """
    type: DeliveryChannelType = DeliveryChannelType.platform
    gmail: Optional[GmailDeliveryConfig] = None

    @model_validator(mode="after")
    def _validate_channel_config(self) -> "HITLDeliveryChannel":
        """Validate that Gmail channel has required config."""
        if self.type == DeliveryChannelType.gmail and self.gmail is None:
            raise ValueError("HITLDeliveryChannel with type='gmail' requires gmail config")
        return self


class HITLInputType(str, Enum):
    """Input types for HITL node user input collection."""
    single_choice = "single_choice"  # pylint: disable=invalid-name  # Reason: Enum value matches JSON spec format
    multi_choice = "multi_choice"  # pylint: disable=invalid-name  # Reason: Enum value matches JSON spec format
    text = "text"  # pylint: disable=invalid-name  # Reason: Enum value matches JSON spec format
    number = "number"  # pylint: disable=invalid-name  # Reason: Enum value matches JSON spec format
    boolean = "boolean"  # pylint: disable=invalid-name  # Reason: Enum value matches JSON spec format
    table = "table"  # pylint: disable=invalid-name  # Reason: Enum value matches JSON spec format


class HITLInputOption(StrictModel):
    """Option for choice-based HITL inputs."""
    value: str = Field()
    label: str = Field()
    requires_text: bool = False  # If true, selecting this option prompts for additional text input


class HITLTableColumn(StrictModel):
    """Column definition for table input type."""
    id: str = Field(min_length=1)
    header: str
    input_type: HITLInputType  # text, single_choice, boolean, number (not nested table)
    options: Optional[List[HITLInputOption]] = None
    required: bool = True


class HITLDisplayItem(StrictModel):
    """Display item shown to user during HITL interrupt."""
    label: str = ""
    value: str = ""  # Expression like ${node.field}


def _validate_table_columns(columns: List["HITLTableColumn"]) -> None:
    """Validate table column definitions."""
    for col in columns:
        if col.input_type == HITLInputType.table:
            raise ValueError("Table columns cannot have input_type=table (no nesting)")
        if col.input_type in (HITLInputType.single_choice, HITLInputType.multi_choice):
            if not col.options or len(col.options) < 2:
                raise ValueError(f"Table column '{col.id}' with input_type={col.input_type.value} requires at least 2 options")


class HITLInputField(StrictModel):
    """Input field definition for collecting user responses."""
    id: str = Field(min_length=1)
    question: str = Field()
    input_type: HITLInputType
    options: Optional[List[HITLInputOption]] = None  # Required for choice types
    required: bool = True
    placeholder: Optional[str] = None
    default_value: Optional[JSONValue] = None
    columns: Optional[List["HITLTableColumn"]] = None
    row_data_expression: Optional[str] = None  # e.g. "${fetch_highlights.body.results}"
    row_display_fields: Optional[List["HITLDisplayItem"]] = None

    @model_validator(mode="after")
    def _validate_options(self) -> "HITLInputField":
        """Validate that choice types have options and table types have columns."""
        if self.input_type in (HITLInputType.single_choice, HITLInputType.multi_choice):
            if not self.options or len(self.options) < 2:
                raise ValueError(f"HITLInputField with input_type={self.input_type.value} requires at least 2 options")
        elif self.input_type == HITLInputType.table:
            if not self.columns:
                raise ValueError("HITLInputField with input_type=table requires columns")
            if not self.row_data_expression:
                raise ValueError("HITLInputField with input_type=table requires row_data_expression")
            _validate_table_columns(self.columns)  # type: ignore[arg-type]  # guarded by `if not self.columns` above
        elif self.options:
            raise ValueError(f"HITLInputField with input_type={self.input_type.value} should not have options")
        return self


class HITLNode(NodeBase):
    """
    Human-In-The-Loop node that pauses workflow execution to collect user input.

    Uses LangGraph's interrupt() mechanism to pause execution and wait for
    user response before continuing.

    Supports multiple delivery channels for notifications:
    - platform: Default web-based HITL (user polls /runs/{id}/interrupt)
    - gmail: Email with form link sent via Gmail
    """
    type: Literal["hitl"] = "hitl"
    title: str = Field()
    description: Optional[str] = None
    display: List[HITLDisplayItem] = Field(default_factory=list)
    inputs: List[HITLInputField] = Field(default_factory=list)
    timeout_seconds: Optional[int] = None  # null or 0 = indefinite wait
    delivery_channels: List[HITLDeliveryChannel] = Field(
        default_factory=lambda: [HITLDeliveryChannel(type=DeliveryChannelType.platform)],
        description="Notification delivery channels (defaults to platform only)"
    )

    @model_validator(mode="after")
    def _validate_inputs(self) -> "HITLNode":
        """Validate unique input IDs."""
        seen_ids = set()
        duplicate_ids = []
        for input_field in self.inputs:
            if input_field.id in seen_ids:
                duplicate_ids.append(input_field.id)
            seen_ids.add(input_field.id)
        if duplicate_ids:
            dup_list = ", ".join(sorted(set(duplicate_ids)))
            raise ValueError(f"HITLNode has duplicate input IDs: {dup_list}")
        return self


# -----------------------------
# Browser Automation Node
# -----------------------------
ALLOWED_IMAGE_GEN_MODELS = frozenset({
    "sourceful/riverflow-v2-fast",
    "google/gemini-2.5-flash-image",
})


class ImageGenNode(NodeBase):
    """
    Image generation node using OpenRouter image generation models.

    Calls OpenRouter API with an image generation model and returns
    the generated image URL/base64.
    """
    type: Literal["image_gen"] = "image_gen"
    inputs: Dict[str, JSONValue] = Field(default_factory=dict)
    # inputs expected: model, prompt, size (optional), num_images (optional)

    @model_validator(mode="after")
    def _validate_inputs(self) -> "ImageGenNode":
        required = ["model", "prompt"]
        missing = [k for k in required if k not in self.inputs]
        if missing:
            raise ValueError(f'ImageGenNode requires {", ".join(missing)} in inputs')

        # pylint: disable=no-member  # Pydantic resolves Field() to dict at runtime
        model = self.inputs.get("model")
        if model not in ALLOWED_IMAGE_GEN_MODELS:
            allowed = ", ".join(sorted(ALLOWED_IMAGE_GEN_MODELS))
            raise ValueError(
                f"ImageGenNode model '{model}' is not allowed. "
                f"Allowed models: {allowed}"
            )
        return self


class BrowserNode(NodeBase):
    """
    Browser automation node using natural language task descriptions.

    Executes browser automation tasks via BrowserUse Agent. Users manage
    browser profiles separately (with saved login sessions), and workflows
    reference profiles by ID for authenticated automation.

    Example:
        {
            "id": "scrape_data",
            "type": "browser",
            "task": "Go to Slack and get the last 10 messages from #general",
            "browser_profile_id": "uuid-of-profile"
        }
    """
    type: Literal["browser"] = "browser"

    # Natural language task description (supports ${expressions})
    task: str = Field()

    # Additional context data passed to the browser agent
    inputs: Dict[str, JSONValue] = Field(default_factory=dict)

    # Reference to user's browser profile with saved login sessions
    # Profiles are managed via /browser/profiles API endpoints
    browser_profile_id: Optional[str] = None

    # Execution limits
    max_steps: int = Field(default=25, ge=1, le=100)
    timeout_seconds: int = Field(default=300, ge=30, le=1800)

    # Optional output contract for structured output validation
    expect_outputs: Optional[OutputContract] = None

    # Screenshot saving configuration
    save_screenshots: bool = Field(
        default=False,
        description="When True, save screenshots to S3 and return WorkflowFileRef objects"
    )

    # LLM model for browser automation (OpenRouter format)
    model: Optional[str] = Field(
        default="qwen/qwen3-vl-8b-thinking",
        description=(
            "OpenRouter model identifier for browser automation agent. "
            "Examples: 'openai/gpt-4o', 'anthropic/claude-sonnet-4.5', 'qwen/qwen3-vl-8b-thinking'. "
            "Defaults to 'qwen/qwen3-vl-8b-thinking' for browser tasks."
        )
    )


class AgentNode(NodeBase):
    """
    Agent node for autonomous multi-step task execution with tool access.

    Uses LangGraph's react_agent internally to enable reasoning, tool calling,
    and iteration until the task is complete. Supports state resolution,
    OAuth credential binding for tools, and comprehensive tracing.

    Example:
        {
            "id": "research_agent",
            "type": "agent",
            "inputs": {
                "model": "gpt-4",
                "prompt": "Find information about ${topic} and summarize the key points",
                "tools": ["web_search", {"name": "gmail_send_email", "connection_id": 42}],
                "max_iterations": 10
            },
            "outputs": {"mode": "text"}
        }
    """
    type: Literal["agent"] = "agent"
    inputs: Dict[str, JSONValue] = Field(default_factory=dict)
    outputs: OutputContract = Field(default_factory=lambda: OutputContract(mode=OutputMode.text))

    @model_validator(mode="after")
    def _validate_agent_inputs(self) -> "AgentNode":
        """Validate required inputs and tool format."""
        inputs = cast(Dict[str, JSONValue], dict(self.inputs))
        required = ["model", "prompt"]
        missing = [key for key in required if key not in inputs]
        if missing:
            raise ValueError(f'AgentNode requires {", ".join(missing)} in inputs')

        model = inputs.get("model")
        if model not in DEFAULT_AGENT_MODEL_IDS:
            allowed = ", ".join(sorted(DEFAULT_AGENT_MODEL_IDS))
            raise ValueError(
                f"AgentNode model '{model}' is not allowed. "
                f"Allowed models: {allowed}"
            )

        self._validate_tool_specs(inputs.get("tools", []))
        self._validate_max_iterations(inputs.get("max_iterations"))

        memory_bank_id = inputs.get("memory_bank_id")
        if memory_bank_id is not None and not isinstance(memory_bank_id, str):
            raise ValueError("AgentNode 'memory_bank_id' must be a string")

        return self

    @staticmethod
    def _validate_tool_specs(tools: JSONValue) -> None:
        """Validate agent tool configuration."""
        if not isinstance(tools, list):
            raise ValueError("AgentNode 'tools' must be a list")

        for index, tool in enumerate(tools):
            if isinstance(tool, dict):
                if "name" not in tool:
                    raise ValueError(f"AgentNode tool at index {index} must have 'name' field")
                continue
            if not isinstance(tool, str):
                raise ValueError(f"AgentNode tool at index {index} must be string or {{name, connection_id}} object")

    @staticmethod
    def _validate_max_iterations(max_iterations: JSONValue) -> None:
        """Validate the optional max_iterations input."""
        if max_iterations is not None and (not isinstance(max_iterations, int) or max_iterations < 1):
            raise ValueError("AgentNode 'max_iterations' must be a positive integer")


Node = Annotated[
    Union[ToolNode, MCPNode, IfNode, ForEachNode, HITLNode, ImageGenNode, BrowserNode, AgentNode],
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
    required_scopes: Optional[List[str]] = None  # OAuth scopes required for this trigger


class TriggerIdentity(StrictModel):
    """Shared trigger identity fields used across API and compiler models."""
    key: str
    title: str
    provider: str
    mode: str
    description: Optional[str] = None


class TriggerDefinition(TriggerIdentity):
    """Complete trigger definition with identity, schemas, and metadata."""
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
    def _validate_workflow(self) -> "WorkflowSpec":  # pylint: disable=too-complex  # Reason: Comprehensive workflow validation requires multiple checks
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
