"""
Pydantic models describing the workflow specification.

These definitions are copied verbatim from the shared design doc so that the
compiler stage can rely on a strongly-typed representation of the workflow
JSON payload.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union, Annotated

from pydantic import BaseModel, Field, ConfigDict, model_validator


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
# Inputs
# -----------------------------
class InputType(str, Enum):
    string = "string"
    integer = "integer"
    number = "number"
    boolean = "boolean"
    object = "object"
    array = "array"


class InputDef(StrictModel):
    type: InputType
    description: Optional[str] = None
    default: Optional[JSONValue] = None
    required: bool = True


# -----------------------------
# Nodes
# -----------------------------
class NodeBase(StrictModel):
    id: str = Field(min_length=1)
    type: str
    out: Optional[str] = None
    meta: Dict[str, JSONValue] = Field(default_factory=dict)


class TaskKind(str, Enum):
    set = "set"


class TaskNode(NodeBase):
    type: Literal["task"] = "task"
    kind: TaskKind
    value: Optional[JSONValue] = None
    in_: Dict[str, JSONValue] = Field(default_factory=dict, alias="in")

    # Optional: declare output contract for tasks (esp for kind=set)
    output: Optional[OutputContract] = None

    @model_validator(mode="after")
    def _validate_set(self) -> "TaskNode":
        if self.kind == TaskKind.set and self.value is None:
            raise ValueError('task kind="set" requires "value"')
        return self


class ToolNode(NodeBase):
    type: Literal["tool"] = "tool"
    tool: str = Field(min_length=1)
    in_: Dict[str, JSONValue] = Field(default_factory=dict, alias="in")

    # Usually derived from ToolRegistry at compile time.
    # But allow client to assert expected schema (optional safety/version check).
    expect_output: Optional[OutputContract] = None


class LLMNode(NodeBase):
    type: Literal["llm"] = "llm"
    model: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    in_: Dict[str, JSONValue] = Field(default_factory=dict, alias="in")

    # Key addition: explicitly declare response mode + schema for structured outputs
    output: OutputContract = Field(default_factory=lambda: OutputContract(mode=OutputMode.text))

    # common knobs
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class IfNode(NodeBase):
    type: Literal["if"] = "if"
    condition: str = Field(min_length=1)
    then: List["Node"] = Field(default_factory=list)
    else_: List["Node"] = Field(default_factory=list, alias="else")


class ForEachNode(NodeBase):
    type: Literal["for_each"] = "for_each"
    items: str = Field(min_length=1)  # expression resolving to list
    body: List["Node"] = Field(default_factory=list)
    item_var: str = "item"
    index_var: str = "index"

    # Optional aggregation contract for what out holds after loop
    # (e.g. list of item results / reduce object)
    output: Optional[OutputContract] = None


Node = Annotated[
    Union[TaskNode, ToolNode, LLMNode, IfNode, ForEachNode],
    Field(discriminator="type"),
]


class WorkflowSpec(StrictModel):
    version: str = Field(default="1")
    inputs: Dict[str, InputDef] = Field(default_factory=dict)
    nodes: List[Node] = Field(default_factory=list)
    output: Optional[JSONValue] = None
    meta: Dict[str, JSONValue] = Field(default_factory=dict)


IfNode.model_rebuild()
ForEachNode.model_rebuild()


