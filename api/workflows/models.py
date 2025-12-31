from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from workflow_compiler.schema.models import WorkflowSpec


class ProblemError(BaseModel):
    code: str
    message: str
    node_id: Optional[str] = None
    location: Optional[str] = None
    expression: Optional[str] = None


class ProblemDetails(BaseModel):
    type: str
    title: str
    status: int
    detail: str
    errors: List[ProblemError] = Field(default_factory=list)


class NodeFieldDescriptor(BaseModel):
    name: str
    kind: str
    required: bool = False
    source: Optional[str] = None


class NodeTypeDescriptor(BaseModel):
    type: str
    title: str
    fields: List[NodeFieldDescriptor]


class NodeTypeResponse(BaseModel):
    node_types: List[NodeTypeDescriptor]


class ToolDescriptor(BaseModel):
    id: str
    name: str
    version: str
    title: str
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None


class ToolRegistryResponse(BaseModel):
    tools: List[ToolDescriptor]


class TriggerDescriptor(BaseModel):
    key: str
    title: str
    provider: str
    mode: str
    description: Optional[str] = None
    event_schema: Dict[str, Any]
    filter_schema: Optional[Dict[str, Any]] = None


class TriggerCatalogResponse(BaseModel):
    triggers: List[TriggerDescriptor]


class TriggerSubscriptionCreateRequest(BaseModel):
    workflow_id: str
    trigger_key: str
    provider_connection_id: Optional[int] = None
    enabled: bool = True
    filters: Dict[str, Any] = Field(default_factory=dict)
    bindings: Dict[str, Any] = Field(default_factory=dict)
    provider_config: Dict[str, Any] = Field(default_factory=dict)


class TriggerSubscriptionUpdateRequest(BaseModel):
    provider_connection_id: Optional[int] = None
    enabled: Optional[bool] = None
    filters: Optional[Dict[str, Any]] = None
    bindings: Optional[Dict[str, Any]] = None
    provider_config: Optional[Dict[str, Any]] = None


class TriggerSubscriptionResponse(BaseModel):
    subscription_id: int
    workflow_id: str
    trigger_key: str
    provider_connection_id: Optional[int] = None
    enabled: bool
    filters: Dict[str, Any] = Field(default_factory=dict)
    bindings: Dict[str, Any] = Field(default_factory=dict)
    provider_config: Dict[str, Any] = Field(default_factory=dict)
    secret_token: Optional[str] = None
    webhook_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class TriggerSubscriptionListResponse(BaseModel):
    items: List[TriggerSubscriptionResponse] = Field(default_factory=list)


class TriggerSubscriptionTestRequest(BaseModel):
    event: Optional[Dict[str, Any]] = None


class TriggerSubscriptionTestResponse(BaseModel):
    inputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)


class ModelDescriptor(BaseModel):
    id: str
    title: str
    supports_json_schema: bool = True


class ModelRegistryResponse(BaseModel):
    models: List[ModelDescriptor]


class SchemaResponse(BaseModel):
    id: str
    json_schema: Dict[str, Any]


class WorkflowWarning(BaseModel):
    code: str
    node_id: str
    message: str


class ValidateRequest(BaseModel):
    spec: WorkflowSpec


class ValidateResponse(BaseModel):
    ok: bool = True
    warnings: List[WorkflowWarning] = Field(default_factory=list)


class CompileOptions(BaseModel):
    emit_graph_preview: bool = False
    emit_type_env: bool = False
    strict_task_output: bool = False


class CompileRequest(BaseModel):
    spec: WorkflowSpec
    options: CompileOptions = Field(default_factory=CompileOptions)


class CompileArtifacts(BaseModel):
    type_env: Optional[Dict[str, Any]] = None
    graph_preview: Optional[Dict[str, Any]] = None


class CompileResponse(BaseModel):
    ok: bool = True
    warnings: List[WorkflowWarning] = Field(default_factory=list)
    artifacts: CompileArtifacts = Field(default_factory=CompileArtifacts)


class WorkflowMeta(BaseModel):
    last_compile_ok: bool = False


class WorkflowBase(BaseModel):
    name: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class WorkflowCreateRequest(WorkflowBase):
    spec: WorkflowSpec


class WorkflowUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    spec: Optional[WorkflowSpec] = None
    tags: Optional[List[str]] = None


class WorkflowSummary(BaseModel):
    workflow_id: str
    name: str
    description: Optional[str] = None
    version: int
    created_at: datetime
    updated_at: datetime


class WorkflowResponse(WorkflowSummary):
    spec: WorkflowSpec
    tags: List[str] = Field(default_factory=list)
    meta: WorkflowMeta = Field(default_factory=WorkflowMeta)


class WorkflowListResponse(BaseModel):
    items: List[WorkflowSummary]
    next_cursor: Optional[str] = None


class RunFromSpecRequest(BaseModel):
    spec: WorkflowSpec
    inputs: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)


class RunFromWorkflowRequest(BaseModel):
    version: Optional[int] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)


class RunProgress(BaseModel):
    completed: int = 0
    total: int = 0


class RunResponse(BaseModel):
    run_id: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    progress: Optional[RunProgress] = None
    current_node_id: Optional[str] = None
    last_error: Optional[str] = None


class RunResultResponse(BaseModel):
    run_id: str
    status: str
    output: Optional[Dict[str, Any]] = None
    state: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None


class RunHistoryResponse(BaseModel):
    run_id: str
    history: List[Dict[str, Any]] = Field(default_factory=list)


class WorkflowRunSummary(BaseModel):
    run_id: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class WorkflowRunListResponse(BaseModel):
    workflow_id: str
    runs: List[WorkflowRunSummary] = Field(default_factory=list)


class ExpressionCursorContext(BaseModel):
    node_id: Optional[str] = None
    field: Optional[str] = None
    prefix: str


class ExpressionSuggestRequest(BaseModel):
    spec: WorkflowSpec
    cursor_context: ExpressionCursorContext


class ExpressionSuggestion(BaseModel):
    label: str
    insert: str
    type: Optional[str] = None


class ExpressionSuggestResponse(BaseModel):
    suggestions: List[ExpressionSuggestion] = Field(default_factory=list)


class ExpressionTypecheckRequest(BaseModel):
    spec: WorkflowSpec
    expression: str
    scope: Optional[Dict[str, Any]] = None


class ExpressionTypecheckResponse(BaseModel):
    ok: bool = True
    type: Optional[Dict[str, Any]] = None


__all__ = [
    "ProblemDetails",
    "ProblemError",
    "NodeFieldDescriptor",
    "NodeTypeDescriptor",
    "NodeTypeResponse",
    "ToolDescriptor",
    "ToolRegistryResponse",
    "TriggerDescriptor",
    "TriggerCatalogResponse",
    "TriggerSubscriptionCreateRequest",
    "TriggerSubscriptionUpdateRequest",
    "TriggerSubscriptionResponse",
    "TriggerSubscriptionListResponse",
    "TriggerSubscriptionTestRequest",
    "TriggerSubscriptionTestResponse",
    "ModelDescriptor",
    "ModelRegistryResponse",
    "SchemaResponse",
    "WorkflowWarning",
    "ValidateRequest",
    "ValidateResponse",
    "CompileOptions",
    "CompileRequest",
    "CompileResponse",
    "CompileArtifacts",
    "WorkflowMeta",
    "WorkflowCreateRequest",
    "WorkflowUpdateRequest",
    "WorkflowResponse",
    "WorkflowSummary",
    "WorkflowListResponse",
    "RunFromSpecRequest",
    "RunFromWorkflowRequest",
    "RunResponse",
    "RunResultResponse",
    "RunHistoryResponse",
    "WorkflowRunSummary",
    "WorkflowRunListResponse",
    "ExpressionSuggestRequest",
    "ExpressionSuggestResponse",
    "ExpressionTypecheckRequest",
    "ExpressionTypecheckResponse",
    "ExpressionSuggestion",
]


