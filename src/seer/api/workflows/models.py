from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from seer.core.schema.models import TriggerIdentity, WorkflowSpec


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


class TriggerDescriptor(TriggerIdentity):
    event_schema: Dict[str, Any]
    filter_schema: Optional[Dict[str, Any]] = None
    config_schema: Optional[Dict[str, Any]] = None
    is_connected: bool = True


class TriggerCatalogResponse(BaseModel):
    triggers: List[TriggerDescriptor]


class TriggerSubscriptionCreateRequest(BaseModel):
    workflow_id: str
    trigger_key: str
    provider_connection_id: Optional[int] = None
    enabled: bool = True
    filters: Dict[str, Any] = Field(default_factory=dict)
    provider_config: Dict[str, Any] = Field(default_factory=dict)
    # Form trigger fields
    form_suffix: Optional[str] = None
    form_fields: Optional[List[Dict[str, Any]]] = None
    form_config: Optional[Dict[str, Any]] = None


class TriggerSubscriptionUpdateRequest(BaseModel):
    provider_connection_id: Optional[int] = None
    enabled: Optional[bool] = None
    filters: Optional[Dict[str, Any]] = None
    provider_config: Optional[Dict[str, Any]] = None


class TriggerSubscriptionResponse(BaseModel):
    subscription_id: int
    workflow_id: str
    trigger_key: str
    provider_connection_id: Optional[int] = None
    enabled: bool
    filters: Dict[str, Any] = Field(default_factory=dict)
    provider_config: Dict[str, Any] = Field(default_factory=dict)
    secret_token: Optional[str] = None
    webhook_url: Optional[str] = None
    form_url: Optional[str] = None
    # Form trigger fields
    form_suffix: Optional[str] = None
    form_fields: Optional[List[Dict[str, Any]]] = None
    form_config: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime


class TriggerSubscriptionListResponse(BaseModel):
    items: List[TriggerSubscriptionResponse] = Field(default_factory=list)


class TriggerSubscriptionTestRequest(BaseModel):
    event: Optional[Dict[str, Any]] = None


class TriggerSubscriptionTestResponse(BaseModel):
    inputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)


class StartListeningResponse(BaseModel):
    webhook_url: str
    secret_token: str
    subscription_id: int


class PendingEventItem(BaseModel):
    event_id: int
    data: Dict[str, Any]
    received_at: str


class PendingEventsResponse(BaseModel):
    events: List[PendingEventItem] = Field(default_factory=list)
    latest_event_id: Optional[int] = None


class ModelDescriptor(BaseModel):
    id: str
    title: str
    supports_json_schema: bool = True


class ModelRegistryResponse(BaseModel):
    models: List[ModelDescriptor]


class SchemaResponse(BaseModel):
    id: str
    json_schema: Dict[str, Any]


class SchemaMetadataGenerateRequest(BaseModel):
    """Request payload for schema metadata generation."""
    json_schema: Dict[str, Any] = Field(
        ...,
        description="JSON Schema object with properties to analyze"
    )


class SchemaMetadataGenerateResponse(BaseModel):
    """Response with generated schema metadata."""
    title: str = Field(..., description="Generated schema title (PascalCase, 2-4 words)")
    description: str = Field(..., description="Generated schema description (1-2 sentences)")


class McpToolsRequest(BaseModel):
    server: str
    server_type: str = "http"
    auth: Optional[Dict[str, Any]] = None


class McpToolDescriptor(BaseModel):
    name: str
    description: str = ""
    input_schema: Optional[Dict[str, Any]] = None


class McpToolsResponse(BaseModel):
    tools: List[McpToolDescriptor]


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


class WorkflowBase(BaseModel):
    name: str


class WorkflowCreateRequest(WorkflowBase):
    spec: WorkflowSpec


class WorkflowUpdateRequest(BaseModel):
    name: Optional[str] = None


class WorkflowDraftPatchRequest(BaseModel):
    spec: WorkflowSpec


class WorkflowPublishRequest(BaseModel):
    pass


class WorkflowVersionRestoreRequest(BaseModel):
    pass


class WorkflowVersionSummary(BaseModel):
    version_id: int
    status: str
    version_number: Optional[int] = None
    created_from_draft_revision: Optional[int] = None
    created_at: datetime


class WorkflowVersionListItem(WorkflowVersionSummary):
    is_latest: bool = False
    is_published: bool = False


class WorkflowVersionListResponse(BaseModel):
    workflow_id: str
    versions: List[WorkflowVersionListItem] = Field(default_factory=list)


class WorkflowSummary(BaseModel):
    workflow_id: str
    name: str
    created_at: datetime
    updated_at: datetime


class WorkflowResponse(WorkflowSummary):
    spec: WorkflowSpec


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
    trigger_event_override: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Custom trigger event envelope to use instead of sample data. "
            "Must contain at least 'trigger_key' and 'data' fields. When provided, "
            "runs a single execution with this event even if workflow has multiple triggers."
        )
    )
    trigger_id: Optional[str] = Field(
        None,
        description=(
            "ID of the specific trigger to run. Required when using "
            "trigger_event_override on a workflow with multiple triggers."
        )
    )


class RunProgress(BaseModel):
    completed: int = 0
    total: int = 0


class RunResponse(BaseModel):
    run_id: str
    status: str
    workflow_id: Optional[str] = None
    workflow_version_id: Optional[int] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    progress: Optional[RunProgress] = None
    current_node_id: Optional[str] = None
    last_error: Optional[str] = None


class RunWithTrigger(RunResponse):
    """Run response with trigger information."""
    trigger_title: str


class MultiRunResponse(BaseModel):
    """Response when multiple runs are created (one per trigger)."""
    runs: List[RunWithTrigger]


class RunResultResponse(BaseModel):
    run_id: str
    status: str
    workflow_id: Optional[str] = None
    workflow_version_id: Optional[int] = None
    output: Optional[Dict[str, Any]] = None
    state: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None


class RunHistoryResponse(BaseModel):
    run_id: str
    history: List[Dict[str, Any]] = Field(default_factory=list)


class WorkflowRunSummary(BaseModel):
    run_id: str
    status: str
    workflow_version_id: Optional[int] = None
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


class WorkflowImportRequest(BaseModel):
    import_data: Dict[str, Any]  # Full export JSON
    name: Optional[str] = None  # Override workflow name
    import_triggers: bool = True  # Whether to import triggers


class WorkflowExportResponse(BaseModel):
    version: str
    workflow: Dict[str, Any]
    triggers: List[Dict[str, Any]]
    metadata: Dict[str, Any]


# -----------------------------
# HITL (Human-In-The-Loop) Models
# -----------------------------
class HITLResumeRequest(BaseModel):
    """Request payload to resume a workflow from an HITL interrupt."""
    responses: Dict[str, Any] = Field(
        ...,
        description="User responses keyed by input field ID"
    )


class HITLInterruptDisplayItem(BaseModel):
    """Display item with evaluated value."""
    label: str
    value: Any


class HITLInterruptInputField(BaseModel):
    """Input field definition for HITL collection."""
    id: str
    question: str
    input_type: str
    options: Optional[List[Dict[str, Any]]] = None
    required: bool = True
    placeholder: Optional[str] = None
    default_value: Optional[Any] = None


class HITLInterruptResponse(BaseModel):
    """Response containing HITL interrupt data for a run."""
    run_id: str
    status: str
    node_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    display: List[HITLInterruptDisplayItem] = Field(default_factory=list)
    inputs: List[HITLInterruptInputField] = Field(default_factory=list)
    timeout_seconds: Optional[int] = None
    expires_at: Optional[str] = None
    is_expired: bool = False


# ============================================================================
# Workflow File Models
# ============================================================================


class WorkflowFileItem(BaseModel):
    """File metadata for a workflow run file."""
    file_id: str
    filename: str
    mime_type: str
    size_bytes: int
    size_human: str
    source_node_id: Optional[str] = None
    source_tool: Optional[str] = None
    created_at: datetime


class WorkflowFileListResponse(BaseModel):
    """Response containing list of files for a workflow run."""
    run_id: str
    files: List[WorkflowFileItem]
    total_count: int
    total_size_bytes: int


class WorkflowFileResponse(BaseModel):
    """Response containing single file metadata."""
    file: WorkflowFileItem


class WorkflowFileDownloadResponse(BaseModel):
    """Response containing presigned download URL."""
    file_id: str
    filename: str
    download_url: str
    expires_in_seconds: int


class WorkflowFileDeleteResponse(BaseModel):
    """Response confirming file deletion."""
    file_id: str
    deleted: bool


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
    "StartListeningResponse",
    "PendingEventItem",
    "PendingEventsResponse",
    "ModelDescriptor",
    "ModelRegistryResponse",
    "SchemaResponse",
    "SchemaMetadataGenerateRequest",
    "SchemaMetadataGenerateResponse",
    "WorkflowWarning",
    "ValidateRequest",
    "ValidateResponse",
    "CompileOptions",
    "CompileRequest",
    "CompileResponse",
    "CompileArtifacts",
    "WorkflowCreateRequest",
    "WorkflowUpdateRequest",
    "WorkflowDraftPatchRequest",
    "WorkflowPublishRequest",
    "WorkflowVersionRestoreRequest",
    "WorkflowResponse",
    "WorkflowSummary",
    "WorkflowListResponse",
    "WorkflowVersionSummary",
    "WorkflowVersionListResponse",
    "WorkflowVersionListItem",
    "RunFromSpecRequest",
    "RunFromWorkflowRequest",
    "RunResponse",
    "RunWithTrigger",
    "MultiRunResponse",
    "RunResultResponse",
    "RunHistoryResponse",
    "WorkflowRunSummary",
    "WorkflowRunListResponse",
    "ExpressionSuggestRequest",
    "ExpressionSuggestResponse",
    "ExpressionTypecheckRequest",
    "ExpressionTypecheckResponse",
    "ExpressionSuggestion",
    "WorkflowImportRequest",
    "WorkflowExportResponse",
    "HITLResumeRequest",
    "HITLInterruptDisplayItem",
    "HITLInterruptInputField",
    "HITLInterruptResponse",
    # Workflow Files
    "WorkflowFileItem",
    "WorkflowFileListResponse",
    "WorkflowFileResponse",
    "WorkflowFileDownloadResponse",
    "WorkflowFileDeleteResponse",
]
