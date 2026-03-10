from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from seer.api.workflows.trigger_models import (
    PendingEventItem,
    PendingEventsResponse,
    StartListeningResponse,
    SubscriptionEventCountResponse,
    TriggerAccountInfo,
    TriggerAccountsResponse,
    TriggerCatalogResponse,
    TriggerDescriptor,
    TriggerEventGenerateRequest,
    TriggerEventGenerateResponse,
    TriggerSubscriptionCreateRequest,
    TriggerSubscriptionListItem,
    TriggerSubscriptionListItemsResponse,
    TriggerSubscriptionListResponse,
    TriggerSubscriptionResponse,
    TriggerSubscriptionTestRequest,
    TriggerSubscriptionTestResponse,
    TriggerSubscriptionToggleRequest,
    TriggerSubscriptionUpdateRequest,
)
from seer.core.schema.models import WorkflowSpec


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


# ============================================================================
# Global Variables
# ============================================================================


class GlobalVariableCreateRequest(BaseModel):
    key: str = Field(..., max_length=255)
    value: str
    is_secret: bool = False
    description: Optional[str] = None


class GlobalVariableUpdateRequest(BaseModel):
    value: Optional[str] = None
    is_secret: Optional[bool] = None
    description: Optional[str] = None


class GlobalVariableItem(BaseModel):
    id: int
    key: str
    value: str
    is_secret: bool
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class GlobalVariableListResponse(BaseModel):
    items: List[GlobalVariableItem]


__all__ = [
    "ProblemDetails", "ProblemError", "NodeFieldDescriptor", "NodeTypeDescriptor",
    "NodeTypeResponse", "ToolDescriptor", "ToolRegistryResponse", "TriggerDescriptor",
    "TriggerCatalogResponse", "TriggerAccountInfo", "TriggerAccountsResponse",
    "TriggerSubscriptionCreateRequest", "TriggerSubscriptionUpdateRequest",
    "TriggerSubscriptionResponse", "TriggerSubscriptionListResponse",
    "TriggerSubscriptionListItem", "TriggerSubscriptionListItemsResponse",
    "TriggerSubscriptionToggleRequest", "TriggerSubscriptionTestRequest",
    "TriggerSubscriptionTestResponse", "StartListeningResponse", "PendingEventItem",
    "PendingEventsResponse", "SubscriptionEventCountResponse",
    "TriggerEventGenerateRequest", "TriggerEventGenerateResponse", "ModelDescriptor",
    "ModelRegistryResponse", "SchemaResponse", "SchemaMetadataGenerateRequest",
    "SchemaMetadataGenerateResponse", "WorkflowWarning", "ValidateRequest",
    "ValidateResponse", "CompileOptions", "CompileRequest", "CompileResponse",
    "CompileArtifacts", "WorkflowCreateRequest", "WorkflowUpdateRequest",
    "WorkflowDraftPatchRequest", "WorkflowPublishRequest", "WorkflowVersionRestoreRequest",
    "WorkflowResponse", "WorkflowSummary", "WorkflowListResponse", "WorkflowVersionSummary",
    "WorkflowVersionListResponse", "WorkflowVersionListItem", "RunFromSpecRequest",
    "RunFromWorkflowRequest", "RunResponse", "RunResultResponse", "RunHistoryResponse",
    "WorkflowRunSummary", "WorkflowRunListResponse", "ExpressionSuggestRequest",
    "ExpressionSuggestResponse", "ExpressionTypecheckRequest", "ExpressionTypecheckResponse",
    "ExpressionSuggestion", "WorkflowImportRequest", "WorkflowExportResponse",
    "HITLResumeRequest", "HITLInterruptDisplayItem", "HITLInterruptInputField",
    "HITLInterruptResponse", "WorkflowFileItem", "WorkflowFileListResponse",
    "WorkflowFileResponse", "WorkflowFileDownloadResponse", "WorkflowFileDeleteResponse",
    "GlobalVariableCreateRequest", "GlobalVariableUpdateRequest",
    "GlobalVariableItem", "GlobalVariableListResponse",
]
