"""
Database models for behavioral learning and personalization.

These models capture user interactions with tool search and workflow creation
to enable:
- Contrastive learning for improved tool ranking
- Sequence modeling for workflow recommendations
- User preference profiling for personalization
"""
from datetime import datetime, timezone
from tortoise import fields, models


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class ToolSearchEvent(models.Model):
    """
    Records every tool search interaction for learning.

    Captures:
    - What user searched for (query)
    - What tools were shown (tools_shown)
    - What tool they selected (tool_selected)
    - What tools they skipped (tools_skipped)

    This data enables contrastive learning:
    - Positive pairs: (query, tool_selected)
    - Negative pairs: (query, tools_skipped)
    """

    id = fields.IntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="tool_searches")
    query = fields.CharField(max_length=500, db_index=True)
    tools_shown = fields.JSONField()  # List[str] - tool names returned by search
    tool_selected = fields.CharField(max_length=100, null=True, db_index=True)
    tools_skipped = fields.JSONField(default=list)  # List[str] - tools shown but not selected
    integration_filter = fields.CharField(max_length=50, null=True)
    context = fields.JSONField(null=True)  # Additional context (workflow_id, session_id, etc.)
    timestamp = fields.DatetimeField(default=_now_utc, db_index=True)

    class Meta:
        table = "tool_search_events"
        ordering = ("-timestamp",)

    def __str__(self) -> str:
        selected = self.tool_selected or "none"
        return f"ToolSearchEvent<query='{self.query[:30]}...' selected={selected}>"


class WorkflowSequence(models.Model):
    """
    Records tool sequences from workflows for sequence modeling.

    Captures:
    - User intent (workflow description)
    - Tool sequence (ordered list of tools used)
    - Trigger type
    - Success/failure status
    - Execution metrics

    This data enables:
    - Next-tool prediction (given history, predict next)
    - Workflow template learning
    - Common pattern discovery
    """

    id = fields.IntField(primary_key=True)
    workflow = fields.ForeignKeyField("models.Workflow", related_name="learned_sequences", null=True)
    user = fields.ForeignKeyField("models.User", related_name="workflow_sequences")
    intent = fields.TextField()  # Workflow description or user's original query
    tool_sequence = fields.JSONField()  # List[str] - ordered tool names
    trigger_type = fields.CharField(max_length=100, null=True, db_index=True)
    success = fields.BooleanField(default=False, db_index=True)
    execution_time_ms = fields.IntField(null=True)
    error_message = fields.TextField(null=True)
    metadata = fields.JSONField(null=True)  # Additional metadata (node_count, complexity, etc.)
    created_at = fields.DatetimeField(default=_now_utc, db_index=True)

    # Link to workflow run for traceability
    workflow_run = fields.ForeignKeyField("models.WorkflowRun", related_name="learned_sequences", null=True)

    class Meta:
        table = "workflow_sequences"
        ordering = ("-created_at",)

    def __str__(self) -> str:
        status = "success" if self.success else "failed"
        return f"WorkflowSequence<{len(self.tool_sequence)} tools, {status}>"


class UserToolPreference(models.Model):
    """
    Aggregated user preferences for tools (computed periodically).

    This is a denormalized view for fast lookup, computed from:
    - ToolSearchEvent (what they search for)
    - WorkflowSequence (what they actually use)

    Used for:
    - Prompt personalization
    - Tool recommendation boosting
    - User behavior clustering
    """

    id = fields.IntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="tool_preferences", unique=True)

    # Top tools by usage frequency
    top_tools = fields.JSONField(default=list)  # List[Dict] - [{"tool": "slack_send", "count": 42}, ...]

    # Tools user has tried but failed with
    failed_tools = fields.JSONField(default=list)  # List[str]

    # Common workflow patterns (clusters)
    common_patterns = fields.JSONField(default=list)  # List[Dict] - [{"pattern": "notification", "tools": [...]}]

    # Time-of-day patterns
    usage_patterns = fields.JSONField(null=True)  # Dict - {"morning": ["email"], "evening": ["slack"]}

    # Metadata
    total_searches = fields.IntField(default=0)
    total_workflows = fields.IntField(default=0)
    success_rate = fields.FloatField(default=0.0)
    last_updated = fields.DatetimeField(default=_now_utc)
    created_at = fields.DatetimeField(default=_now_utc)

    class Meta:
        table = "user_tool_preferences"
        ordering = ("-last_updated",)

    def __str__(self) -> str:
        return f"UserToolPreference<user={self.user_id} searches={self.total_searches}>"
