"""Database models for workflow templates."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict
from tortoise import fields, models


class TemplateCategory(str, Enum):
    """Categories for workflow templates."""

    MARKETING = "marketing"
    CUSTOMER_SUPPORT = "customer_support"
    SALES = "sales"
    PRODUCTIVITY = "productivity"
    ENGINEERING = "engineering"
    OPERATIONS = "operations"
    HR = "hr"
    OTHER = "other"


class TemplateSource(str, Enum):
    """Source type for workflow templates."""

    SYSTEM = "system"  # Built-in templates
    ORGANIZATION = "org"  # Future: org-shared templates
    COMMUNITY = "community"  # Future: user-submitted templates


class WorkflowTemplate(models.Model):
    """Database model for workflow templates."""

    id = fields.IntField(primary_key=True)
    slug = fields.CharField(max_length=100, unique=True, db_index=True)
    name = fields.CharField(max_length=255)
    description = fields.TextField()
    category = fields.CharEnumField(TemplateCategory, max_length=30)
    tags = fields.JSONField(default=list)  # ["gmail", "slack"]
    source = fields.CharEnumField(TemplateSource, max_length=20, default=TemplateSource.SYSTEM)
    created_by = fields.ForeignKeyField("models.User", related_name="templates", null=True, on_delete=fields.SET_NULL)
    organization = fields.ForeignKeyField("models.Organization", related_name="templates", null=True, on_delete=fields.SET_NULL)
    visibility = fields.CharField(max_length=10, default="private")  # "private" | "public"
    spec = fields.JSONField()  # WorkflowSpec with ${config.xxx} placeholders
    required_integrations = fields.JSONField(default=list)
    # Format: [{"provider": "google", "integration_type": "gmail", "reason": "..."}]
    icon = fields.CharField(max_length=50, null=True)
    preview_image_url = fields.CharField(max_length=500, null=True)
    is_published = fields.BooleanField(default=False)
    is_featured = fields.BooleanField(default=False)
    usage_count = fields.IntField(default=0)
    source_workflow = fields.ForeignKeyField(
        "models.Workflow", related_name="templates", null=True, on_delete=fields.SET_NULL
    )
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "workflow_templates"
        ordering = ("-created_at",)
        indexes = (
            ("category", "is_published"),
            ("is_featured", "is_published"),
        )

    def __str__(self) -> str:
        return f"WorkflowTemplate<{self.slug}>"


def make_template_public_id(template_id: int) -> str:
    """Create a public-facing template ID."""
    return f"tpl_{template_id}"


def parse_template_public_id(public_id: str) -> int:
    """Parse a public-facing template ID back to internal ID."""
    if not public_id.startswith("tpl_"):
        raise ValueError(f"Invalid template ID format: {public_id}")
    try:
        return int(public_id[4:])
    except ValueError as exc:
        raise ValueError(f"Invalid template ID format: {public_id}") from exc


class WorkflowTemplatePublic(BaseModel):
    """Pydantic model for WorkflowTemplate API responses."""

    model_config = ConfigDict(from_attributes=True)

    template_id: str
    slug: str
    name: str
    description: str
    category: str
    tags: list[str]
    source: str
    icon: str | None
    preview_image_url: str | None
    is_featured: bool
    usage_count: int
    visibility: str

    @classmethod
    def from_orm(cls, obj: WorkflowTemplate) -> "WorkflowTemplatePublic":
        """Create a public response from a database model."""
        return cls(
            template_id=make_template_public_id(obj.id),
            slug=obj.slug,
            name=obj.name,
            description=obj.description,
            category=obj.category.value,
            tags=obj.tags or [],
            source=obj.source.value,
            icon=obj.icon,
            preview_image_url=obj.preview_image_url,
            is_featured=obj.is_featured,
            usage_count=obj.usage_count,
            visibility=obj.visibility,
        )
