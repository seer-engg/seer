"""Common configuration for knowledge base tools."""
from seer.tools.base import ResourcePickerConfig

# Resource picker configuration for kb_id parameter
KNOWLEDGE_BASE_PICKER: ResourcePickerConfig = {
    "resource_type": "knowledge_base",
    "display_field": "name",
    "value_field": "kb_id",
    "search_enabled": True,
    "endpoint": "/api/v1/knowledge-bases",
}
