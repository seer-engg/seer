from types import SimpleNamespace

from api.integrations.services import _format_supabase_secret_name
from shared.tools.supabase import _resolve_rest_url


def test_format_supabase_secret_name_aliases() -> None:
    assert _format_supabase_secret_name("service_role") == "supabase_service_role_key"
    assert _format_supabase_secret_name("anon") == "supabase_anon_key"
    assert _format_supabase_secret_name("custom") == "supabase_custom_key"


def test_resolve_rest_url_falls_back_to_project_ref() -> None:
    resource = SimpleNamespace(resource_metadata={}, resource_key="demo-ref")
    assert _resolve_rest_url(resource) == "https://demo-ref.supabase.co/rest/v1"

    resource_with_metadata = SimpleNamespace(
        resource_metadata={"rest_url": "https://example.supabase.co/rest/v1"},
        resource_key="ignored",
    )
    assert _resolve_rest_url(resource_with_metadata) == "https://example.supabase.co/rest/v1"


