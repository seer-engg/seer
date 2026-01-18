"""
Supabase database webhook management service.
Creates and manages Postgres triggers in Supabase projects that POST
webhook events to Seer when database rows change (INSERT, UPDATE, DELETE).
"""

from __future__ import annotations

from typing import Any, Dict, List

import httpx

from seer.services.integrations.constants import SUPABASE_RESOURCE_PROVIDER
from seer.config import config
from seer.database import IntegrationResource, TriggerSubscription
from seer.logger import get_logger
from seer.tools.oauth_manager import get_oauth_token

logger = get_logger(__name__)


class SupabaseWebhookError(Exception):
    """Raised when Supabase webhook operations fail."""


async def create_database_webhook(
    subscription: TriggerSubscription,
    webhook_url: str,
    secret: str,
) -> Dict[str, Any]:
    """
    Creates a Postgres trigger in the Supabase project that sends webhook events.
    Args:
        subscription: The trigger subscription containing configuration
        webhook_url: The full URL to POST webhook events to (includes subscription_id and secret)
    Returns:
        Dict with metadata about the created trigger
    Raises:
        SupabaseWebhookError: If webhook creation fails
    """
    config_dict = subscription.provider_config or {}
    integration_resource_id = config_dict.get("integration_resource_id")
    table_name = config_dict.get("table")
    schema_name = config_dict.get("schema", "public")
    events = config_dict.get("events", ["INSERT", "UPDATE", "DELETE"])

    if not integration_resource_id:
        raise SupabaseWebhookError("integration_resource_id is required in provider_config")
    if not table_name:
        raise SupabaseWebhookError("table name is required in provider_config")
    if not events:
        raise SupabaseWebhookError("events list is required in provider_config")

    # Fetch the Supabase project resource
    try:
        resource = await IntegrationResource.get(
            id=integration_resource_id,
            user_id=subscription.user_id,
            provider=SUPABASE_RESOURCE_PROVIDER,
        )
    except Exception as exc:
        logger.error(
            "Failed to fetch Supabase resource",
            extra={"resource_id": integration_resource_id, "error": str(exc)},
        )
        raise SupabaseWebhookError(f"Supabase project resource not found: {exc}") from exc

    project_ref = resource.resource_key
    if not project_ref:
        raise SupabaseWebhookError("Supabase project_ref not found in resource")

    # Get OAuth access token for Management API
    oauth_connection = await resource.oauth_connection
    if not oauth_connection:
        raise SupabaseWebhookError("No OAuth connection found for Supabase project")

    try:
        _, access_token = await get_oauth_token(
            await resource.user,
            connection_id=str(oauth_connection.id),
            provider="supabase_mgmt",
        )
    except Exception as exc:
        logger.error("Failed to get Supabase OAuth token", extra={"error": str(exc)})
        raise SupabaseWebhookError(f"Failed to authenticate with Supabase: {exc}") from exc

    # Generate unique trigger and function names
    trigger_name = f"seer_webhook_trigger_{subscription.id}"
    function_name = f"notify_seer_webhook_{subscription.id}"

    # Build the SQL to create the trigger
    sql = _build_trigger_sql(
        function_name=function_name,
        trigger_name=trigger_name,
        table_name=table_name,
        schema_name=schema_name,
        events=events,
        webhook_url=webhook_url,
        secret=secret,
    )

    # Execute the SQL via Management API
    try:
        await _execute_sql(access_token, project_ref, sql)
    except Exception as exc:
        logger.error(
            "Failed to create Supabase webhook trigger",
            extra={
                "project_ref": project_ref,
                "table": table_name,
                "error": str(exc),
            },
        )
        raise SupabaseWebhookError(f"Failed to create database trigger: {exc}") from exc

    logger.info(
        "Created Supabase webhook trigger",
        extra={
            "subscription_id": subscription.id,
            "project_ref": project_ref,
            "table": f"{schema_name}.{table_name}",
            "events": events,
        },
    )

    return {
        "trigger_name": trigger_name,
        "function_name": function_name,
        "table": f"{schema_name}.{table_name}",
        "events": events,
    }


async def delete_database_webhook(subscription: TriggerSubscription) -> None:
    """
    Removes the Postgres trigger from the Supabase project.
    Args:
        subscription: The trigger subscription to clean up
    Raises:
        SupabaseWebhookError: If webhook deletion fails (non-fatal, logged only)
    """
    config_dict = subscription.provider_config or {}
    integration_resource_id = config_dict.get("integration_resource_id")
    table_name = config_dict.get("table")
    schema_name = config_dict.get("schema", "public")

    if not integration_resource_id or not table_name:
        logger.warning(
            "Skipping webhook deletion - missing configuration",
            extra={"subscription_id": subscription.id},
        )
        return

    try:
        resource = await IntegrationResource.get(
            id=integration_resource_id,
            user_id=subscription.user_id,
            provider=SUPABASE_RESOURCE_PROVIDER,
        )
        project_ref = resource.resource_key
        oauth_connection = await resource.oauth_connection

        if not oauth_connection:
            logger.warning("No OAuth connection for webhook deletion")
            return

        _, access_token = await get_oauth_token(
            await resource.user,
            connection_id=str(oauth_connection.id),
            provider="supabase_mgmt",
        )

        trigger_name = f"seer_webhook_trigger_{subscription.id}"
        function_name = f"notify_seer_webhook_{subscription.id}"

        sql = _build_cleanup_sql(
            function_name=function_name,
            trigger_name=trigger_name,
            table_name=table_name,
            schema_name=schema_name,
        )

        await _execute_sql(access_token, project_ref, sql)

        logger.info(
            "Deleted Supabase webhook trigger",
            extra={
                "subscription_id": subscription.id,
                "project_ref": project_ref,
            },
        )

    except Exception as exc:
        # Non-fatal: log and continue since subscription is being deleted anyway
        logger.warning(
            "Failed to delete Supabase webhook trigger (non-fatal)",
            extra={"subscription_id": subscription.id, "error": str(exc)},
        )


def _build_trigger_sql(
    function_name: str,
    trigger_name: str,
    table_name: str,
    schema_name: str,
    events: List[str],
    webhook_url: str,
    secret: str,
) -> str:
    """
    Generates SQL to create a Postgres trigger that sends webhook events.
    """
    # Validate events
    valid_events = {"INSERT", "UPDATE", "DELETE"}
    filtered_events = [e.upper() for e in events if e.upper() in valid_events]
    if not filtered_events:
        raise SupabaseWebhookError(f"No valid events provided. Must be one of: {valid_events}")

    event_clause = " OR ".join(filtered_events)

    # Build the function that sends the webhook
    sql = f"""
-- Create function to send webhook notification
create extension if not exists pg_net with schema extensions;
select net.worker_restart();
CREATE OR REPLACE FUNCTION {schema_name}.{function_name}()
RETURNS TRIGGER AS $$
DECLARE
  request_id bigint;
  payload jsonb;
BEGIN
  -- Build the payload
  payload := jsonb_build_object(
    'type', TG_OP,
    'table', TG_TABLE_NAME,
    'schema', TG_TABLE_SCHEMA,
    'record', CASE WHEN TG_OP = 'DELETE' THEN NULL ELSE row_to_json(NEW) END,
    'old_record', CASE WHEN TG_OP IN ('UPDATE', 'DELETE') THEN row_to_json(OLD) ELSE NULL END
  );
  -- Send webhook using pg_net extension
  SELECT INTO request_id net.http_post(
    url := '{webhook_url}',
    headers := '{{"Content-Type": "application/json", "X-Seer-Webhook-Secret": "{secret}"}}'::jsonb,
    body := payload
  );
  -- Return appropriate value based on operation
  IF TG_OP = 'DELETE' THEN
    RETURN OLD;
  ELSE
    RETURN NEW;
  END IF;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
-- Create trigger on the table
DROP TRIGGER IF EXISTS {trigger_name} ON {schema_name}.{table_name};
CREATE TRIGGER {trigger_name}
AFTER {event_clause} ON {schema_name}.{table_name}
FOR EACH ROW EXECUTE FUNCTION {schema_name}.{function_name}();
"""
    return sql.strip()


def _build_cleanup_sql(
    function_name: str,
    trigger_name: str,
    table_name: str,
    schema_name: str,
) -> str:
    """
    Generates SQL to remove the trigger and function.
    """
    sql = f"""
-- Drop trigger
DROP TRIGGER IF EXISTS {trigger_name} ON {schema_name}.{table_name};
-- Drop function
DROP FUNCTION IF EXISTS {schema_name}.{function_name}();
"""
    return sql.strip()


async def _execute_sql(access_token: str, project_ref: str, sql: str) -> Any:
    """
    Executes SQL on a Supabase project via the Management API.
    Uses the /v1/projects/{ref}/database/query endpoint.
    """
    api_base = config.supabase_management_api_base or "https://api.supabase.com"
    url = f"{api_base.rstrip('/')}/v1/projects/{project_ref}/database/query"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    payload = {"query": sql}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json() if response.text else None
    except httpx.HTTPStatusError as exc:
        error_detail = exc.response.text[:500] if exc.response.text else "Unknown error"
        logger.error(
            "Supabase SQL execution failed",
            extra={
                "status_code": exc.response.status_code,
                "error": error_detail,
                "project_ref": project_ref,
            },
        )
        raise SupabaseWebhookError(f"SQL execution failed: {error_detail}") from exc
    except Exception as exc:
        logger.exception("Unexpected error executing Supabase SQL")
        raise SupabaseWebhookError(f"Unexpected error: {exc}") from exc


__all__ = [
    "SupabaseWebhookError",
    "create_database_webhook",
    "delete_database_webhook",
]
