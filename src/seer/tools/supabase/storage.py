from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any, Dict, Optional

import httpx
from fastapi import HTTPException

from seer.core.files.resolver import FileResolutionError, resolve_file_input
from seer.core.files.schemas import FILE_INPUT_SCHEMA, FILE_OUTPUT_SCHEMA
from seer.logger import get_logger
from seer.tools.supabase.common import (
    SupabaseProjectTool,
    _service_request_json_or_ok,
    _require_project_and_key,
    _resolve_storage_url,
    _service_headers,
)

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext

logger = get_logger("shared.tools.supabase.storage")


# -----------------------------
# Storage (/storage/v1)
# -----------------------------

class SupabaseStorageListBucketsTool(SupabaseProjectTool):
    name = "supabase_storage_list_buckets"
    description = "List Storage buckets in a Supabase project."

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "integration_resource_id": {"type": "integer"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "search": {"type": "string"},
            },
            "required": ["integration_resource_id"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {"type": "array", "items": {"type": "object", "additionalProperties": True}}

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any], credentials: Optional[Any] = None) -> Any:
        resource, service_key = _require_project_and_key(credentials)
        storage_url = _resolve_storage_url(resource)
        if not storage_url:
            raise HTTPException(status_code=400, detail="Supabase project metadata is missing storage URL. Please re-bind.")

        params = {
            "limit": arguments.get("limit", 100),
            "offset": arguments.get("offset", 0),
        }
        if arguments.get("search"):
            params["search"] = arguments["search"]

        url = f"{storage_url.rstrip('/')}/bucket"
        return await _service_request_json_or_ok(
            "GET", service_key, url, params=params, logger_obj=logger, error_detail="List buckets failed"
        )


class SupabaseStorageCreateBucketTool(SupabaseProjectTool):
    name = "supabase_storage_create_bucket"
    description = "Create a Storage bucket."

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "integration_resource_id": {"type": "integer"},
                "name": {"type": "string", "description": "Bucket name (also used as id if id omitted)."},
                "public": {"type": "boolean", "default": False},
                "file_size_limit": {"type": ["integer", "string", "null"], "description": "Optional, e.g. 1000000 or '100MB'."},
                "allowed_mime_types": {"type": ["array", "null"], "items": {"type": "string"}},
            },
            "required": ["integration_resource_id", "name"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {"type": "object", "additionalProperties": True}

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any], credentials: Optional[Any] = None) -> Any:
        resource, service_key = _require_project_and_key(credentials)
        storage_url = _resolve_storage_url(resource)
        if not storage_url:
            raise HTTPException(status_code=400, detail="Supabase project metadata is missing storage URL. Please re-bind.")

        payload = {
            "name": arguments["name"],
            "public": bool(arguments.get("public", False)),
        }
        if "file_size_limit" in arguments:
            payload["file_size_limit"] = arguments.get("file_size_limit")
        if "allowed_mime_types" in arguments:
            payload["allowed_mime_types"] = arguments.get("allowed_mime_types")

        url = f"{storage_url.rstrip('/')}/bucket"
        return await _service_request_json_or_ok(
            "POST", service_key, url, json_body=payload, logger_obj=logger, error_detail="Create bucket failed",
            extra_headers={"Content-Type": "application/json"},
        )


class SupabaseStorageUploadObjectTool(SupabaseProjectTool):
    name = "supabase_storage_upload_object"
    description = (
        "Upload/overwrite an object into a Supabase storage bucket. "
        "Accepts a file reference from another tool (e.g., ${download.file}) "
        "or a static file from user storage."
    )

    def get_parameters_schema(self) -> Dict[str, Any]:
        file_schema = FILE_INPUT_SCHEMA.copy()
        file_schema["description"] = "File to upload (from parent node or user storage)"
        return {
            "type": "object",
            "properties": {
                "integration_resource_id": {"type": "integer"},
                "bucket": {"type": "string"},
                "path": {"type": "string", "description": "Object path inside the bucket, e.g. 'folder/a.png'"},
                "file": file_schema,
                "content_type": {"type": "string", "description": "Content type (auto-detected from file if not provided)"},
                "cache_control": {"type": "string", "description": "Optional cache control header, e.g. '3600'."},
            },
            "required": ["integration_resource_id", "bucket", "path", "file"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {"type": "object", "additionalProperties": True}

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        credentials: Optional[Any] = None,
        *,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Any:
        resource, service_key = _require_project_and_key(credentials)
        storage_url = _resolve_storage_url(resource)
        if not storage_url:
            raise HTTPException(status_code=400, detail="Supabase project metadata is missing storage URL. Please re-bind.")

        bucket = arguments["bucket"]
        path = arguments["path"].lstrip("/")
        cache_control = arguments.get("cache_control")

        file_input = arguments.get("file")
        if not file_input:
            raise HTTPException(status_code=400, detail="file is required")

        # Resolve file content using the unified resolver
        try:
            data, resolved_mime_type, _ = await resolve_file_input(file_input, context)
        except FileResolutionError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        content_type = arguments.get("content_type") or resolved_mime_type or "application/octet-stream"

        # Simple PUT to /object/<bucket>/<path>
        url = f"{storage_url.rstrip('/')}/object/{bucket}/{path}"

        logger.info("Uploading to Supabase storage: bucket=%s path=%s size=%d", bucket, path, len(data))

        extra = {"Content-Type": content_type}
        if cache_control:
            extra["cache-control"] = cache_control

        return await _service_request_json_or_ok(
            "PUT",
            service_key,
            url,
            extra_headers=extra,
            content=data,
            logger_obj=logger,
            error_detail="Upload failed",
            ok_fallback={"ok": True},
            timeout=60.0,
        )


class SupabaseStorageDownloadObjectTool(SupabaseProjectTool):
    name = "supabase_storage_download_object"
    description = (
        "Download an object from Supabase Storage. Returns a file reference "
        "for efficient handling in workflows."
    )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "integration_resource_id": {"type": "integer"},
                "bucket": {"type": "string"},
                "path": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["public", "authenticated"],
                    "default": "authenticated",
                    "description": "Use documented serving routes; private buckets typically use 'authenticated'.",
                },
            },
            "required": ["integration_resource_id", "bucket", "path"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file": FILE_OUTPUT_SCHEMA,
                "size_bytes": {"type": "integer"},
                "content_base64": {
                    "type": "string",
                    "description": "File content as base64 (only included when not in workflow context)"
                },
            },
        }

    async def _store_in_workflow_fs(
        self, context: Optional["WorkflowRuntimeContext"], content: bytes, *, content_type: str, path: str, bucket: str
    ) -> Optional[Dict[str, Any]]:
        """Store downloaded content in workflow file system if available."""
        if not (context and context.workflow_run_id and context.has_file_system):
            return None

        filename = path.rsplit("/", 1)[-1] if "/" in path else path
        try:
            file_ref = await context.file_system.store_file_with_record(
                user=context.user,
                run_id=context.workflow_run_id,
                filename=filename,
                data=content,
                mime_type=content_type,
                source_tool="supabase_storage_download_object",
            )
            logger.info("Stored Supabase file in workflow file system: bucket=%s path=%s file_id=%s", bucket, path, file_ref.file_id)
            return {"file": file_ref.to_dict(), "size_bytes": len(content)}
        except OSError as e:
            logger.warning("Failed to store file in workflow file system, falling back to base64: %s", e)
            return None

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        credentials: Optional[Any] = None,
        *,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Any:
        resource, service_key = _require_project_and_key(credentials)
        storage_url = _resolve_storage_url(resource)
        if not storage_url:
            raise HTTPException(status_code=400, detail="Supabase project metadata is missing storage URL. Please re-bind.")

        bucket, path = arguments["bucket"], arguments["path"].lstrip("/")
        mode = arguments.get("mode") or "authenticated"
        url = f"{storage_url.rstrip('/')}/object/{mode}/{bucket}/{path}"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.get(url, headers=_service_headers(service_key))
                if resp.status_code >= 400:
                    raise HTTPException(status_code=resp.status_code, detail=f"Download failed: {resp.text[:500]}")

                content = resp.content
                content_type = resp.headers.get("content-type", "application/octet-stream")

                # Try to store in workflow file system first
                result = await self._store_in_workflow_fs(context, content, content_type=content_type, path=path, bucket=bucket)
                if result:
                    return result

                # Fallback: return base64 encoded content
                return {"content_base64": base64.b64encode(content).decode("utf-8"), "size_bytes": len(content)}
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Supabase storage download error", extra={"bucket": bucket, "path": path})
            raise HTTPException(status_code=500, detail=f"Supabase request failed: {str(exc)}") from exc


class SupabaseStorageCreateSignedObjectUrlTool(SupabaseProjectTool):
    name = "supabase_storage_create_signed_object_url"
    description = "Create a signed URL for a Storage object (server-side)."

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "integration_resource_id": {"type": "integer"},
                "bucket": {"type": "string"},
                "path": {"type": "string"},
                "expires_in": {"type": "integer", "minimum": 1, "maximum": 604800, "default": 3600},
            },
            "required": ["integration_resource_id", "bucket", "path"],
        }
    def get_output_schema(self) -> Dict[str, Any]:
        return {"type": "object", "additionalProperties": True}

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any], credentials: Optional[Any] = None) -> Any:
        resource, service_key = _require_project_and_key(credentials)
        storage_url = _resolve_storage_url(resource)
        if not storage_url:
            raise HTTPException(status_code=400, detail="Supabase project metadata is missing storage URL. Please re-bind.")

        bucket = arguments["bucket"]
        path = arguments["path"].lstrip("/")
        expires_in = int(arguments.get("expires_in", 3600))

        # URL pattern referenced in Supabase Storage docs: /storage/v1/object/sign/<bucket>/<path>
        url = f"{storage_url.rstrip('/')}/object/sign/{bucket}/{path}"
        payload = {"expiresIn": expires_in}

        return await _service_request_json_or_ok(
            "POST",
            service_key,
            url,
            json_body=payload,
            logger_obj=logger,
            error_detail="Create signed URL failed",
            extra_headers={"Content-Type": "application/json"},
        )
