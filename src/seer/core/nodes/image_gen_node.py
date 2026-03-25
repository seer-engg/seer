"""
ImageGenNode - Generate images via OpenRouter image generation models.

Supports models like sourceful/riverflow-v2-fast and google/gemini-2.5-flash-image.
Calls the OpenRouter chat completions API with image generation parameters.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import httpx

from seer.config import config
from seer.core.errors import ExecutionError
from seer.core.nodes.base import BaseNodeType, NodeExecutionContext, TypeRegistrationContext, get_trace_key
from seer.logger import get_logger
from seer.core.nodes.registry import register_node_type
from seer.core.schema.models import ImageGenNode
from seer.runtime_credit_limits import check_runtime_credit_limit

if TYPE_CHECKING:
    from seer.core.expr.evaluator import EvaluationContext
    from seer.core.expr.typecheck import TypeEnvironment
    from seer.core.files.models import WorkflowFileRef
    from seer.core.runtime.nodes import RuntimeServices
    from seer.core.schema.models import NodeBase

logger = get_logger(__name__)


class ImageGenNodeType(BaseNodeType):
    """Implementation of the image generation node type."""

    @property
    def type_literal(self) -> str:
        return "image_gen"

    @property
    def model_class(self) -> type["NodeBase"]:
        return ImageGenNode

    @staticmethod
    def _extract_image_from_images_array(images: list) -> str:
        """Extract image URL from message.images array format."""
        for img in images:
            if not isinstance(img, dict):
                continue
            if img.get("type") == "image_url":
                url = img.get("image_url", {}).get("url", "")
                if url:
                    return url
            elif "url" in img:
                url = img.get("url", "")
                if url:
                    return url
        return ""

    @staticmethod
    def _extract_image_from_string_content(content: str) -> str:
        """Extract image URL from string content (may be markdown or data URL)."""
        import re  # pylint: disable=import-outside-toplevel  # Reason: Only needed for response parsing

        # Check if content is a data URL
        if content.startswith("data:"):
            return content

        # Check if content contains a URL (may be wrapped in markdown)
        url_match = re.search(r'https?://[^\s\)\"\']+', content)
        if url_match:
            return url_match.group(0)

        return ""

    @staticmethod
    def _extract_image_from_content_blocks(content: list) -> str:
        """Extract image URL from multimodal content blocks."""
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "image_url":
                return block.get("image_url", {}).get("url", "")
            if block_type == "image":
                return block.get("url", "") or block.get("image_url", "")
        return ""

    @staticmethod
    def _extract_image_url_from_response(data: Dict[str, Any]) -> str:
        """
        Extract image URL from OpenRouter API response.

        Handles multiple response formats:
        - message.images array (primary format for image generation)
        - message.content as string URL or markdown
        - message.content as multimodal blocks array
        """
        choices = data.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})

        # PRIMARY: Check message.images array (OpenRouter image generation format)
        images = message.get("images", [])
        if images and isinstance(images, list):
            url = ImageGenNodeType._extract_image_from_images_array(images)
            if url:
                return url

        # FALLBACK: Check message.content
        content = message.get("content", "")

        if isinstance(content, str):
            return ImageGenNodeType._extract_image_from_string_content(content)

        if isinstance(content, list):
            return ImageGenNodeType._extract_image_from_content_blocks(content)

        return ""

    async def _check_credit_limit(self, context: Any) -> None:
        """Check credit limit before image generation call."""
        await check_runtime_credit_limit(context, logger)

    def _track_usage_async(self, usage_metadata: Dict[str, Any], context: Any) -> None:
        """Track image gen usage asynchronously (fire and forget)."""
        if not context or not context.user:
            logger.warning("Cannot track image gen usage: no user context")
            return

        # pylint: disable=import-outside-toplevel  # Reason: Late import for optional feature
        from seer.observability.cost_tracking import CostTracker
        from seer.observability.exceptions import RunCostCapExceeded

        async def do_track():
            try:
                await CostTracker.track_and_enforce_cap(
                    usage_metadata=usage_metadata,
                    context=context,
                    operation="workflow_execution",
                )
            except RunCostCapExceeded:
                raise
            except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Log error without crashing workflow
                logger.error("Failed to track image gen usage: %s", str(e), exc_info=True)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(do_track())
            else:
                loop.run_until_complete(do_track())
        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Log scheduling error without crashing workflow
            logger.error("Failed to schedule image gen usage tracking: %s", e)

    async def _download_image(self, image_url: str) -> Tuple[bytes, str]:
        """
        Download image from URL and detect MIME type.

        Args:
            image_url: The URL to download the image from.

        Returns:
            Tuple of (image_bytes, mime_type).

        Raises:
            ExecutionError: If download fails.
        """
        # Skip download for data URLs (base64 encoded)
        if image_url.startswith("data:"):
            # Parse data URL: data:image/png;base64,<data>
            import base64  # pylint: disable=import-outside-toplevel  # Reason: Only needed for data URL parsing

            try:
                header, b64_data = image_url.split(",", 1)
                mime_type = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
                image_bytes = base64.b64decode(b64_data)
                return image_bytes, mime_type
            except Exception as exc:
                raise ExecutionError(f"Failed to parse data URL: {exc}") from exc

        # Download from HTTP URL
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(image_url, follow_redirects=True)
                if response.status_code != 200:
                    raise ExecutionError(f"Failed to download image (status {response.status_code}): {image_url}")

                image_bytes = response.content

                # Detect MIME type from Content-Type header or URL extension
                content_type = response.headers.get("content-type", "")
                if content_type and "/" in content_type:
                    # Strip charset or other parameters
                    mime_type = content_type.split(";")[0].strip()
                else:
                    # Fallback to URL extension detection
                    mime_type = self._detect_mime_from_url(image_url)

                return image_bytes, mime_type
        except httpx.HTTPError as exc:
            raise ExecutionError(f"Failed to download image from {image_url}: {exc}") from exc

    @staticmethod
    def _detect_mime_from_url(url: str) -> str:
        """Detect MIME type from URL extension."""
        ext_to_mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".svg": "image/svg+xml",
            ".bmp": "image/bmp",
        }
        lower_url = url.lower().split("?")[0]  # Remove query params
        for ext, mime in ext_to_mime.items():
            if lower_url.endswith(ext):
                return mime
        return "image/png"  # Default to PNG

    async def _store_image(
        self,
        image_data: bytes,
        mime_type: str,
        node_id: str,
        ctx: NodeExecutionContext,
    ) -> Optional[Dict[str, Any]]:
        """
        Store image to WorkflowFileSystem.

        Args:
            image_data: The raw image bytes.
            mime_type: The MIME type of the image.
            node_id: The node ID for filename generation.
            ctx: The node execution context.

        Returns:
            WorkflowFileRef dict or None if storage is unavailable.
        """
        runtime_ctx = ctx.runtime_context
        if not runtime_ctx or not runtime_ctx.workflow_run_id:
            logger.debug("No workflow run context available for image storage")
            return None

        if not runtime_ctx.has_file_system:
            logger.debug("Workflow file system not configured")
            return None

        # Generate filename with extension based on MIME type
        ext = mime_type.split("/")[-1] if "/" in mime_type else "png"
        # Handle special cases
        if ext == "jpeg":
            ext = "jpg"
        elif ext == "svg+xml":
            ext = "svg"

        filename = f"image_gen_{node_id}_{uuid.uuid4().hex[:8]}.{ext}"

        try:
            file_ref: WorkflowFileRef = await runtime_ctx.file_system.store_file_with_record(
                user=runtime_ctx.user,
                run_id=runtime_ctx.workflow_run_id,
                filename=filename,
                data=image_data,
                mime_type=mime_type,
                source_tool="image_gen_node",
                organization_id=runtime_ctx.organization_id,
            )
            logger.info("Stored generated image in workflow file system: %s", file_ref.file_id)
            return file_ref.to_dict()
        except OSError as e:
            logger.warning("Failed to store image in workflow file system: %s", e)
            return None

    @staticmethod
    def _validate_inputs(node: ImageGenNode) -> Tuple[str, str, str]:
        """Validate and extract model, prompt_template, and size from node inputs."""
        model = node.inputs.get("model")
        if not isinstance(model, str):
            raise ExecutionError(f"ImageGenNode {node.id}: 'model' must be a string in inputs")

        prompt_template = node.inputs.get("prompt")
        if not isinstance(prompt_template, str):
            raise ExecutionError(f"ImageGenNode {node.id}: 'prompt' must be a string in inputs")

        size = node.inputs.get("size", "1024x1024")
        if not isinstance(size, str):
            size = "1024x1024"

        return model, prompt_template, size

    @staticmethod
    def _build_trace_entry(
        node_id: str,
        inputs: Dict[str, Any],
        usage_metadata: Dict[str, Any],
        model: str,
        *,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
    ) -> Dict[str, Any]:
        """Build a trace entry for success or error cases."""
        entry: Dict[str, Any] = {
            "node_id": node_id,
            "node_type": "image_gen",
            "inputs": inputs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if error:
            entry["error"] = {"type": error.__class__.__name__, "message": str(error)}
            entry["status"] = "failed"
        else:
            entry["output"] = result
            entry["output_key"] = node_id
            entry["status"] = "succeeded"
            entry["usage"] = {
                "model": usage_metadata.get("model", model),
                "input_tokens": usage_metadata.get("input_tokens", 0),
                "output_tokens": usage_metadata.get("output_tokens", 0),
                "total_tokens": usage_metadata.get("input_tokens", 0) + usage_metadata.get("output_tokens", 0),
            }

        return entry

    async def _download_and_store_image_safe(
        self,
        image_url: str,
        node_id: str,
        ctx: NodeExecutionContext,
    ) -> Optional[Dict[str, Any]]:
        """Download and store image, returning None on any failure (graceful degradation)."""
        if not image_url:
            return None
        try:
            image_bytes, mime_type = await self._download_image(image_url)
            return await self._store_image(image_bytes, mime_type, node_id, ctx)
        except ExecutionError as e:
            logger.warning("Failed to download/store generated image: %s", e)
        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Graceful degradation - continue with URL
            logger.warning("Unexpected error storing generated image: %s", e)
        return None

    async def _execute_api_call(
        self,
        node_id: str,
        ctx: NodeExecutionContext,
        inputs: Dict[str, Any],
        *,
        model: str,
        prompt: str,
        size: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute OpenRouter API call with error handling and trace generation."""
        api_key = config.openrouter_api_key
        assert api_key is not None  # Checked by caller in execute_async
        try:
            return await self._call_openrouter(api_key=api_key, model=model, prompt=prompt, size=size)
        except ExecutionError:
            raise
        except Exception as exc:
            trace_key = get_trace_key(node_id, ctx.state, ctx.loop_body_map or {}, ctx.nested_loop_parents or {})
            trace_entry = self._build_trace_entry(node_id, inputs, {}, model, error=exc)
            error_trace = {trace_key: trace_entry}
            ctx.state.update(error_trace)  # type: ignore[arg-type]
            raise ExecutionError(f"ImageGenNode '{node_id}' failed: {exc}", trace_data=error_trace) from exc

    @staticmethod
    def _build_result(
        image_url: str,
        prompt: str,
        model: str,
        size: str,
        file_ref_dict: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the result dictionary for the image generation output."""
        result: Dict[str, Any] = {"image_url": image_url, "prompt": prompt, "model": model, "size": size}
        if file_ref_dict:
            result["file"] = file_ref_dict
        return result

    def _build_eval_context(self, ctx: NodeExecutionContext) -> "EvaluationContext":
        """Build evaluation context for template rendering."""
        # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
        from seer.core.expr.evaluator import EvaluationContext
        from seer.core.runtime.state import INTERNAL_STATE_PREFIX

        visible_state = {k: v for k, v in ctx.state.items() if not k.startswith(INTERNAL_STATE_PREFIX)}
        return EvaluationContext(state=visible_state, locals=ctx.locals_ctx or {}, config=ctx.config, trigger=ctx.trigger, vars=ctx.vars)

    async def execute_async(
        self,
        node: ImageGenNode,  # type: ignore[override]
        ctx: NodeExecutionContext,
        services: "RuntimeServices",  # pylint: disable=unused-argument  # Reason: Required by BaseNodeType interface
    ) -> Dict[str, Any]:
        """Execute image generation node via OpenRouter API."""
        # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
        from seer.core.expr.evaluator import render_template

        await self._check_credit_limit(ctx.runtime_context)

        # Validate inputs and render prompt
        model, prompt_template, size = self._validate_inputs(node)
        prompt = render_template(self._build_eval_context(ctx), prompt_template)
        inputs = {"prompt_template": prompt_template, "rendered_prompt": prompt, "model": model, "size": size}

        # Call OpenRouter API
        if not config.openrouter_api_key:
            raise ExecutionError(f"ImageGenNode {node.id}: OpenRouter API key not configured")

        image_url, usage_metadata = await self._execute_api_call(
            node.id, ctx, inputs, model=model, prompt=prompt, size=size
        )

        if usage_metadata:
            self._track_usage_async(usage_metadata, ctx.runtime_context)

        # Build result and output with trace
        result = self._build_result(image_url, prompt, model, size, await self._download_and_store_image_safe(image_url, node.id, ctx))
        trace_key = get_trace_key(node.id, ctx.state, ctx.loop_body_map or {}, ctx.nested_loop_parents or {})
        output: Dict[str, Any] = {node.id: result, trace_key: self._build_trace_entry(node.id, inputs, usage_metadata, model, result=result)}

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("ImageGen node '%s' output keys: %s", node.id, list(output.keys()))

        return output

    @staticmethod
    def _build_openrouter_payload(model: str, prompt: str, size: str) -> Dict[str, Any]:
        """Build the request payload for OpenRouter API."""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Add size parameter via provider options
        if size:
            width, _, height = size.partition("x")
            if width.isdigit() and height.isdigit():
                payload["provider"] = {"sort": "throughput"}
                payload["metadata"] = {
                    "image_size": size,
                    "width": int(width),
                    "height": int(height),
                }

        return payload

    @staticmethod
    def _extract_usage_metadata(data: Dict[str, Any], fallback_model: str) -> Dict[str, Any]:
        """Extract usage metadata from OpenRouter response."""
        usage = data.get("usage", {})
        return {
            "model": data.get("model", fallback_model),
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }

    @staticmethod
    async def _call_openrouter(
        api_key: str,
        model: str,
        prompt: str,
        size: str,
    ) -> tuple[str, Dict[str, Any]]:
        """
        Call OpenRouter API for image generation.

        Returns tuple of (image_url, usage_metadata).
        """
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://seer.app",
        }

        payload = ImageGenNodeType._build_openrouter_payload(model, prompt, size)

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload, headers=headers)

        if response.status_code != 200:
            raise ExecutionError(
                f"OpenRouter API error (status {response.status_code}): {response.text}"
            )

        data = response.json()
        image_url = ImageGenNodeType._extract_image_url_from_response(data)
        usage_metadata = ImageGenNodeType._extract_usage_metadata(data, model)

        return image_url, usage_metadata

    def register_type_sync(
        self,
        node: ImageGenNode,  # type: ignore[override]
        env: "TypeEnvironment",
        ctx: TypeRegistrationContext,  # pylint: disable=unused-argument  # Reason: Required by BaseNodeType interface
    ) -> None:
        """Register image gen node's output schema (image_url string + optional file ref)."""
        schema = {
            "type": "object",
            "properties": {
                "image_url": {"type": "string"},
                "file": {"type": "object"},  # WorkflowFileRef when storage is available
                "prompt": {"type": "string"},
                "model": {"type": "string"},
                "size": {"type": "string"},
            },
            "required": ["image_url", "prompt", "model", "size"],  # file is optional
        }
        if node.id:
            env.register(node.id, schema)


# Auto-register on module import
register_node_type(ImageGenNodeType())
