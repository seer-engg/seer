"""
Unit tests for LinkedIn tool wrappers.

Tests the LinkedIn create_post tool with image upload support.
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from seer.tools.linkedin.linkedin import LinkedInTool, LINKEDIN_TOOL_SCOPES


# =============================================================================
# LinkedInTool Basic Tests
# =============================================================================


@pytest.mark.unit
class TestLinkedInToolBasic:
    """Tests for LinkedInTool basic functionality."""

    def test_tool_initialization(self):
        """Test LinkedInTool initializes correctly."""
        tool = LinkedInTool(
            tool_name="create_post",
            description="Create a post",
            parameters_schema={"type": "object", "properties": {}},
            integration_type="linkedin"
        )

        assert tool.name == "linkedin_create_post"
        assert tool.tool_name == "create_post"
        assert tool.description == "Create a post"
        assert tool.integration_type == "linkedin"
        assert tool.provider == "linkedin"
        assert tool.required_scopes == LINKEDIN_TOOL_SCOPES["create_post"]

    def test_tool_scope_mapping(self):
        """Test tool-to-scope mapping is correct."""
        assert LINKEDIN_TOOL_SCOPES["get_profile"] == ["openid", "profile", "email"]
        assert LINKEDIN_TOOL_SCOPES["create_post"] == ["w_member_social"]

    def test_tool_parameters_schema(self):
        """Test tool returns correct parameters schema."""
        tool = LinkedInTool(
            tool_name="create_post",
            description="Create a post",
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
        )

        schema = tool.get_parameters_schema()
        assert schema["type"] == "object"
        assert "text" in schema["properties"]


# =============================================================================
# Create Post Tests (Text Only)
# =============================================================================


@pytest.mark.unit
class TestLinkedInCreatePostTextOnly:
    """Tests for LinkedIn create_post without images (backward compatibility)."""

    @pytest.fixture
    def create_post_tool(self):
        """Create a create_post tool instance."""
        return LinkedInTool(
            tool_name="create_post",
            description="Create a post on LinkedIn",
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "visibility": {"type": "string", "enum": ["PUBLIC", "CONNECTIONS"]},
                    "images": {"type": "array", "items": {}}
                },
                "required": ["text"]
            }
        )

    @pytest.mark.asyncio
    async def test_create_post_text_only(self, create_post_tool):
        """Test creating a text-only post (backward compatible)."""
        mock_userinfo_response = MagicMock()
        mock_userinfo_response.status_code = 200
        mock_userinfo_response.json.return_value = {"sub": "test-user-123"}

        mock_post_response = MagicMock()
        mock_post_response.status_code = 201
        mock_post_response.json.return_value = {"id": "urn:li:share:123456"}

        with patch("seer.tools.linkedin.linkedin.httpx.AsyncClient") as mock_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_userinfo_response)
            mock_http.post = AsyncMock(return_value=mock_post_response)
            mock_client.return_value.__aenter__.return_value = mock_http

            result = await create_post_tool.execute(
                access_token="test-token",
                arguments={"text": "Hello LinkedIn!", "visibility": "PUBLIC"}
            )

            assert result["id"] == "urn:li:share:123456"
            # Verify legacy ugcPosts endpoint was used for text-only
            call_args = mock_http.post.call_args
            assert "ugcPosts" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_create_post_missing_text_raises(self, create_post_tool):
        """Test that missing text raises HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            await create_post_tool.execute(
                access_token="test-token",
                arguments={"visibility": "PUBLIC"}
            )

        assert exc_info.value.status_code == 400
        assert "text is required" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_create_post_missing_token_raises(self, create_post_tool):
        """Test that missing access token raises HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            await create_post_tool.execute(
                access_token=None,
                arguments={"text": "Hello"}
            )

        assert exc_info.value.status_code == 401


# =============================================================================
# Create Post with Images Tests
# =============================================================================


@pytest.mark.unit
class TestLinkedInCreatePostWithImages:
    """Tests for LinkedIn create_post with image attachments."""

    @pytest.fixture
    def create_post_tool(self):
        """Create a create_post tool instance."""
        return LinkedInTool(
            tool_name="create_post",
            description="Create a post on LinkedIn",
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "visibility": {"type": "string"},
                    "images": {"type": "array", "items": {}}
                },
                "required": ["text"]
            }
        )

    @pytest.fixture
    def mock_workflow_context(self):
        """Create a mock workflow runtime context."""
        context = MagicMock()
        context.has_file_system = True
        return context

    @pytest.mark.asyncio
    async def test_create_post_with_single_image(self, create_post_tool, mock_workflow_context):
        """Test creating a post with a single image."""
        # Mock userinfo response
        mock_userinfo_response = MagicMock()
        mock_userinfo_response.status_code = 200
        mock_userinfo_response.json.return_value = {"sub": "test-user-123"}

        # Mock image upload init response
        mock_init_response = MagicMock()
        mock_init_response.status_code = 200
        mock_init_response.json.return_value = {
            "value": {
                "uploadUrl": "https://linkedin.com/upload/12345",
                "image": "urn:li:image:C4E10AQFoyyAjHPMQuQ"
            }
        }

        # Mock image upload response
        mock_upload_response = MagicMock()
        mock_upload_response.status_code = 201

        # Mock post creation response
        mock_post_response = MagicMock()
        mock_post_response.status_code = 201
        mock_post_response.json.return_value = {"id": "urn:li:ugcPost:123456"}

        # Mock file resolution via WorkflowFileSystem
        image_bytes = b"fake-image-data"
        with patch("seer.tools.linkedin.linkedin.WorkflowFileSystem") as mock_fs_class:
            mock_fs = MagicMock()
            mock_fs.get_file_content = AsyncMock(return_value=image_bytes)
            mock_fs_class.instance.return_value = mock_fs

            with patch("seer.tools.linkedin.linkedin.httpx.AsyncClient") as mock_client:
                mock_http = AsyncMock()
                mock_http.get = AsyncMock(return_value=mock_userinfo_response)
                mock_http.post = AsyncMock(side_effect=[mock_init_response, mock_post_response])
                mock_http.put = AsyncMock(return_value=mock_upload_response)
                mock_client.return_value.__aenter__.return_value = mock_http

                # Provide complete file ref structure for parse_file_ref
                file_ref = {
                    "_type": "workflow_file_ref",
                    "file_id": "file-123",
                    "storage_path": "s3://bucket/run/file-123/test.jpg",
                    "filename": "test.jpg",
                    "mime_type": "image/jpeg",
                    "size_bytes": 1024,
                    "workflow_run_id": "run-1",
                    "created_at": "2024-01-01T00:00:00",
                }
                result = await create_post_tool.execute(
                    access_token="test-token",
                    arguments={
                        "text": "Check out this image!",
                        "images": [file_ref]
                    },
                    context=mock_workflow_context
                )

                assert result["id"] == "urn:li:ugcPost:123456"

                # Verify file content was retrieved
                mock_fs.get_file_content.assert_called_once()

                # Verify Posts API was used (not ugcPosts)
                post_calls = mock_http.post.call_args_list
                assert any("rest/posts" in str(call) for call in post_calls)

    @pytest.mark.asyncio
    async def test_create_post_with_multiple_images(self, create_post_tool, mock_workflow_context):
        """Test creating a post with multiple images uses MultiImage format."""
        mock_userinfo_response = MagicMock()
        mock_userinfo_response.status_code = 200
        mock_userinfo_response.json.return_value = {"sub": "test-user-123"}

        mock_init_response = MagicMock()
        mock_init_response.status_code = 200

        mock_upload_response = MagicMock()
        mock_upload_response.status_code = 201

        mock_post_response = MagicMock()
        mock_post_response.status_code = 201
        mock_post_response.json.return_value = {"id": "urn:li:ugcPost:789"}

        # Return different image URNs for each upload
        init_responses = [
            {"value": {"uploadUrl": "https://linkedin.com/upload/1", "image": "urn:li:image:IMG1"}},
            {"value": {"uploadUrl": "https://linkedin.com/upload/2", "image": "urn:li:image:IMG2"}},
            {"value": {"uploadUrl": "https://linkedin.com/upload/3", "image": "urn:li:image:IMG3"}},
        ]
        mock_init_response.json.side_effect = init_responses

        with patch("seer.tools.linkedin.linkedin.WorkflowFileSystem") as mock_fs_class:
            mock_fs = MagicMock()
            mock_fs.get_file_content = AsyncMock(return_value=b"image-data")
            mock_fs_class.instance.return_value = mock_fs

            with patch("seer.tools.linkedin.linkedin.httpx.AsyncClient") as mock_client:
                mock_http = AsyncMock()
                mock_http.get = AsyncMock(return_value=mock_userinfo_response)
                # 3 init uploads + 1 post creation
                mock_http.post = AsyncMock(side_effect=[
                    mock_init_response, mock_init_response, mock_init_response,
                    mock_post_response
                ])
                mock_http.put = AsyncMock(return_value=mock_upload_response)
                mock_client.return_value.__aenter__.return_value = mock_http

                # Provide complete file ref structures for parse_file_ref
                def make_file_ref(file_id: str) -> dict:
                    return {
                        "_type": "workflow_file_ref",
                        "file_id": file_id,
                        "storage_path": f"s3://bucket/run/{file_id}/image.png",
                        "filename": "image.png",
                        "mime_type": "image/png",
                        "size_bytes": 1024,
                        "workflow_run_id": "run-1",
                        "created_at": "2024-01-01T00:00:00",
                    }

                result = await create_post_tool.execute(
                    access_token="test-token",
                    arguments={
                        "text": "Multiple images!",
                        "images": [
                            make_file_ref("file-1"),
                            make_file_ref("file-2"),
                            make_file_ref("file-3"),
                        ]
                    },
                    context=mock_workflow_context
                )

                assert result["id"] == "urn:li:ugcPost:789"
                # Verify 3 file resolutions
                assert mock_fs.get_file_content.call_count == 3

    @pytest.mark.asyncio
    async def test_create_post_max_images_validation(self, create_post_tool):
        """Test that more than 9 images raises HTTPException."""
        # Create 10 fake image inputs
        images = [{"_type": "workflow_file_ref", "file_id": f"file-{i}"} for i in range(10)]

        with pytest.raises(HTTPException) as exc_info:
            await create_post_tool.execute(
                access_token="test-token",
                arguments={"text": "Too many images", "images": images}
            )

        assert exc_info.value.status_code == 400
        assert "Maximum 9 images" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_create_post_file_resolution_error(self, create_post_tool, mock_workflow_context):
        """Test that unsupported image input types are handled properly."""
        mock_userinfo_response = MagicMock()
        mock_userinfo_response.status_code = 200
        mock_userinfo_response.json.return_value = {"sub": "test-user-123"}

        with patch("seer.tools.linkedin.linkedin.httpx.AsyncClient") as mock_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_userinfo_response)
            mock_client.return_value.__aenter__.return_value = mock_http

            # Test with unsupported input type (list) - should raise ValueError -> HTTPException 400
            with pytest.raises(HTTPException) as exc_info:
                await create_post_tool.execute(
                    access_token="test-token",
                    arguments={
                        "text": "Post with bad image",
                        "images": [["nested", "list", "not", "supported"]]
                    },
                    context=mock_workflow_context
                )

            assert exc_info.value.status_code == 400
            assert "Invalid image input" in str(exc_info.value.detail)


# =============================================================================
# Image Upload Helper Tests
# =============================================================================


@pytest.mark.unit
class TestLinkedInImageUpload:
    """Tests for the _upload_image helper method."""

    @pytest.fixture
    def create_post_tool(self):
        """Create a create_post tool instance."""
        return LinkedInTool(
            tool_name="create_post",
            description="Create a post on LinkedIn"
        )

    @pytest.mark.asyncio
    async def test_upload_image_success(self, create_post_tool):
        """Test successful image upload."""
        mock_init_response = MagicMock()
        mock_init_response.status_code = 200
        mock_init_response.json.return_value = {
            "value": {
                "uploadUrl": "https://linkedin.com/upload/abc123",
                "image": "urn:li:image:TEST123"
            }
        }

        mock_upload_response = MagicMock()
        mock_upload_response.status_code = 201

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_init_response)
        mock_http.put = AsyncMock(return_value=mock_upload_response)

        image_urn = await create_post_tool._upload_image(
            http_client=mock_http,
            access_token="test-token",
            person_urn="person-123",
            image_bytes=b"fake-image-bytes"
        )

        assert image_urn == "urn:li:image:TEST123"

        # Verify init upload was called correctly
        init_call = mock_http.post.call_args
        assert "initializeUpload" in init_call[0][0]
        assert init_call[1]["json"]["initializeUploadRequest"]["owner"] == "urn:li:person:person-123"

        # Verify binary upload was called
        put_call = mock_http.put.call_args
        assert put_call[1]["content"] == b"fake-image-bytes"
        assert put_call[1]["headers"]["Content-Type"] == "application/octet-stream"

    @pytest.mark.asyncio
    async def test_upload_image_init_401_raises(self, create_post_tool):
        """Test that 401 during init raises HTTPException."""
        mock_init_response = MagicMock()
        mock_init_response.status_code = 401

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_init_response)

        with pytest.raises(HTTPException) as exc_info:
            await create_post_tool._upload_image(
                http_client=mock_http,
                access_token="bad-token",
                person_urn="person-123",
                image_bytes=b"data"
            )

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_upload_image_init_403_raises(self, create_post_tool):
        """Test that 403 during init raises HTTPException with scope hint."""
        mock_init_response = MagicMock()
        mock_init_response.status_code = 403

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_init_response)

        with pytest.raises(HTTPException) as exc_info:
            await create_post_tool._upload_image(
                http_client=mock_http,
                access_token="token",
                person_urn="person-123",
                image_bytes=b"data"
            )

        assert exc_info.value.status_code == 403
        assert "w_member_social" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_upload_image_missing_upload_url_raises(self, create_post_tool):
        """Test that missing uploadUrl in response raises HTTPException."""
        mock_init_response = MagicMock()
        mock_init_response.status_code = 200
        mock_init_response.json.return_value = {"value": {}}  # Missing uploadUrl

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_init_response)

        with pytest.raises(HTTPException) as exc_info:
            await create_post_tool._upload_image(
                http_client=mock_http,
                access_token="token",
                person_urn="person-123",
                image_bytes=b"data"
            )

        assert exc_info.value.status_code == 500
        assert "upload URL or image URN" in str(exc_info.value.detail)


# =============================================================================
# Tool Registration Tests
# =============================================================================


@pytest.mark.unit
class TestLinkedInToolRegistration:
    """Tests for LinkedIn tool registration."""

    def test_register_linkedin_tools(self):
        """Test that LinkedIn tools are registered correctly."""
        from seer.tools.linkedin.linkedin import register_linkedin_tools
        from seer.tools.base import get_tool, clear_registry

        clear_registry()
        register_linkedin_tools()

        # Verify get_profile is registered
        get_profile = get_tool("linkedin_get_profile")
        assert get_profile is not None
        assert get_profile.tool_name == "get_profile"

        # Verify create_post is registered
        create_post = get_tool("linkedin_create_post")
        assert create_post is not None
        assert create_post.tool_name == "create_post"

        # Verify create_post has images in schema
        schema = create_post.get_parameters_schema()
        assert "images" in schema["properties"]
        assert schema["properties"]["images"]["type"] == "array"
        assert schema["properties"]["images"]["maxItems"] == 9

        clear_registry()
