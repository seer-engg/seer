"""Unit tests for YouTube tools."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
import httpx

from seer.tools.google.youtube import (
    YouTubeSearchTool,
    YouTubeGetVideoTool,
    YouTubeGetChannelTool,
    YouTubeListPlaylistsTool,
    YouTubeGetPlaylistItemsTool,
    YouTubeUploadVideoTool,
)


@pytest.fixture
def mock_access_token():
    return "test_youtube_access_token"


@pytest.fixture
def search_tool():
    return YouTubeSearchTool()


@pytest.fixture
def get_video_tool():
    return YouTubeGetVideoTool()


@pytest.fixture
def get_channel_tool():
    return YouTubeGetChannelTool()


@pytest.fixture
def list_playlists_tool():
    return YouTubeListPlaylistsTool()


@pytest.fixture
def get_playlist_items_tool():
    return YouTubeGetPlaylistItemsTool()


@pytest.fixture
def upload_video_tool():
    return YouTubeUploadVideoTool()


# ==============================================================================
# YouTubeSearchTool Tests
# ==============================================================================


@pytest.mark.unit
class TestYouTubeSearchTool:
    """Tests for YouTubeSearchTool."""

    def test_tool_metadata(self, search_tool):
        """Test tool has correct metadata."""
        assert search_tool.name == "youtube_search"
        assert search_tool.integration_type == "youtube"
        assert "youtube.readonly" in search_tool.required_scopes[0]

    @pytest.mark.asyncio
    async def test_search_videos_returns_results(self, search_tool, mock_access_token):
        """Test searching for videos returns search results."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {
            "kind": "youtube#searchListResponse",
            "items": [
                {
                    "kind": "youtube#searchResult",
                    "id": {"kind": "youtube#video", "videoId": "dQw4w9WgXcQ"},
                    "snippet": {
                        "title": "Rick Astley - Never Gonna Give You Up",
                        "channelTitle": "Rick Astley",
                        "description": "The official video...",
                    },
                }
            ],
            "pageInfo": {"totalResults": 1000, "resultsPerPage": 25},
        }

        with patch.object(
            search_tool, "_execute_request_with_retry", return_value=mock_response
        ):
            result = await search_tool.execute(
                mock_access_token,
                {"q": "rick astley"},
            )

        assert result["kind"] == "youtube#searchListResponse"
        assert len(result["items"]) == 1
        assert result["items"][0]["id"]["videoId"] == "dQw4w9WgXcQ"

    @pytest.mark.asyncio
    async def test_search_requires_query(self, search_tool, mock_access_token):
        """Test that search query is required."""
        with pytest.raises(ValueError, match="Search query.*is required"):
            await search_tool.execute(mock_access_token, {})

    @pytest.mark.asyncio
    async def test_search_with_type_filter(self, search_tool, mock_access_token):
        """Test searching with type filter (channel, playlist)."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {"items": []}

        captured_params = {}

        async def capture_request(*args, **kwargs):
            captured_params.update(kwargs.get("params", {}))
            return mock_response

        with patch.object(
            search_tool, "_execute_request_with_retry", side_effect=capture_request
        ):
            await search_tool.execute(
                mock_access_token,
                {"q": "python tutorial", "type": "channel"},
            )

        assert captured_params.get("type") == "channel"

    @pytest.mark.asyncio
    async def test_search_with_date_filters(self, search_tool, mock_access_token):
        """Test searching with date range filters."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {"items": []}

        captured_params = {}

        async def capture_request(*args, **kwargs):
            captured_params.update(kwargs.get("params", {}))
            return mock_response

        with patch.object(
            search_tool, "_execute_request_with_retry", side_effect=capture_request
        ):
            await search_tool.execute(
                mock_access_token,
                {
                    "q": "python",
                    "published_after": "2024-01-01T00:00:00Z",
                    "published_before": "2024-12-31T23:59:59Z",
                },
            )

        assert captured_params.get("publishedAfter") == "2024-01-01T00:00:00Z"
        assert captured_params.get("publishedBefore") == "2024-12-31T23:59:59Z"

    @pytest.mark.asyncio
    async def test_search_max_results_bounded(self, search_tool, mock_access_token):
        """Test max_results is bounded between 1 and 50."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {"items": []}

        captured_params = {}

        async def capture_request(*args, **kwargs):
            captured_params.update(kwargs.get("params", {}))
            return mock_response

        with patch.object(
            search_tool, "_execute_request_with_retry", side_effect=capture_request
        ):
            # Test upper bound
            await search_tool.execute(
                mock_access_token,
                {"q": "test", "max_results": 100},
            )
            assert captured_params.get("maxResults") == 50


# ==============================================================================
# YouTubeGetVideoTool Tests
# ==============================================================================


@pytest.mark.unit
class TestYouTubeGetVideoTool:
    """Tests for YouTubeGetVideoTool."""

    def test_tool_metadata(self, get_video_tool):
        """Test tool has correct metadata."""
        assert get_video_tool.name == "youtube_get_video"
        assert get_video_tool.integration_type == "youtube"
        assert "youtube.readonly" in get_video_tool.required_scopes[0]

    @pytest.mark.asyncio
    async def test_get_video_by_id(self, get_video_tool, mock_access_token):
        """Test getting video details by ID."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {
            "kind": "youtube#videoListResponse",
            "items": [
                {
                    "id": "dQw4w9WgXcQ",
                    "snippet": {
                        "title": "Rick Astley - Never Gonna Give You Up",
                        "description": "The official video...",
                        "channelTitle": "Rick Astley",
                    },
                    "contentDetails": {"duration": "PT3M33S"},
                    "statistics": {"viewCount": "1500000000"},
                }
            ],
        }

        with patch.object(
            get_video_tool, "_execute_request_with_retry", return_value=mock_response
        ):
            result = await get_video_tool.execute(
                mock_access_token,
                {"video_id": "dQw4w9WgXcQ"},
            )

        assert result["kind"] == "youtube#videoListResponse"
        assert result["items"][0]["id"] == "dQw4w9WgXcQ"
        assert result["items"][0]["statistics"]["viewCount"] == "1500000000"

    @pytest.mark.asyncio
    async def test_get_video_requires_video_id(self, get_video_tool, mock_access_token):
        """Test that video_id is required."""
        with pytest.raises(ValueError, match="video_id is required"):
            await get_video_tool.execute(mock_access_token, {})

    @pytest.mark.asyncio
    async def test_get_video_custom_parts(self, get_video_tool, mock_access_token):
        """Test getting video with custom parts parameter."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {"items": []}

        captured_params = {}

        async def capture_request(*args, **kwargs):
            captured_params.update(kwargs.get("params", {}))
            return mock_response

        with patch.object(
            get_video_tool, "_execute_request_with_retry", side_effect=capture_request
        ):
            await get_video_tool.execute(
                mock_access_token,
                {"video_id": "abc123", "part": "snippet,status,player"},
            )

        assert captured_params.get("part") == "snippet,status,player"


# ==============================================================================
# YouTubeGetChannelTool Tests
# ==============================================================================


@pytest.mark.unit
class TestYouTubeGetChannelTool:
    """Tests for YouTubeGetChannelTool."""

    def test_tool_metadata(self, get_channel_tool):
        """Test tool has correct metadata."""
        assert get_channel_tool.name == "youtube_get_channel"
        assert get_channel_tool.integration_type == "youtube"
        assert "youtube.readonly" in get_channel_tool.required_scopes[0]

    @pytest.mark.asyncio
    async def test_get_channel_by_id(self, get_channel_tool, mock_access_token):
        """Test getting channel info by ID."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {
            "kind": "youtube#channelListResponse",
            "items": [
                {
                    "id": "UCuAXFkgsw1L7xaCfnd5JJOw",
                    "snippet": {
                        "title": "Rick Astley",
                        "description": "Official YouTube channel...",
                    },
                    "statistics": {
                        "subscriberCount": "3500000",
                        "videoCount": "120",
                    },
                }
            ],
        }

        with patch.object(
            get_channel_tool, "_execute_request_with_retry", return_value=mock_response
        ):
            result = await get_channel_tool.execute(
                mock_access_token,
                {"channel_id": "UCuAXFkgsw1L7xaCfnd5JJOw"},
            )

        assert result["kind"] == "youtube#channelListResponse"
        assert result["items"][0]["snippet"]["title"] == "Rick Astley"

    @pytest.mark.asyncio
    async def test_get_channel_by_handle(self, get_channel_tool, mock_access_token):
        """Test getting channel info by handle."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {"items": [{"id": "channel123"}]}

        captured_params = {}

        async def capture_request(*args, **kwargs):
            captured_params.update(kwargs.get("params", {}))
            return mock_response

        with patch.object(
            get_channel_tool, "_execute_request_with_retry", side_effect=capture_request
        ):
            await get_channel_tool.execute(
                mock_access_token,
                {"for_handle": "@rickastley"},
            )

        assert captured_params.get("forHandle") == "@rickastley"

    @pytest.mark.asyncio
    async def test_get_channel_mine(self, get_channel_tool, mock_access_token):
        """Test getting authenticated user's channel."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {"items": []}

        captured_params = {}

        async def capture_request(*args, **kwargs):
            captured_params.update(kwargs.get("params", {}))
            return mock_response

        with patch.object(
            get_channel_tool, "_execute_request_with_retry", side_effect=capture_request
        ):
            await get_channel_tool.execute(
                mock_access_token,
                {"channel_id": "mine"},
            )

        assert captured_params.get("mine") == "true"
        assert "id" not in captured_params

    @pytest.mark.asyncio
    async def test_get_channel_requires_identifier(self, get_channel_tool, mock_access_token):
        """Test that at least one identifier is required."""
        with pytest.raises(ValueError, match="At least one of"):
            await get_channel_tool.execute(mock_access_token, {})


# ==============================================================================
# YouTubeListPlaylistsTool Tests
# ==============================================================================


@pytest.mark.unit
class TestYouTubeListPlaylistsTool:
    """Tests for YouTubeListPlaylistsTool."""

    def test_tool_metadata(self, list_playlists_tool):
        """Test tool has correct metadata."""
        assert list_playlists_tool.name == "youtube_list_playlists"
        assert list_playlists_tool.integration_type == "youtube"
        assert "youtube.readonly" in list_playlists_tool.required_scopes[0]

    @pytest.mark.asyncio
    async def test_list_playlists_by_channel(self, list_playlists_tool, mock_access_token):
        """Test listing playlists for a channel."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {
            "kind": "youtube#playlistListResponse",
            "items": [
                {
                    "id": "PL123",
                    "snippet": {"title": "My Playlist"},
                    "contentDetails": {"itemCount": 10},
                }
            ],
        }

        with patch.object(
            list_playlists_tool, "_execute_request_with_retry", return_value=mock_response
        ):
            result = await list_playlists_tool.execute(
                mock_access_token,
                {"channel_id": "UCxyz"},
            )

        assert result["kind"] == "youtube#playlistListResponse"
        assert len(result["items"]) == 1
        assert result["items"][0]["id"] == "PL123"

    @pytest.mark.asyncio
    async def test_list_playlists_mine(self, list_playlists_tool, mock_access_token):
        """Test listing authenticated user's playlists."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {"items": []}

        captured_params = {}

        async def capture_request(*args, **kwargs):
            captured_params.update(kwargs.get("params", {}))
            return mock_response

        with patch.object(
            list_playlists_tool, "_execute_request_with_retry", side_effect=capture_request
        ):
            await list_playlists_tool.execute(
                mock_access_token,
                {"mine": True},
            )

        assert captured_params.get("mine") == "true"

    @pytest.mark.asyncio
    async def test_list_playlists_requires_identifier(self, list_playlists_tool, mock_access_token):
        """Test that at least one filter is required."""
        with pytest.raises(ValueError, match="At least one of"):
            await list_playlists_tool.execute(mock_access_token, {})


# ==============================================================================
# YouTubeGetPlaylistItemsTool Tests
# ==============================================================================


@pytest.mark.unit
class TestYouTubeGetPlaylistItemsTool:
    """Tests for YouTubeGetPlaylistItemsTool."""

    def test_tool_metadata(self, get_playlist_items_tool):
        """Test tool has correct metadata."""
        assert get_playlist_items_tool.name == "youtube_get_playlist_items"
        assert get_playlist_items_tool.integration_type == "youtube"
        assert "youtube.readonly" in get_playlist_items_tool.required_scopes[0]

    @pytest.mark.asyncio
    async def test_get_playlist_items(self, get_playlist_items_tool, mock_access_token):
        """Test getting items from a playlist."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {
            "kind": "youtube#playlistItemListResponse",
            "items": [
                {
                    "id": "item1",
                    "snippet": {
                        "title": "Video 1",
                        "position": 0,
                        "resourceId": {"videoId": "vid1"},
                    },
                },
                {
                    "id": "item2",
                    "snippet": {
                        "title": "Video 2",
                        "position": 1,
                        "resourceId": {"videoId": "vid2"},
                    },
                },
            ],
            "pageInfo": {"totalResults": 2},
        }

        with patch.object(
            get_playlist_items_tool, "_execute_request_with_retry", return_value=mock_response
        ):
            result = await get_playlist_items_tool.execute(
                mock_access_token,
                {"playlist_id": "PL123"},
            )

        assert result["kind"] == "youtube#playlistItemListResponse"
        assert len(result["items"]) == 2
        assert result["items"][0]["snippet"]["resourceId"]["videoId"] == "vid1"

    @pytest.mark.asyncio
    async def test_get_playlist_items_requires_playlist_id(self, get_playlist_items_tool, mock_access_token):
        """Test that playlist_id is required."""
        with pytest.raises(ValueError, match="playlist_id is required"):
            await get_playlist_items_tool.execute(mock_access_token, {})

    @pytest.mark.asyncio
    async def test_get_playlist_items_with_video_filter(self, get_playlist_items_tool, mock_access_token):
        """Test filtering playlist items by video ID."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {"items": []}

        captured_params = {}

        async def capture_request(*args, **kwargs):
            captured_params.update(kwargs.get("params", {}))
            return mock_response

        with patch.object(
            get_playlist_items_tool, "_execute_request_with_retry", side_effect=capture_request
        ):
            await get_playlist_items_tool.execute(
                mock_access_token,
                {"playlist_id": "PL123", "video_id": "specificVideo"},
            )

        assert captured_params.get("videoId") == "specificVideo"


# ==============================================================================
# YouTubeUploadVideoTool Tests
# ==============================================================================


@pytest.mark.unit
class TestYouTubeUploadVideoTool:
    """Tests for YouTubeUploadVideoTool."""

    def test_tool_metadata(self, upload_video_tool):
        """Test tool has correct metadata."""
        assert upload_video_tool.name == "youtube_upload_video"
        assert upload_video_tool.integration_type == "youtube"
        assert "youtube.upload" in upload_video_tool.required_scopes[0]

    @pytest.mark.asyncio
    async def test_upload_requires_title(self, upload_video_tool, mock_access_token):
        """Test that title is required."""
        with pytest.raises(ValueError, match="title is required"):
            await upload_video_tool.execute(
                mock_access_token,
                {"video_base64": "dmlkZW9jb250ZW50"},  # "videocontent" in base64
            )

    @pytest.mark.asyncio
    async def test_upload_requires_video_content(self, upload_video_tool, mock_access_token):
        """Test that video_url or video_base64 is required."""
        with pytest.raises(ValueError, match="video_url or video_base64 is required"):
            await upload_video_tool.execute(
                mock_access_token,
                {"title": "Test Video"},
            )

    @pytest.mark.asyncio
    async def test_upload_with_base64(self, upload_video_tool, mock_access_token):
        """Test uploading video with base64 content."""
        # Mock the resumable upload initialization response
        mock_init_response = Mock(spec=httpx.Response)
        mock_init_response.status_code = 200
        mock_init_response.headers = {"Location": "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&upload_id=xyz"}

        # Mock the actual upload response
        mock_upload_response = Mock(spec=httpx.Response)
        mock_upload_response.status_code = 200
        mock_upload_response.json.return_value = {
            "id": "newVideoId",
            "snippet": {"title": "Test Video"},
            "status": {"privacyStatus": "private"},
        }

        with patch.object(
            upload_video_tool, "_init_resumable_upload", return_value="https://www.googleapis.com/upload/youtube/v3/videos?upload_id=xyz"
        ):
            with patch.object(
                upload_video_tool, "_upload_video_content", return_value={
                    "id": "newVideoId",
                    "snippet": {"title": "Test Video"},
                    "status": {"privacyStatus": "private"},
                }
            ):
                result = await upload_video_tool.execute(
                    mock_access_token,
                    {
                        "title": "Test Video",
                        "description": "A test video",
                        "video_base64": "dmlkZW9jb250ZW50",  # "videocontent" in base64
                    },
                )

        assert result["id"] == "newVideoId"
        assert result["snippet"]["title"] == "Test Video"

    @pytest.mark.asyncio
    async def test_upload_invalid_base64(self, upload_video_tool, mock_access_token):
        """Test that invalid base64 raises an error."""
        with pytest.raises(ValueError, match="Invalid base64"):
            await upload_video_tool.execute(
                mock_access_token,
                {"title": "Test", "video_base64": "not-valid-base64!!!"},
            )

    @pytest.mark.asyncio
    async def test_upload_with_url(self, upload_video_tool, mock_access_token):
        """Test uploading video from URL."""
        mock_download_response = AsyncMock()
        mock_download_response.status_code = 200
        mock_download_response.content = b"video binary content"

        with patch.object(
            upload_video_tool, "_download_video", return_value=b"video binary content"
        ):
            with patch.object(
                upload_video_tool, "_init_resumable_upload", return_value="https://upload-url"
            ):
                with patch.object(
                    upload_video_tool, "_upload_video_content", return_value={
                        "id": "urlVideoId",
                        "snippet": {"title": "URL Video"},
                    }
                ):
                    result = await upload_video_tool.execute(
                        mock_access_token,
                        {
                            "title": "URL Video",
                            "video_url": "https://example.com/video.mp4",
                        },
                    )

        assert result["id"] == "urlVideoId"

    @pytest.mark.asyncio
    async def test_upload_privacy_settings(self, upload_video_tool, mock_access_token):
        """Test upload with different privacy settings."""
        captured_metadata = {}

        async def capture_init(*args, **kwargs):
            # The metadata is passed directly to _init_resumable_upload
            return "https://upload-url"

        with patch.object(
            upload_video_tool, "_download_video", return_value=b"content"
        ):
            with patch.object(
                upload_video_tool, "_init_resumable_upload", side_effect=capture_init
            ) as mock_init:
                with patch.object(
                    upload_video_tool, "_upload_video_content", return_value={"id": "vid"}
                ):
                    await upload_video_tool.execute(
                        mock_access_token,
                        {
                            "title": "Public Video",
                            "privacy_status": "public",
                            "video_url": "https://example.com/video.mp4",
                        },
                    )

                    # Check the metadata passed to _init_resumable_upload
                    call_args = mock_init.call_args
                    metadata = call_args[0][1]  # Second positional argument
                    assert metadata["status"]["privacyStatus"] == "public"


# ==============================================================================
# Error Handling Tests
# ==============================================================================


@pytest.mark.unit
class TestYouTubeToolErrorHandling:
    """Tests for error handling across YouTube tools."""

    @pytest.mark.asyncio
    async def test_handles_api_error_404(self, get_video_tool, mock_access_token):
        """Test handling of 404 not found error."""
        from fastapi import HTTPException

        # Mock _execute_request_with_retry to raise HTTPException (simulating 404)
        with patch.object(
            get_video_tool, "_execute_request_with_retry",
            side_effect=HTTPException(status_code=404, detail="Resource not found")
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_video_tool.execute(
                    mock_access_token,
                    {"video_id": "nonexistent"},
                )
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_handles_api_error_403(self, search_tool, mock_access_token):
        """Test handling of 403 permission denied error."""
        from fastapi import HTTPException

        with patch.object(
            search_tool, "_execute_request_with_retry",
            side_effect=HTTPException(status_code=403, detail="Permission denied")
        ):
            with pytest.raises(HTTPException) as exc_info:
                await search_tool.execute(
                    mock_access_token,
                    {"q": "test"},
                )
            assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_handles_rate_limit_429(self, search_tool, mock_access_token):
        """Test handling of 429 rate limit error."""
        from fastapi import HTTPException

        with patch.object(
            search_tool, "_execute_request_with_retry",
            side_effect=HTTPException(status_code=429, detail="Rate limit exceeded")
        ):
            with pytest.raises(HTTPException) as exc_info:
                await search_tool.execute(
                    mock_access_token,
                    {"q": "test"},
                )
            assert exc_info.value.status_code == 429


# ==============================================================================
# Integration Type and Scope Tests
# ==============================================================================


@pytest.mark.unit
class TestYouTubeToolScopes:
    """Tests for YouTube tool scopes and integration types."""

    def test_all_read_tools_use_readonly_scope(self):
        """Test all read tools use youtube.readonly scope."""
        read_tools = [
            YouTubeSearchTool(),
            YouTubeGetVideoTool(),
            YouTubeGetChannelTool(),
            YouTubeListPlaylistsTool(),
            YouTubeGetPlaylistItemsTool(),
        ]
        for tool in read_tools:
            assert "youtube.readonly" in tool.required_scopes[0], f"{tool.name} should use readonly scope"

    def test_upload_tool_uses_upload_scope(self):
        """Test upload tool uses youtube.upload scope."""
        tool = YouTubeUploadVideoTool()
        assert "youtube.upload" in tool.required_scopes[0]

    def test_all_tools_have_youtube_integration_type(self):
        """Test all tools have 'youtube' as integration type."""
        all_tools = [
            YouTubeSearchTool(),
            YouTubeGetVideoTool(),
            YouTubeGetChannelTool(),
            YouTubeListPlaylistsTool(),
            YouTubeGetPlaylistItemsTool(),
            YouTubeUploadVideoTool(),
        ]
        for tool in all_tools:
            assert tool.integration_type == "youtube", f"{tool.name} should have 'youtube' integration type"

    def test_all_tools_have_google_provider(self):
        """Test all tools have 'google' as provider."""
        all_tools = [
            YouTubeSearchTool(),
            YouTubeGetVideoTool(),
            YouTubeGetChannelTool(),
            YouTubeListPlaylistsTool(),
            YouTubeGetPlaylistItemsTool(),
            YouTubeUploadVideoTool(),
        ]
        for tool in all_tools:
            assert tool.provider == "google", f"{tool.name} should have 'google' provider"
