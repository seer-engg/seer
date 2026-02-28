"""
YouTube read operations - search, video details, channel info, and playlists.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from seer.logger import get_logger
from seer.tools.google.base import GoogleAPIClient
from seer.tools.google.youtube.helpers import (
    CHANNEL_ID_PARAM_SCHEMA,
    CHANNEL_LIST_RESPONSE_SCHEMA,
    MAX_RESULTS_PARAM_SCHEMA,
    PAGE_TOKEN_PARAM_SCHEMA,
    PLAYLIST_ID_PARAM_SCHEMA,
    PLAYLIST_ITEM_LIST_RESPONSE_SCHEMA,
    PLAYLIST_LIST_RESPONSE_SCHEMA,
    SEARCH_LIST_RESPONSE_SCHEMA,
    VIDEO_ID_PARAM_SCHEMA,
    VIDEO_LIST_RESPONSE_SCHEMA,
    YOUTUBE_API_BASE,
    YOUTUBE_INTEGRATION_TYPE,
    YOUTUBE_READONLY_SCOPE,
)

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("shared.tools.youtube.read")


class YouTubeSearchTool(GoogleAPIClient):
    """Search for YouTube videos, channels, or playlists."""

    name = "youtube_search"
    description = "Search YouTube for videos, channels, or playlists. Returns matching results with basic metadata."
    required_scopes = YOUTUBE_READONLY_SCOPE
    integration_type = YOUTUBE_INTEGRATION_TYPE

    def get_output_schema(self) -> Dict[str, Any]:
        return SEARCH_LIST_RESPONSE_SCHEMA

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "q": {
                    "type": "string",
                    "description": "Search query string",
                },
                "type": {
                    "type": "string",
                    "description": "Type of resource to search for",
                    "enum": ["video", "channel", "playlist"],
                    "default": "video",
                },
                "max_results": MAX_RESULTS_PARAM_SCHEMA,
                "order": {
                    "type": "string",
                    "description": "Sort order for results",
                    "enum": ["date", "rating", "relevance", "title", "videoCount", "viewCount"],
                    "default": "relevance",
                },
                "published_after": {
                    "type": "string",
                    "description": "Filter for videos published after this date (ISO 8601, e.g., '2024-01-01T00:00:00Z')",
                },
                "published_before": {
                    "type": "string",
                    "description": "Filter for videos published before this date (ISO 8601)",
                },
                "channel_id": {
                    "type": "string",
                    "description": "Restrict search to a specific channel",
                },
                "video_duration": {
                    "type": "string",
                    "description": "Filter by video duration",
                    "enum": ["any", "long", "medium", "short"],
                },
                "video_definition": {
                    "type": "string",
                    "description": "Filter by video definition",
                    "enum": ["any", "high", "standard"],
                },
                "region_code": {
                    "type": "string",
                    "description": "ISO 3166-1 alpha-2 country code to restrict search results",
                },
                "relevance_language": {
                    "type": "string",
                    "description": "ISO 639-1 language code to prioritize results in that language",
                },
                "safe_search": {
                    "type": "string",
                    "description": "Safe search filtering mode",
                    "enum": ["moderate", "none", "strict"],
                    "default": "moderate",
                },
                "page_token": PAGE_TOKEN_PARAM_SCHEMA,
            },
            "required": ["q"],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = credentials, context

        query = arguments.get("q")
        if not query:
            raise ValueError("Search query (q) is required")

        max_results = min(max(1, int(arguments.get("max_results", 25) or 25)), 50)

        params: Dict[str, Any] = {
            "part": "snippet",
            "q": query,
            "type": arguments.get("type", "video"),
            "maxResults": max_results,
            "order": arguments.get("order", "relevance"),
            "safeSearch": arguments.get("safe_search", "moderate"),
        }

        if arguments.get("published_after"):
            params["publishedAfter"] = arguments["published_after"]
        if arguments.get("published_before"):
            params["publishedBefore"] = arguments["published_before"]
        if arguments.get("channel_id"):
            params["channelId"] = arguments["channel_id"]
        if arguments.get("video_duration"):
            params["videoDuration"] = arguments["video_duration"]
        if arguments.get("video_definition"):
            params["videoDefinition"] = arguments["video_definition"]
        if arguments.get("region_code"):
            params["regionCode"] = arguments["region_code"]
        if arguments.get("relevance_language"):
            params["relevanceLanguage"] = arguments["relevance_language"]
        if arguments.get("page_token"):
            params["pageToken"] = arguments["page_token"]

        logger.info(
            "Searching YouTube: query=%s, type=%s, max_results=%s",
            query,
            params["type"],
            max_results,
        )

        resp = await self._make_request(
            "GET",
            f"{YOUTUBE_API_BASE}/search",
            access_token,
            params=params,
        )
        return resp.json()


class YouTubeGetVideoTool(GoogleAPIClient):
    """Get detailed information about one or more YouTube videos."""

    name = "youtube_get_video"
    description = "Get detailed information about YouTube videos including title, description, statistics, and content details."
    required_scopes = YOUTUBE_READONLY_SCOPE
    integration_type = YOUTUBE_INTEGRATION_TYPE

    def get_output_schema(self) -> Dict[str, Any]:
        return VIDEO_LIST_RESPONSE_SCHEMA

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "video_id": {
                    **VIDEO_ID_PARAM_SCHEMA,
                    "description": "YouTube video ID or comma-separated list of IDs (max 50)",
                },
                "part": {
                    "type": "string",
                    "description": "Comma-separated list of resource parts to include (snippet, contentDetails, statistics, status, player)",
                    "default": "snippet,contentDetails,statistics",
                },
            },
            "required": ["video_id"],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = credentials, context

        video_id = arguments.get("video_id")
        if not video_id:
            raise ValueError("video_id is required")

        part = arguments.get("part", "snippet,contentDetails,statistics")

        params: Dict[str, Any] = {
            "part": part,
            "id": video_id,
        }

        logger.info(
            "Getting YouTube video details: video_id=%s, part=%s",
            video_id,
            part,
        )

        resp = await self._make_request(
            "GET",
            f"{YOUTUBE_API_BASE}/videos",
            access_token,
            params=params,
        )
        return resp.json()


class YouTubeGetChannelTool(GoogleAPIClient):
    """Get information about a YouTube channel."""

    name = "youtube_get_channel"
    description = "Get information about a YouTube channel including title, description, subscriber count, and video count."
    required_scopes = YOUTUBE_READONLY_SCOPE
    integration_type = YOUTUBE_INTEGRATION_TYPE

    def get_output_schema(self) -> Dict[str, Any]:
        return CHANNEL_LIST_RESPONSE_SCHEMA

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "channel_id": {
                    **CHANNEL_ID_PARAM_SCHEMA,
                    "description": "YouTube channel ID, comma-separated list of IDs (max 50), or 'mine' for authenticated user's channel",
                },
                "for_username": {
                    "type": "string",
                    "description": "Legacy username of the channel (alternative to channel_id)",
                },
                "for_handle": {
                    "type": "string",
                    "description": "Channel handle (e.g., '@username', alternative to channel_id)",
                },
                "part": {
                    "type": "string",
                    "description": "Comma-separated list of resource parts to include (snippet, statistics, contentDetails, brandingSettings)",
                    "default": "snippet,statistics",
                },
            },
            "required": [],  # At least one of channel_id, for_username, or for_handle required
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = credentials, context

        channel_id = arguments.get("channel_id")
        for_username = arguments.get("for_username")
        for_handle = arguments.get("for_handle")

        if not channel_id and not for_username and not for_handle:
            raise ValueError("At least one of channel_id, for_username, or for_handle is required")

        part = arguments.get("part", "snippet,statistics")

        params: Dict[str, Any] = {
            "part": part,
        }

        if channel_id:
            if channel_id == "mine":
                params["mine"] = "true"
            else:
                params["id"] = channel_id
        elif for_username:
            params["forUsername"] = for_username
        elif for_handle:
            params["forHandle"] = for_handle

        logger.info(
            "Getting YouTube channel: channel_id=%s, for_username=%s, for_handle=%s, part=%s",
            channel_id,
            for_username,
            for_handle,
            part,
        )

        resp = await self._make_request(
            "GET",
            f"{YOUTUBE_API_BASE}/channels",
            access_token,
            params=params,
        )
        return resp.json()


class YouTubeListPlaylistsTool(GoogleAPIClient):
    """List playlists for a channel or the authenticated user."""

    name = "youtube_list_playlists"
    description = "List playlists from a YouTube channel or the authenticated user's account."
    required_scopes = YOUTUBE_READONLY_SCOPE
    integration_type = YOUTUBE_INTEGRATION_TYPE

    def get_output_schema(self) -> Dict[str, Any]:
        return PLAYLIST_LIST_RESPONSE_SCHEMA

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "channel_id": {
                    **CHANNEL_ID_PARAM_SCHEMA,
                    "description": "Channel ID to list playlists from (omit to list authenticated user's playlists)",
                },
                "playlist_id": {
                    **PLAYLIST_ID_PARAM_SCHEMA,
                    "description": "Comma-separated list of playlist IDs to retrieve (max 50)",
                },
                "mine": {
                    "type": "boolean",
                    "description": "If true, list only the authenticated user's playlists",
                    "default": False,
                },
                "max_results": MAX_RESULTS_PARAM_SCHEMA,
                "part": {
                    "type": "string",
                    "description": "Comma-separated list of resource parts to include (snippet, contentDetails, status, player)",
                    "default": "snippet,contentDetails",
                },
                "page_token": PAGE_TOKEN_PARAM_SCHEMA,
            },
            "required": [],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = credentials, context

        channel_id = arguments.get("channel_id")
        playlist_id = arguments.get("playlist_id")
        mine = arguments.get("mine", False)

        if not channel_id and not playlist_id and not mine:
            raise ValueError("At least one of channel_id, playlist_id, or mine=true is required")

        max_results = min(max(1, int(arguments.get("max_results", 25) or 25)), 50)
        part = arguments.get("part", "snippet,contentDetails")

        params: Dict[str, Any] = {
            "part": part,
            "maxResults": max_results,
        }

        if channel_id:
            params["channelId"] = channel_id
        if playlist_id:
            params["id"] = playlist_id
        if mine:
            params["mine"] = "true"
        if arguments.get("page_token"):
            params["pageToken"] = arguments["page_token"]

        logger.info(
            "Listing YouTube playlists: channel_id=%s, mine=%s, max_results=%s",
            channel_id,
            mine,
            max_results,
        )

        resp = await self._make_request(
            "GET",
            f"{YOUTUBE_API_BASE}/playlists",
            access_token,
            params=params,
        )
        return resp.json()


class YouTubeGetPlaylistItemsTool(GoogleAPIClient):
    """Get videos in a YouTube playlist."""

    name = "youtube_get_playlist_items"
    description = "Get the videos contained in a YouTube playlist with their metadata."
    required_scopes = YOUTUBE_READONLY_SCOPE
    integration_type = YOUTUBE_INTEGRATION_TYPE

    def get_output_schema(self) -> Dict[str, Any]:
        return PLAYLIST_ITEM_LIST_RESPONSE_SCHEMA

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "playlist_id": {
                    **PLAYLIST_ID_PARAM_SCHEMA,
                    "description": "YouTube playlist ID to retrieve items from",
                },
                "max_results": MAX_RESULTS_PARAM_SCHEMA,
                "part": {
                    "type": "string",
                    "description": "Comma-separated list of resource parts to include (snippet, contentDetails, status)",
                    "default": "snippet,contentDetails",
                },
                "video_id": {
                    "type": "string",
                    "description": "Filter results to only include items for this video ID",
                },
                "page_token": PAGE_TOKEN_PARAM_SCHEMA,
            },
            "required": ["playlist_id"],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = credentials, context

        playlist_id = arguments.get("playlist_id")
        if not playlist_id:
            raise ValueError("playlist_id is required")

        max_results = min(max(1, int(arguments.get("max_results", 25) or 25)), 50)
        part = arguments.get("part", "snippet,contentDetails")

        params: Dict[str, Any] = {
            "part": part,
            "playlistId": playlist_id,
            "maxResults": max_results,
        }

        if arguments.get("video_id"):
            params["videoId"] = arguments["video_id"]
        if arguments.get("page_token"):
            params["pageToken"] = arguments["page_token"]

        logger.info(
            "Getting YouTube playlist items: playlist_id=%s, max_results=%s",
            playlist_id,
            max_results,
        )

        resp = await self._make_request(
            "GET",
            f"{YOUTUBE_API_BASE}/playlistItems",
            access_token,
            params=params,
        )
        return resp.json()
