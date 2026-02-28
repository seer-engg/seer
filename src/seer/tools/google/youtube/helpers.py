"""
Shared helpers and constants for YouTube tools.
"""

from typing import Any, Dict

# YouTube Data API v3 base URL
YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"

# YouTube upload URL (uses resumable uploads endpoint)
YOUTUBE_UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"

# Common integration type for all YouTube tools
YOUTUBE_INTEGRATION_TYPE = "youtube"

# OAuth scopes
YOUTUBE_READONLY_SCOPE = ["https://www.googleapis.com/auth/youtube.readonly"]
YOUTUBE_UPLOAD_SCOPE = ["https://www.googleapis.com/auth/youtube.upload"]
YOUTUBE_FULL_SCOPE = ["https://www.googleapis.com/auth/youtube"]


# ==============================================================================
# PARAMETER SCHEMAS
# ==============================================================================

VIDEO_ID_PARAM_SCHEMA: Dict[str, Any] = {
    "type": "string",
    "description": "YouTube video ID (e.g., 'dQw4w9WgXcQ')",
}

CHANNEL_ID_PARAM_SCHEMA: Dict[str, Any] = {
    "type": "string",
    "description": "YouTube channel ID (e.g., 'UC...')",
}

PLAYLIST_ID_PARAM_SCHEMA: Dict[str, Any] = {
    "type": "string",
    "description": "YouTube playlist ID (e.g., 'PL...')",
}

MAX_RESULTS_PARAM_SCHEMA: Dict[str, Any] = {
    "type": "integer",
    "description": "Maximum number of results to return (default: 25, max: 50)",
    "minimum": 1,
    "maximum": 50,
    "default": 25,
}

PAGE_TOKEN_PARAM_SCHEMA: Dict[str, Any] = {
    "type": "string",
    "description": "Token for pagination to fetch next page of results",
}


# ==============================================================================
# OUTPUT SCHEMAS - Video
# ==============================================================================

THUMBNAIL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "Thumbnail URL"},
        "width": {"type": "integer", "description": "Thumbnail width in pixels"},
        "height": {"type": "integer", "description": "Thumbnail height in pixels"},
    },
}

THUMBNAILS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "default": THUMBNAIL_SCHEMA,
        "medium": THUMBNAIL_SCHEMA,
        "high": THUMBNAIL_SCHEMA,
        "standard": THUMBNAIL_SCHEMA,
        "maxres": THUMBNAIL_SCHEMA,
    },
    "additionalProperties": True,
}

VIDEO_SNIPPET_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "publishedAt": {"type": "string", "description": "Publication timestamp (ISO 8601)"},
        "channelId": {"type": "string", "description": "Channel ID"},
        "title": {"type": "string", "description": "Video title"},
        "description": {"type": "string", "description": "Video description"},
        "thumbnails": THUMBNAILS_SCHEMA,
        "channelTitle": {"type": "string", "description": "Channel name"},
        "tags": {"type": "array", "items": {"type": "string"}, "description": "Video tags"},
        "categoryId": {"type": "string", "description": "Video category ID"},
        "liveBroadcastContent": {"type": "string", "description": "Live broadcast status"},
        "defaultLanguage": {"type": "string", "description": "Default language"},
        "defaultAudioLanguage": {"type": "string", "description": "Default audio language"},
    },
    "additionalProperties": True,
}

VIDEO_CONTENT_DETAILS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "duration": {"type": "string", "description": "Video duration (ISO 8601 format, e.g., 'PT4M13S')"},
        "dimension": {"type": "string", "description": "Video dimension (2d or 3d)"},
        "definition": {"type": "string", "description": "Video definition (hd or sd)"},
        "caption": {"type": "string", "description": "Whether captions are available (true/false)"},
        "licensedContent": {"type": "boolean", "description": "Whether video has licensed content"},
        "projection": {"type": "string", "description": "Video projection (rectangular or 360)"},
    },
    "additionalProperties": True,
}

VIDEO_STATISTICS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "viewCount": {"type": "string", "description": "View count"},
        "likeCount": {"type": "string", "description": "Like count"},
        "dislikeCount": {"type": "string", "description": "Dislike count (deprecated)"},
        "favoriteCount": {"type": "string", "description": "Favorite count"},
        "commentCount": {"type": "string", "description": "Comment count"},
    },
    "additionalProperties": True,
}

VIDEO_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "description": "Resource type ('youtube#video')"},
        "etag": {"type": "string", "description": "ETag"},
        "id": {"type": "string", "description": "Video ID"},
        "snippet": VIDEO_SNIPPET_SCHEMA,
        "contentDetails": VIDEO_CONTENT_DETAILS_SCHEMA,
        "statistics": VIDEO_STATISTICS_SCHEMA,
    },
    "additionalProperties": True,
}

VIDEO_LIST_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "description": "Resource type ('youtube#videoListResponse')"},
        "etag": {"type": "string", "description": "ETag"},
        "nextPageToken": {"type": "string", "description": "Token for next page"},
        "prevPageToken": {"type": "string", "description": "Token for previous page"},
        "pageInfo": {
            "type": "object",
            "properties": {
                "totalResults": {"type": "integer", "description": "Total number of results"},
                "resultsPerPage": {"type": "integer", "description": "Results per page"},
            },
        },
        "items": {
            "type": "array",
            "items": VIDEO_SCHEMA,
        },
    },
    "additionalProperties": True,
}


# ==============================================================================
# OUTPUT SCHEMAS - Channel
# ==============================================================================

CHANNEL_SNIPPET_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Channel title"},
        "description": {"type": "string", "description": "Channel description"},
        "customUrl": {"type": "string", "description": "Custom channel URL"},
        "publishedAt": {"type": "string", "description": "Channel creation timestamp (ISO 8601)"},
        "thumbnails": THUMBNAILS_SCHEMA,
        "country": {"type": "string", "description": "Country associated with the channel"},
    },
    "additionalProperties": True,
}

CHANNEL_STATISTICS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "viewCount": {"type": "string", "description": "Total view count"},
        "subscriberCount": {"type": "string", "description": "Subscriber count"},
        "hiddenSubscriberCount": {"type": "boolean", "description": "Whether subscriber count is hidden"},
        "videoCount": {"type": "string", "description": "Number of videos"},
    },
    "additionalProperties": True,
}

CHANNEL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "description": "Resource type ('youtube#channel')"},
        "etag": {"type": "string", "description": "ETag"},
        "id": {"type": "string", "description": "Channel ID"},
        "snippet": CHANNEL_SNIPPET_SCHEMA,
        "statistics": CHANNEL_STATISTICS_SCHEMA,
    },
    "additionalProperties": True,
}

CHANNEL_LIST_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "description": "Resource type ('youtube#channelListResponse')"},
        "etag": {"type": "string", "description": "ETag"},
        "nextPageToken": {"type": "string", "description": "Token for next page"},
        "pageInfo": {
            "type": "object",
            "properties": {
                "totalResults": {"type": "integer", "description": "Total number of results"},
                "resultsPerPage": {"type": "integer", "description": "Results per page"},
            },
        },
        "items": {
            "type": "array",
            "items": CHANNEL_SCHEMA,
        },
    },
    "additionalProperties": True,
}


# ==============================================================================
# OUTPUT SCHEMAS - Playlist
# ==============================================================================

PLAYLIST_SNIPPET_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "publishedAt": {"type": "string", "description": "Playlist creation timestamp (ISO 8601)"},
        "channelId": {"type": "string", "description": "Channel ID"},
        "title": {"type": "string", "description": "Playlist title"},
        "description": {"type": "string", "description": "Playlist description"},
        "thumbnails": THUMBNAILS_SCHEMA,
        "channelTitle": {"type": "string", "description": "Channel name"},
    },
    "additionalProperties": True,
}

PLAYLIST_CONTENT_DETAILS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "itemCount": {"type": "integer", "description": "Number of items in playlist"},
    },
}

PLAYLIST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "description": "Resource type ('youtube#playlist')"},
        "etag": {"type": "string", "description": "ETag"},
        "id": {"type": "string", "description": "Playlist ID"},
        "snippet": PLAYLIST_SNIPPET_SCHEMA,
        "contentDetails": PLAYLIST_CONTENT_DETAILS_SCHEMA,
    },
    "additionalProperties": True,
}

PLAYLIST_LIST_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "description": "Resource type ('youtube#playlistListResponse')"},
        "etag": {"type": "string", "description": "ETag"},
        "nextPageToken": {"type": "string", "description": "Token for next page"},
        "pageInfo": {
            "type": "object",
            "properties": {
                "totalResults": {"type": "integer", "description": "Total number of results"},
                "resultsPerPage": {"type": "integer", "description": "Results per page"},
            },
        },
        "items": {
            "type": "array",
            "items": PLAYLIST_SCHEMA,
        },
    },
    "additionalProperties": True,
}


# ==============================================================================
# OUTPUT SCHEMAS - Playlist Items
# ==============================================================================

PLAYLIST_ITEM_SNIPPET_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "publishedAt": {"type": "string", "description": "Item addition timestamp (ISO 8601)"},
        "channelId": {"type": "string", "description": "Channel ID"},
        "title": {"type": "string", "description": "Video title"},
        "description": {"type": "string", "description": "Video description"},
        "thumbnails": THUMBNAILS_SCHEMA,
        "channelTitle": {"type": "string", "description": "Channel name"},
        "playlistId": {"type": "string", "description": "Playlist ID"},
        "position": {"type": "integer", "description": "Position in playlist"},
        "resourceId": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "description": "Resource type ('youtube#video')"},
                "videoId": {"type": "string", "description": "Video ID"},
            },
        },
        "videoOwnerChannelTitle": {"type": "string", "description": "Video owner channel name"},
        "videoOwnerChannelId": {"type": "string", "description": "Video owner channel ID"},
    },
    "additionalProperties": True,
}

PLAYLIST_ITEM_CONTENT_DETAILS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "videoId": {"type": "string", "description": "Video ID"},
        "startAt": {"type": "string", "description": "Start position"},
        "endAt": {"type": "string", "description": "End position"},
        "note": {"type": "string", "description": "Note about the item"},
        "videoPublishedAt": {"type": "string", "description": "Video publication timestamp"},
    },
    "additionalProperties": True,
}

PLAYLIST_ITEM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "description": "Resource type ('youtube#playlistItem')"},
        "etag": {"type": "string", "description": "ETag"},
        "id": {"type": "string", "description": "Playlist item ID"},
        "snippet": PLAYLIST_ITEM_SNIPPET_SCHEMA,
        "contentDetails": PLAYLIST_ITEM_CONTENT_DETAILS_SCHEMA,
    },
    "additionalProperties": True,
}

PLAYLIST_ITEM_LIST_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "description": "Resource type ('youtube#playlistItemListResponse')"},
        "etag": {"type": "string", "description": "ETag"},
        "nextPageToken": {"type": "string", "description": "Token for next page"},
        "pageInfo": {
            "type": "object",
            "properties": {
                "totalResults": {"type": "integer", "description": "Total number of results"},
                "resultsPerPage": {"type": "integer", "description": "Results per page"},
            },
        },
        "items": {
            "type": "array",
            "items": PLAYLIST_ITEM_SCHEMA,
        },
    },
    "additionalProperties": True,
}


# ==============================================================================
# OUTPUT SCHEMAS - Search
# ==============================================================================

SEARCH_RESULT_ID_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "description": "Resource type ('youtube#video', 'youtube#channel', 'youtube#playlist')"},
        "videoId": {"type": "string", "description": "Video ID (if type is video)"},
        "channelId": {"type": "string", "description": "Channel ID (if type is channel)"},
        "playlistId": {"type": "string", "description": "Playlist ID (if type is playlist)"},
    },
}

SEARCH_RESULT_SNIPPET_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "publishedAt": {"type": "string", "description": "Publication timestamp (ISO 8601)"},
        "channelId": {"type": "string", "description": "Channel ID"},
        "title": {"type": "string", "description": "Title"},
        "description": {"type": "string", "description": "Description"},
        "thumbnails": THUMBNAILS_SCHEMA,
        "channelTitle": {"type": "string", "description": "Channel name"},
        "liveBroadcastContent": {"type": "string", "description": "Live broadcast status"},
    },
    "additionalProperties": True,
}

SEARCH_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "description": "Resource type ('youtube#searchResult')"},
        "etag": {"type": "string", "description": "ETag"},
        "id": SEARCH_RESULT_ID_SCHEMA,
        "snippet": SEARCH_RESULT_SNIPPET_SCHEMA,
    },
    "additionalProperties": True,
}

SEARCH_LIST_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "description": "Resource type ('youtube#searchListResponse')"},
        "etag": {"type": "string", "description": "ETag"},
        "nextPageToken": {"type": "string", "description": "Token for next page"},
        "prevPageToken": {"type": "string", "description": "Token for previous page"},
        "regionCode": {"type": "string", "description": "Region code used for search"},
        "pageInfo": {
            "type": "object",
            "properties": {
                "totalResults": {"type": "integer", "description": "Total number of results"},
                "resultsPerPage": {"type": "integer", "description": "Results per page"},
            },
        },
        "items": {
            "type": "array",
            "items": SEARCH_RESULT_SCHEMA,
        },
    },
    "additionalProperties": True,
}


# ==============================================================================
# VIDEO UPLOAD SCHEMAS
# ==============================================================================

VIDEO_UPLOAD_REQUEST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "snippet": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Video title (required, max 100 chars)"},
                "description": {"type": "string", "description": "Video description (max 5000 chars)"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Video tags"},
                "categoryId": {"type": "string", "description": "Video category ID (default: 22 for People & Blogs)"},
                "defaultLanguage": {"type": "string", "description": "Default language (ISO 639-1)"},
            },
            "required": ["title"],
        },
        "status": {
            "type": "object",
            "properties": {
                "privacyStatus": {
                    "type": "string",
                    "enum": ["private", "public", "unlisted"],
                    "description": "Video privacy status (default: private)",
                },
                "publishAt": {"type": "string", "description": "Scheduled publish time (ISO 8601)"},
                "selfDeclaredMadeForKids": {"type": "boolean", "description": "Whether video is made for kids"},
                "embeddable": {"type": "boolean", "description": "Whether video is embeddable (default: true)"},
                "license": {
                    "type": "string",
                    "enum": ["youtube", "creativeCommon"],
                    "description": "Video license (default: youtube)",
                },
            },
        },
    },
    "required": ["snippet"],
}
