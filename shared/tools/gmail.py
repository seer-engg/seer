"""
Gmail read tool using direct HTTP API calls.

Uses Gmail REST API with OAuth tokens - no google-auth library.
"""
from typing import Any, Dict, List, Optional
import httpx
from fastapi import HTTPException

from shared.tools.base import BaseTool, register_tool
from shared.logger import get_logger

logger = get_logger("shared.tools.gmail")


class GmailReadTool(BaseTool):
    """
    Tool for reading emails from Gmail inbox.
    
    Uses Gmail REST API v1 with direct HTTP calls.
    Requires OAuth scope: https://www.googleapis.com/auth/gmail.readonly
    """
    
    name = "gmail_read_emails"
    description = "Read emails from Gmail inbox. Supports filtering by labels, query, and max results."
    required_scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get JSON schema for Gmail read tool parameters."""
        return {
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of emails to return (default: 10, max: 100)",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10
                },
                "label_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of label IDs to filter by (e.g., ['INBOX', 'UNREAD'])",
                    "default": ["INBOX"]
                },
                "q": {
                    "type": "string",
                    "description": "Gmail search query (e.g., 'from:example@gmail.com', 'subject:meeting')",
                    "default": None
                },
                "include_body": {
                    "type": "boolean",
                    "description": "Whether to include full email body (default: false)",
                    "default": False
                }
            },
            "required": []
        }
    
    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute Gmail read tool.
        
        Args:
            access_token: OAuth access token (required)
            arguments: Tool arguments
        
        Returns:
            List of email objects with id, threadId, snippet, payload (subject, from, date)
        """
        if not access_token:
            raise HTTPException(
                status_code=401,
                detail="Gmail tool requires OAuth access token"
            )
        
        max_results = min(arguments.get("max_results", 10), 100)
        label_ids = arguments.get("label_ids", ["INBOX"])
        query = arguments.get("q")
        include_body = arguments.get("include_body", False)
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
        
        # Build query parameters for listing messages
        params: Dict[str, Any] = {
            "maxResults": max_results
        }
        
        if label_ids:
            params["labelIds"] = ",".join(label_ids)
        
        if query:
            params["q"] = query
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Step 1: List messages
                logger.info(f"Fetching Gmail messages: max_results={max_results}, label_ids={label_ids}, q={query}")
                
                list_response = await client.get(
                    "https://www.googleapis.com/gmail/v1/users/me/messages",
                    headers=headers,
                    params=params
                )
                
                if list_response.status_code == 401:
                    raise HTTPException(
                        status_code=401,
                        detail="Gmail API authentication failed. Token may be expired or invalid."
                    )
                
                list_response.raise_for_status()
                list_data = list_response.json()
                
                messages = list_data.get("messages", [])
                if not messages:
                    logger.info("No messages found matching criteria")
                    return []
                
                logger.info(f"Found {len(messages)} messages, fetching details...")
                
                # Step 2: Fetch full message details
                results = []
                for msg in messages[:max_results]:
                    msg_id = msg["id"]
                    
                    # Build params for message detail
                    msg_params: Dict[str, Any] = {}
                    if include_body:
                        msg_params["format"] = "full"
                    else:
                        msg_params["format"] = "metadata"
                        msg_params["metadataHeaders"] = "From,To,Subject,Date"
                    
                    msg_response = await client.get(
                        f"https://www.googleapis.com/gmail/v1/users/me/messages/{msg_id}",
                        headers=headers,
                        params=msg_params
                    )
                    
                    if msg_response.status_code == 404:
                        logger.warning(f"Message {msg_id} not found, skipping")
                        continue
                    
                    msg_response.raise_for_status()
                    msg_data = msg_response.json()
                    
                    # Extract relevant fields
                    payload = msg_data.get("payload", {})
                    headers_list = payload.get("headers", [])
                    
                    # Convert headers list to dict for easier access
                    headers_dict = {h["name"]: h["value"] for h in headers_list}
                    
                    # Build email object
                    email_obj = {
                        "id": msg_data.get("id"),
                        "threadId": msg_data.get("threadId"),
                        "snippet": msg_data.get("snippet", ""),
                        "subject": headers_dict.get("Subject", ""),
                        "from": headers_dict.get("From", ""),
                        "to": headers_dict.get("To", ""),
                        "date": headers_dict.get("Date", ""),
                        "labelIds": msg_data.get("labelIds", [])
                    }
                    
                    # Include body if requested
                    if include_body:
                        # Extract body from payload
                        body_text = ""
                        if payload.get("body", {}).get("data"):
                            import base64
                            body_data = payload["body"]["data"]
                            body_text = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="ignore")
                        elif payload.get("parts"):
                            # Multipart message
                            for part in payload["parts"]:
                                if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                                    import base64
                                    body_data = part["body"]["data"]
                                    body_text = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="ignore")
                                    break
                        
                        email_obj["body"] = body_text
                    
                    results.append(email_obj)
                
                logger.info(f"Successfully fetched {len(results)} email details")
                return results
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Gmail API error: {e.response.status_code} - {e.response.text[:500]}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Gmail API error: {e.response.text[:500]}"
            )
        except httpx.TimeoutException:
            logger.error("Gmail API request timed out")
            raise HTTPException(
                status_code=504,
                detail="Gmail API request timed out"
            )
        except Exception as e:
            logger.exception(f"Unexpected error reading Gmail: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error reading Gmail: {str(e)}"
            )


# Register the tool
_gmail_tool = GmailReadTool()
register_tool(_gmail_tool)

