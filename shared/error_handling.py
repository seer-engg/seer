"""Standardized error handling utilities for Seer agents"""

import traceback
import json
from typing import Optional


def create_error_response(error_message: str, exception: Optional[Exception] = None) -> str:
    """
    Create standardized error response with optional traceback.
    
    Args:
        error_message: User-friendly error message
        exception: Optional exception to include traceback from
        
    Returns:
        JSON string with standardized error format
    """
    error_data = {
        "success": False,
        "error": error_message
    }
    
    if exception:
        error_data["traceback"] = traceback.format_exc()
    
    return json.dumps(error_data)


def create_success_response(data: dict = None) -> str:
    """
    Create standardized success response.
    
    Args:
        data: Optional additional data to include
        
    Returns:
        JSON string with standardized success format
    """
    response = {"success": True}
    if data:
        response.update(data)
    
    return json.dumps(response)
