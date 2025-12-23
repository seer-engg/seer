"""
Scope validation utilities for OAuth connections.

Validates that an OAuth connection has the required scopes for tool execution.
"""
from typing import List, Optional
from shared.database.models_oauth import OAuthConnection
from shared.logger import get_logger

logger = get_logger("shared.tools.scope_validator")


def validate_scope(connection: OAuthConnection, required_scope: str) -> bool:
    """
    Validate that an OAuth connection has the required scope.
    
    Args:
        connection: OAuthConnection instance
        required_scope: Required OAuth scope (e.g., "https://www.googleapis.com/auth/gmail.readonly")
    
    Returns:
        True if connection has the required scope, False otherwise
    
    Note:
        Handles scope matching logic:
        - Exact match: required scope must be in granted scopes
        - Partial match: if required is "gmail.readonly" and granted has "gmail", that's acceptable
          (but we prefer exact match for security)
        - For Google APIs, full URL scopes are required for exact match
    """
    if not connection.scopes:
        logger.warning(f"Connection {connection.id} has no scopes stored")
        return False
    
    granted_scopes = connection.scopes.strip()
    if not granted_scopes:
        return False
    
    # Split scopes - handle both comma-separated (GitHub) and space-separated (Google) formats
    # First check if scopes are comma-separated (no spaces between them)
    if ',' in granted_scopes and ' ' not in granted_scopes:
        granted_scope_list = [s.strip() for s in granted_scopes.split(',')]
    else:
        # Space-separated (Google style) or mixed - split by whitespace
        granted_scope_list = granted_scopes.split()
    
    # Check for exact match first
    if required_scope in granted_scope_list:
        return True
    
    # For Google APIs, check if we have a broader scope
    # e.g., if required is "gmail.readonly" and we have "gmail", that's not sufficient
    # But if required is "gmail" and we have "gmail.readonly", that's sufficient
    if "googleapis.com" in required_scope:
        # Check if any granted scope is a superset of required scope
        # This handles cases like: required="gmail.readonly", granted="gmail" (not sufficient)
        # But: required="gmail", granted="gmail.readonly" (sufficient)
        for granted in granted_scope_list:
            if granted == required_scope:
                return True
            # If granted scope contains required scope as a prefix (with proper delimiter)
            # This is conservative - we don't allow "gmail" to satisfy "gmail.readonly"
            if required_scope.startswith(granted + "."):
                return True
    
    logger.warning(
        f"Connection {connection.id} missing required scope '{required_scope}'. "
        f"Granted scopes: {granted_scopes}"
    )
    return False


def validate_scopes(connection: OAuthConnection, required_scopes: List[str]) -> tuple[bool, Optional[str]]:
    """
    Validate that an OAuth connection has all required scopes.
    
    Args:
        connection: OAuthConnection instance
        required_scopes: List of required OAuth scopes
    
    Returns:
        Tuple of (is_valid, missing_scope)
        - is_valid: True if all scopes are present
        - missing_scope: First missing scope if any, None if all present
    """
    for scope in required_scopes:
        if not validate_scope(connection, scope):
            return False, scope
    return True, None

