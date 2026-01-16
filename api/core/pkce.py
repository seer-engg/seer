"""PKCE (Proof Key for Code Exchange) utilities for OAuth 2.1.

This module provides utilities for validating PKCE challenges and verifiers
as required by RFC 7636 and OAuth 2.1.

PKCE is mandatory for OAuth 2.1 and provides protection against authorization
code interception attacks.
"""

import hashlib
import base64
from typing import Optional


def generate_code_challenge(code_verifier: str, method: str = "S256") -> str:
    """Generate a code challenge from a code verifier.

    Args:
        code_verifier: The code verifier (43-128 character random string)
        method: Challenge method ("S256" for SHA-256, "plain" for no transformation)

    Returns:
        str: The code challenge

    Raises:
        ValueError: If method is not supported or verifier is invalid
    """
    if not code_verifier or len(code_verifier) < 43 or len(code_verifier) > 128:
        raise ValueError("Code verifier must be 43-128 characters")

    if method == "S256":
        # Hash the verifier with SHA-256
        digest = hashlib.sha256(code_verifier.encode('ascii')).digest()
        # Base64 URL-encode without padding
        challenge = base64.urlsafe_b64encode(digest).decode('ascii').rstrip('=')
        return challenge
    if method == "plain":
        # Plain method - challenge equals verifier
        return code_verifier
    raise ValueError(f"Unsupported code challenge method: {method}")


def validate_code_verifier(
    code_verifier: str,
    code_challenge: str,
    method: str = "S256"
) -> bool:
    """Validate a code verifier against a previously stored code challenge.

    This function should be called during the token exchange to verify that
    the client presenting the authorization code is the same one that initiated
    the authorization request.

    Args:
        code_verifier: The code verifier sent by the client during token exchange
        code_challenge: The code challenge stored during authorization
        method: The challenge method that was used ("S256" or "plain")

    Returns:
        bool: True if the verifier is valid, False otherwise

    Example:
        >>> # During authorization request
        >>> challenge = generate_code_challenge("my_verifier_123...")
        >>> # Store challenge in authorization code record
        >>>
        >>> # During token exchange
        >>> if validate_code_verifier("my_verifier_123...", stored_challenge):
        >>>     # Issue access token
        >>>     pass
    """
    if not code_verifier or not code_challenge:
        return False

    try:
        computed_challenge = generate_code_challenge(code_verifier, method)
        return computed_challenge == code_challenge
    except ValueError:
        return False


def validate_pkce_parameters(
    code_challenge: Optional[str],
    code_challenge_method: Optional[str] = "S256"
) -> tuple[bool, Optional[str]]:
    """Validate PKCE parameters during authorization request.

    Args:
        code_challenge: The code challenge from the authorization request
        code_challenge_method: The challenge method (default: "S256")

    Returns:
        tuple[bool, Optional[str]]: (is_valid, error_message)

    Example:
        >>> valid, error = validate_pkce_parameters(challenge, "S256")
        >>> if not valid:
        >>>     raise HTTPException(400, detail=error)
    """
    # OAuth 2.1 requires PKCE
    if not code_challenge:
        return False, "code_challenge is required (PKCE is mandatory for OAuth 2.1)"

    # Validate challenge method
    if code_challenge_method not in ["S256", "plain"]:
        return False, f"Unsupported code_challenge_method: {code_challenge_method}"

    # Validate challenge format (base64url encoded, 43-128 characters)
    if len(code_challenge) < 43 or len(code_challenge) > 128:
        return False, "code_challenge must be 43-128 characters"

    # Ensure it's base64url encoded (alphanumeric, -, _)
    if not all(c.isalnum() or c in '-_' for c in code_challenge):
        return False, "code_challenge must be base64url encoded"

    return True, None
