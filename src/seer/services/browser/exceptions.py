"""Custom exceptions for browser service error handling."""


class BrowserServiceError(Exception):
    """Base exception for browser service errors."""


class TargetDetachmentError(BrowserServiceError):
    """CDP target detached unexpectedly and recovery failed.

    This typically occurs when:
    - The browser tab crashes or is killed
    - Navigation causes the page to unload unexpectedly
    - CDP connection is lost during an operation
    - browser-use's focus recovery mechanism times out
    """

    def __init__(self, session_id: str, message: str = "CDP target detached"):
        self.session_id = session_id
        super().__init__(f"{message} for session {session_id}")


class CDPConnectionError(BrowserServiceError):
    """CDP connection to Chrome lost or failed."""


class BrowserSessionExpiredError(BrowserServiceError):
    """Browser session timed out or was killed."""
