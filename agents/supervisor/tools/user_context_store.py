"""
User context store - stores user_id and connection_ids per thread.
Thread-scoped storage for user credentials from LangGraph context.
"""
from typing import Dict, Optional, Any
from contextvars import ContextVar
from agents.supervisor.state import SupervisorState

# Context variable to track current thread_id in tool execution
_current_thread_id: ContextVar[Optional[str]] = ContextVar('_current_thread_id', default=None)

class UserContextStore:
    """Thread-scoped store for user context (user_id, connected_accounts)."""
    
    def __init__(self):
        # Key: thread_id, Value: dict with user_id and connected_accounts
        self._user_contexts: Dict[str, Dict[str, Any]] = {}
    
    def store_user_context(self, state: SupervisorState, thread_id: Optional[str] = None):
        """Store user context from SupervisorState for a thread.
        
        Args:
            state: SupervisorState with context field
            thread_id: Optional thread ID. If None, uses "default"
        """
        thread_id = thread_id or "default"
        
        context = state.get("context", {})
        integrations = context.get("integrations", {})
        
        # Extract user_id from context
        user_id = context.get("user_id") or context.get("user_email")
        
        # Debug logging to verify what user_id Supervisor received
        import logging
        logger = logging.getLogger(__name__)
        if user_id:
            logger.info("[UserContextStore] Received user_id from context: %s", user_id)
        else:
            logger.warning("[UserContextStore] No user_id found in context, will use env var fallback")
        
        # Extract connected account IDs from integrations
        # Skip sandbox mode - only use actual connected accounts
        connected_accounts = {}
        logger.info("[UserContextStore] Raw integrations dict: %s", integrations)
        for integration_type, selection in integrations.items():
            logger.info("[UserContextStore] Processing integration %s: %s", integration_type, selection)
            if isinstance(selection, dict):
                # Skip if mode is "sandbox" or id is "sandbox"
                mode = selection.get("mode")
                account_id = selection.get("id")
                logger.info("[UserContextStore] Integration %s - mode: %s, id: %s", integration_type, mode, account_id)
                
                # account_id is now OAuthConnection.id (format: "provider:id" or just "id")
                # No need to validate format - accept any ID
                
                if mode == "sandbox" or account_id == "sandbox":
                    # Don't add to connected_accounts - Supervisor will use default/sandbox account
                    logger.debug("[UserContextStore] Skipping sandbox selection for %s", integration_type)
                    continue
                if account_id:
                    connected_accounts[integration_type] = account_id
                    logger.info("[UserContextStore] Found connected account for %s: %s", integration_type, account_id)
            elif selection is None:
                logger.debug("[UserContextStore] Integration %s is None, skipping", integration_type)
        
        logger.info("[UserContextStore] Final connected_accounts: %s", connected_accounts)
        
        # Extract resource-specific IDs (workspace GID, project GID, repo ID, etc.)
        # These are passed from the frontend so workers don't need to discover them
        resource_ids = {}
        for integration_type, selection in integrations.items():
            if isinstance(selection, dict):
                logger.info("[UserContextStore] Checking resource IDs for %s: %s", integration_type, selection)
                # Log all keys in the selection dict to debug what fields are present
                logger.info("[UserContextStore] Selection keys for %s: %s", integration_type, list(selection.keys()))
                # Extract resource-specific IDs for each integration
                workspace_gid = selection.get("workspaceGid") or selection.get("workspace_gid")  # Try both camelCase and snake_case
                project_gid = selection.get("projectGid") or selection.get("project_gid")
                repo_id = selection.get("repoId") or selection.get("repo_id")
                folder_id = selection.get("folderId") or selection.get("folder_id")
                
                logger.info("[UserContextStore] Extracted values for %s: workspaceGid=%s, projectGid=%s, repoId=%s, folderId=%s", 
                           integration_type, workspace_gid, project_gid, repo_id, folder_id)
                
                if workspace_gid:
                    resource_ids[f"{integration_type}_workspace_gid"] = workspace_gid
                    logger.info("[UserContextStore] Found workspace_gid for %s: %s", integration_type, workspace_gid)
                if project_gid:
                    resource_ids[f"{integration_type}_project_gid"] = project_gid
                    logger.info("[UserContextStore] Found project_gid for %s: %s", integration_type, project_gid)
                if repo_id:
                    resource_ids[f"{integration_type}_repo_id"] = repo_id
                    logger.info("[UserContextStore] Found repo_id for %s: %s", integration_type, repo_id)
                if folder_id:
                    resource_ids[f"{integration_type}_folder_id"] = folder_id
                    logger.info("[UserContextStore] Found folder_id for %s: %s", integration_type, folder_id)
        
        if resource_ids:
            logger.info("[UserContextStore] Resource IDs extracted: %s", resource_ids)
        else:
            logger.warning("[UserContextStore] No resource IDs found in integrations context. Available keys in selections: %s", 
                          {k: list(v.keys()) if isinstance(v, dict) else str(v) for k, v in integrations.items()})
        
        self._user_contexts[thread_id] = {
            "user_id": user_id,
            "connected_accounts": connected_accounts,
            "resource_ids": resource_ids  # Store resource-specific IDs for workers
        }
    
    def get_user_context(self, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Get user context for a thread.
        
        Args:
            thread_id: Optional thread ID. If None, tries to get from context variable, then falls back to "default"
        
        Returns:
            dict with:
            - user_id: str (from context or env var fallback)
            - connected_accounts: Dict[str, str] (maps integration_type -> connection_id)
        """
        # Try to get thread_id from context variable (set by worker invocation)
        if thread_id is None:
            thread_id = _current_thread_id.get()
        
        # Fallback to "default" if still None
        thread_id = thread_id or "default"
        
        context = self._user_contexts.get(thread_id, {})
        
        return {
            "user_id": context.get("user_id"),
            "connected_accounts": context.get("connected_accounts", {}),
            "resource_ids": context.get("resource_ids", {})  # Include resource-specific IDs (workspaceGid, projectGid, etc.)
        }
    
    def set_current_thread_id(self, thread_id: str):
        """Set the current thread_id in context variable for tool execution."""
        _current_thread_id.set(thread_id)
    
    def clear_user_context(self, thread_id: Optional[str] = None):
        """Clear user context for a thread."""
        thread_id = thread_id or "default"
        self._user_contexts.pop(thread_id, None)

# Global instance
_user_context_store = UserContextStore()

def get_user_context_store() -> UserContextStore:
    """Get the global user context store instance."""
    return _user_context_store

