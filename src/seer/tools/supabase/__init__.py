from __future__ import annotations

from seer.logger import get_logger
from seer.tools.base import register_tool
from seer.tools.supabase.auth_admin import (
    SupabaseAuthAdminCreateUserTool,
    SupabaseAuthAdminDeleteUserTool,
    SupabaseAuthAdminListUsersTool,
)
from seer.tools.supabase.database import (
    SupabaseRpcCallTool,
    SupabaseTableDeleteTool,
    SupabaseTableInsertTool,
    SupabaseTableQueryTool,
    SupabaseTableUpdateTool,
    SupabaseTableUpsertTool,
)
from seer.tools.supabase.edge_functions import SupabaseFunctionInvokeTool
from seer.tools.supabase.storage import (
    SupabaseStorageCreateBucketTool,
    SupabaseStorageCreateSignedObjectUrlTool,
    SupabaseStorageDownloadObjectTool,
    SupabaseStorageListBucketsTool,
    SupabaseStorageUploadObjectTool,
)

logger = get_logger("shared.tools.supabase")


def register_supabase_tools() -> None:
    register_tool(SupabaseTableQueryTool())
    register_tool(SupabaseTableInsertTool())
    register_tool(SupabaseTableUpsertTool())
    register_tool(SupabaseTableUpdateTool())
    register_tool(SupabaseTableDeleteTool())
    register_tool(SupabaseRpcCallTool())
    register_tool(SupabaseFunctionInvokeTool())
    register_tool(SupabaseAuthAdminListUsersTool())
    register_tool(SupabaseAuthAdminCreateUserTool())
    register_tool(SupabaseAuthAdminDeleteUserTool())
    register_tool(SupabaseStorageListBucketsTool())
    register_tool(SupabaseStorageCreateBucketTool())
    register_tool(SupabaseStorageUploadObjectTool())
    register_tool(SupabaseStorageDownloadObjectTool())
    register_tool(SupabaseStorageCreateSignedObjectUrlTool())


__all__ = [
    "SupabaseTableQueryTool",
    "SupabaseFunctionInvokeTool",
    "SupabaseAuthAdminListUsersTool",
    "SupabaseAuthAdminCreateUserTool",
    "SupabaseAuthAdminDeleteUserTool",
    "SupabaseStorageListBucketsTool",
    "SupabaseStorageCreateBucketTool",
    "SupabaseStorageUploadObjectTool",
    "SupabaseStorageDownloadObjectTool",
    "SupabaseStorageCreateSignedObjectUrlTool",
]


register_supabase_tools()
