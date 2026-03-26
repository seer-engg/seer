"""
Workflow services module.
Provides backwards-compatible imports for the router.
"""

# Import from catalog module
from .catalog import (
    get_trigger_accounts,
    list_mcp_tools,
    list_browser_models,
    list_models,
    list_node_types,
    list_triggers,
)

# Import from execution module
from .execution import (
    _create_run_record,
    cancel_workflow_run,
    get_workflow_run_interrupt,
    list_workflow_runs,
    resume_workflow_run,
    run_saved_workflow,
)

# Import from history module
from .history import (
    get_run_history,
    get_run_status,
)

# Import from workflows module
from .lifecycle import (
    apply_workflow_from_spec,
    create_workflow,
    delete_workflow,
    export_workflow,
    get_workflow,
    import_workflow,
    list_workflow_versions,
    list_workflows,
    patch_workflow_draft,
    publish_workflow,
    restore_workflow_version,
    update_workflow,
)

# Import from triggers module
from .triggers import (
    delete_trigger_subscription,
    generate_cron_event,
    get_pending_events,
    get_subscription_event_count,
    get_trigger_subscription,
    list_trigger_subscriptions,
    list_trigger_subscriptions_extended,
    start_listening_for_trigger,
    test_trigger_subscription,
    sync_trigger_subscriptions,
    update_trigger_subscription,
)

# Import from files module
from .files import (
    list_run_files,
    get_run_file,
    get_run_file_download_url,
    delete_run_file,
)

# Import from share_template module
from .share_template import generate_template_description

__all__ = [
    "generate_template_description",
    "list_node_types",
    "list_mcp_tools",
    "list_browser_models",
    "list_models",
    "list_triggers",
    "get_trigger_accounts",
    "create_workflow",
    "list_workflows",
    "get_workflow",
    "list_workflow_versions",
    "update_workflow",
    "apply_workflow_from_spec",
    "patch_workflow_draft",
    "restore_workflow_version",
    "publish_workflow",
    "delete_workflow",
    "export_workflow",
    "import_workflow",
    "run_saved_workflow",
    "resume_workflow_run",
    "get_workflow_run_interrupt",
    "list_workflow_runs",
    "get_run_status",
    "get_run_history",
    "list_trigger_subscriptions",
    "list_trigger_subscriptions_extended",
    "get_trigger_subscription",
    "update_trigger_subscription",
    "delete_trigger_subscription",
    "test_trigger_subscription",
    "sync_trigger_subscriptions",
    "start_listening_for_trigger",
    "get_pending_events",
    "get_subscription_event_count",
    "generate_cron_event",
    "_create_run_record",
    "cancel_workflow_run",
    # File management
    "list_run_files",
    "get_run_file",
    "get_run_file_download_url",
    "delete_run_file",
]
