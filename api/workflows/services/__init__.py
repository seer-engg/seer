"""
Workflow services module.

Provides backwards-compatible imports for the router.
"""

# Import from catalog module
from .catalog import (
    compile_spec,
    list_models,
    list_node_types,
    list_tools,
    list_triggers,
    resolve_schema,
    validate_spec,
)

# Import from execution module
from .execution import (
    execute_saved_workflow_run,
    list_workflow_runs,
    run_draft_workflow,
    run_saved_workflow,
)

# Import from expressions module
from .expressions import (
    suggest_expression,
    typecheck_expression,
)

# Import from history module
from .history import (
    cancel_run,
    get_run_history,
    get_run_result,
    get_run_status,
    list_run_steps,
)

# Import from triggers module
from .triggers import (
    create_trigger_subscription,
    delete_trigger_subscription,
    get_trigger_subscription,
    list_trigger_subscriptions,
    test_trigger_subscription,
    update_trigger_subscription,
)

# Import from workflows module
from .workflows import (
    apply_workflow_from_spec,
    create_workflow,
    delete_workflow,
    get_workflow,
    list_workflow_versions,
    list_workflows,
    patch_workflow_draft,
    publish_workflow,
    restore_workflow_version,
    update_workflow,
)

# Re-export for backwards compatibility
__all__ = [
    # Catalog & registries
    "compile_spec",
    "list_models",
    "list_node_types",
    "list_tools",
    "list_triggers",
    "resolve_schema",
    "validate_spec",
    # Execution
    "execute_saved_workflow_run",
    "list_workflow_runs",
    "run_draft_workflow",
    "run_saved_workflow",
    # Expression support
    "suggest_expression",
    "typecheck_expression",
    # History & run management
    "cancel_run",
    "get_run_history",
    "get_run_result",
    "get_run_status",
    "list_run_steps",
    # Trigger subscriptions
    "create_trigger_subscription",
    "delete_trigger_subscription",
    "get_trigger_subscription",
    "list_trigger_subscriptions",
    "test_trigger_subscription",
    "update_trigger_subscription",
    # Workflow CRUD
    "apply_workflow_from_spec",
    "create_workflow",
    "delete_workflow",
    "get_workflow",
    "list_workflow_versions",
    "list_workflows",
    "patch_workflow_draft",
    "publish_workflow",
    "restore_workflow_version",
    "update_workflow",
]
