# pylint: disable=import-outside-toplevel,redefined-outer-name
# Reason: Test fixtures use lazy imports and pytest fixture pattern requires name reuse
"""
Taskiq in-process executor for E2E tests.

Instead of running tasks through Redis queues (which requires a worker process),
this fixture patches the .kiq() method to execute tasks directly in-process.

Benefits:
- Deterministic: Tasks execute immediately in the same async context
- Fast: No serialization/deserialization overhead
- Observable: Can track all executed tasks for assertions
- No subprocess: Simpler test setup
"""
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest


class TaskExecutionTracker:
    """
    Tracks task executions for test assertions.

    Records all tasks that were executed, their arguments, and results.
    """

    def __init__(self):
        self.executed_tasks: List[Dict[str, Any]] = []
        self.task_results: Dict[str, Any] = {}

    def record_execution(
        self,
        task_name: str,
        args: tuple,
        kwargs: dict,
        result: Any = None,
        error: Exception | None = None,
    ):
        """Record a task execution."""
        self.executed_tasks.append({
            "task_name": task_name,
            "args": args,
            "kwargs": kwargs,
            "result": result,
            "error": str(error) if error else None,
        })

    def get_executions(self, task_name: str) -> List[Dict[str, Any]]:
        """Get all executions of a specific task."""
        return [t for t in self.executed_tasks if t["task_name"] == task_name]

    def clear(self):
        """Clear all recorded executions."""
        self.executed_tasks.clear()
        self.task_results.clear()


def create_direct_kiq_wrapper(original_task, tracker: TaskExecutionTracker):
    """
    Create a wrapper that executes the task directly instead of queueing.

    The wrapper mimics the .kiq() interface but runs the task synchronously
    in the current async context.
    """
    async def direct_kiq(*args, **kwargs):
        """Execute task directly and return a mock TaskiqResult."""
        task_name = getattr(original_task, "task_name", original_task.__name__)

        # Create mock result object
        mock_result = AsyncMock()
        mock_result.task_id = f"mock_task_{len(tracker.executed_tasks)}"
        mock_result.is_ready = True

        try:
            # Execute the actual task function directly
            result = await original_task.fn(*args, **kwargs)
            tracker.record_execution(task_name, args, kwargs, result=result)
            mock_result.result = result
            mock_result.error = None
        except Exception as e:
            tracker.record_execution(task_name, args, kwargs, error=e)
            mock_result.result = None
            mock_result.error = e
            # Re-raise to match real behavior
            raise

        # Mock the wait_result method
        async def wait_result(timeout=None):
            return mock_result.result

        mock_result.wait_result = wait_result

        return mock_result

    return direct_kiq


@pytest.fixture(scope="function")
def taskiq_direct_executor(e2e_app) -> Dict[str, Any]:
    """
    Patch Taskiq tasks to execute directly in-process.

    This fixture patches the .kiq() method on workflow and trigger tasks
    to execute them immediately without going through Redis.

    IMPORTANT: Depends on e2e_app to ensure seer modules are imported
    with the correct test environment configuration.

    Args:
        e2e_app: FastAPI app fixture (ensures correct environment)

    Returns:
        Dict with:
        - tracker: TaskExecutionTracker for asserting on task executions
        - patches: List of active patches (for cleanup)

    Usage:
        async def test_workflow(taskiq_direct_executor, ...):
            # Execute something that triggers tasks
            await api_client.post("/v1/workflows/.../execute")

            # Assert tasks were executed
            tracker = taskiq_direct_executor["tracker"]
            assert len(tracker.get_executions("workflow_execution_task")) == 1
    """
    tracker = TaskExecutionTracker()
    patches = []

    # Import the actual task modules - this happens AFTER e2e_app has set up
    # the environment, so the config will be correct
    from seer.worker.tasks.workflows import workflow_execution_task
    from seer.worker.tasks.triggers import trigger_event_task

    # Patch workflow execution task
    workflow_kiq_wrapper = create_direct_kiq_wrapper(workflow_execution_task, tracker)
    workflow_patch = patch.object(workflow_execution_task, "kiq", workflow_kiq_wrapper)
    patches.append(workflow_patch)

    # Patch trigger event task
    trigger_kiq_wrapper = create_direct_kiq_wrapper(trigger_event_task, tracker)
    trigger_patch = patch.object(trigger_event_task, "kiq", trigger_kiq_wrapper)
    patches.append(trigger_patch)

    # Start all patches
    for p in patches:
        p.start()

    yield {
        "tracker": tracker,
        "patches": patches,
    }

    # Stop all patches
    for p in patches:
        p.stop()


@pytest.fixture(scope="function")
def mock_task_execution():
    """
    Simpler fixture that just mocks task execution without running code.

    Use this when you want to verify tasks are queued but don't need
    to actually execute them (faster for API-only tests).

    Returns:
        Dict with mock objects for each task type
    """
    from unittest.mock import AsyncMock, patch

    workflow_mock = AsyncMock()
    workflow_mock.return_value.task_id = "mock_wf_task"
    workflow_mock.return_value.is_ready = False

    trigger_mock = AsyncMock()
    trigger_mock.return_value.task_id = "mock_trigger_task"
    trigger_mock.return_value.is_ready = False

    with patch("seer.worker.tasks.workflows.workflow_execution_task.kiq", workflow_mock):
        with patch("seer.worker.tasks.triggers.trigger_event_task.kiq", trigger_mock):
            yield {
                "workflow_task_mock": workflow_mock,
                "trigger_task_mock": trigger_mock,
            }


__all__ = [
    "taskiq_direct_executor",
    "mock_task_execution",
    "TaskExecutionTracker",
]
