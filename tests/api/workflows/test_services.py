from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import sys


import pytest

from api.workflows import models as api_models
from api.workflows import services
from workflow_compiler.schema.models import WorkflowSpec
from shared.tools.google.gmail import GmailReadTool
from shared.tools import executor as tool_executor
from shared.database.models import User

tool = GmailReadTool()
TOOL_NAME = tool.name
email_schema = tool.get_output_schema()
MODEL_ID = "gpt-5-nano"
import logging

def pytest_configure():
    logging.getLogger("api.workflows.services").setLevel(logging.DEBUG)
    logging.getLogger("workflow_compiler.runtime").setLevel(logging.DEBUG)

from shared.database.config import TORTOISE_ORM
from tortoise import Tortoise

import asyncio

pytest_configure()

import pytest_asyncio

@pytest_asyncio.fixture(scope="session", autouse=True)
async def tortoise_db():
    await Tortoise.init(config=TORTOISE_ORM)
    yield
    # await Tortoise.close_connections()

# -----------------------------------------------------------------------------
# Global knobs users can override with real fixtures/specs later on.
# -----------------------------------------------------------------------------
TEST_WORKFLOW_SPEC_DATA: Dict[str, Any] = {
        "version": "1",
        "inputs": {
            "user_id": {"type": "integer", "required": True, "description": "Owner of the Gmail mailbox"},
        },
        "nodes": [
            {
                "id": "fetch_emails",
                "type": "tool",
                "tool": TOOL_NAME,
                "in": {
                    "user_id": "${inputs.user_id}",
                    "max_results": 3,
                    "label_ids": ["INBOX"],
                    "include_body": True,
                },
                "out": "emails",
                "expect_output": {
                    "mode": "json",
                    "schema": {"json_schema": email_schema},
                },
            },
            {
                "id": "summarize",
                "type": "llm",
                "model": MODEL_ID,
                "prompt": (
                    "Summarize the following emails for a busy engineer. "
                    "Group similar items and call out any requests or deadlines."
                ),
                "in": {"emails": "${emails}"},
                "out": "inbox_summary",
                "output": {"mode": "text"},
            },
        ],
        "output": "${inbox_summary}",
    }
TEST_USER_ID: int = 1
TEST_SCHEMA_ID = "schemas.IssueSummary@v1"

# -----------------------------------------------------------------------------
# Helper data classes & fixtures
# -----------------------------------------------------------------------------


class _InMemoryWorkflowRecord:
    _store: Dict[int, "_WorkflowRecordInstance"] = {}
    _pk: int = 0

    @classmethod
    async def create(cls, **kwargs):
        cls._pk += 1
        record = _WorkflowRecordInstance(id=cls._pk, **kwargs)
        cls._store[record.id] = record
        return record

    @classmethod
    def filter(cls, **kwargs):
        return _WorkflowRecordQuery(list(cls._store.values())).filter(**kwargs)

    @classmethod
    async def get(cls, **kwargs):
        for record in cls._store.values():
            if _matches(record, kwargs):
                return record
        raise services.DoesNotExist


class _WorkflowRecordQuery:
    def __init__(self, items: List["_WorkflowRecordInstance"]):
        self._items = list(items)

    def filter(self, **kwargs):
        filtered = [item for item in self._items if _matches(item, kwargs)]
        return _WorkflowRecordQuery(filtered)

    def order_by(self, field: str):
        reverse = field.startswith("-")
        key = field.lstrip("-")
        ordered = sorted(self._items, key=lambda item: getattr(item, key), reverse=reverse)
        return _WorkflowRecordQuery(ordered)

    async def limit(self, count: int):
        return self._items[:count]


class _WorkflowRecordInstance:
    def __init__(
        self,
        *,
        id: int,
        user,
        name: str,
        description: Optional[str],
        spec: Dict[str, Any],
        version: int,
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        now = datetime.now(timezone.utc)
        self.id = id
        self.user = user
        self.name = name
        self.description = description
        self.spec = spec
        self.version = version
        self.tags = tags or []
        self.meta = meta or {}
        self.last_compile_ok = False
        self.created_at = now
        self.updated_at = now

    @property
    def workflow_id(self) -> str:
        return f"wf_{self.id}"

    async def save(self):
        self.updated_at = datetime.now(timezone.utc)
        _InMemoryWorkflowRecord._store[self.id] = self

    async def delete(self):
        _InMemoryWorkflowRecord._store.pop(self.id, None)


class _InMemoryWorkflowRun:
    _store: Dict[int, "_WorkflowRunInstance"] = {}
    _pk: int = 0

    @classmethod
    async def create(cls, **kwargs):
        cls._pk += 1
        run = _WorkflowRunInstance(id=cls._pk, **kwargs)
        cls._store[run.id] = run
        return run

    @classmethod
    def filter(cls, **kwargs):
        return _WorkflowRunQuery(list(cls._store.values()), cls._store).filter(**kwargs)

    @classmethod
    async def get(cls, **kwargs):
        for run in cls._store.values():
            if _matches(run, kwargs):
                return run
        raise services.DoesNotExist


class _WorkflowRunQuery:
    def __init__(self, items, store):
        self._items = list(items)
        self._store = store
        self._criteria = {}

    def filter(self, **kwargs):
        merged = dict(self._criteria)
        merged.update(kwargs)
        filtered = [item for item in self._items if _matches(item, merged)]
        query = _WorkflowRunQuery(filtered, self._store)
        query._criteria = merged
        return query

    async def update(self, **kwargs):
        for item in self._items:
            for key, value in kwargs.items():
                setattr(item, key, value)
            self._store[item.id] = item


class _WorkflowRunInstance:
    def __init__(
        self,
        *,
        id: int,
        user,
        workflow,
        workflow_version,
        spec,
        inputs,
        config,
        status,
    ):
        now = datetime.now(timezone.utc)
        self.id = id
        self.user = user
        self.workflow = workflow
        self.workflow_version = workflow_version
        self.spec = spec
        self.inputs = inputs
        self.config = config
        self.status = status
        self.output = None
        self.error = None
        self.metrics = None
        self.created_at = now
        self.started_at = None
        self.finished_at = None

    @property
    def run_id(self) -> str:
        return f"run_{self.id}"

    async def fetch_from_db(self):
        stored = _InMemoryWorkflowRun._store[self.id]
        self.__dict__.update(stored.__dict__)


def _matches(obj, criteria: Dict[str, Any]) -> bool:
    for key, value in criteria.items():
        if getattr(obj, key) != value:
            return False
    return True


@pytest.fixture
def workflow_spec() -> WorkflowSpec:
    if not TEST_WORKFLOW_SPEC_DATA:
        pytest.skip("Provide TEST_WORKFLOW_SPEC_DATA")
    return WorkflowSpec.model_validate(deepcopy(TEST_WORKFLOW_SPEC_DATA))


@pytest.fixture
async def db_user():
    if TEST_USER_ID is None:
        pytest.skip("Provide TEST_USER_ID")
    user = await User.get(id=TEST_USER_ID)
    return user


@pytest.fixture(autouse=True)
def setup_in_memory_services(monkeypatch):
    monkeypatch.setattr(services, "WorkflowRecord", _InMemoryWorkflowRecord)
    monkeypatch.setattr(services, "WorkflowRun", _InMemoryWorkflowRun)

    services.compiler.schema_registry.register(
        TEST_SCHEMA_ID,
        {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["title", "summary"],
        },
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workflow_crud_validate_compile_flow(db_user, workflow_spec):
    create_payload = api_models.WorkflowCreateRequest(
        name="Issue triage",
        description="Filters repo issues",
        tags=["github"],
        spec=workflow_spec,
    )
    created = await services.create_workflow(db_user, create_payload)
    assert created.workflow_id.startswith("wf_")
    assert created.version == 1

    list_resp = await services.list_workflows(db_user, limit=10)
    assert len(list_resp.items) == 1

    update_payload = api_models.WorkflowUpdateRequest(
        name="Updated issue triage",
        spec=workflow_spec,
    )
    updated = await services.update_workflow(db_user, created.workflow_id, update_payload)
    assert updated.name == "Updated issue triage"
    assert updated.version == 2

    validate_resp = services.validate_spec(api_models.ValidateRequest(spec=workflow_spec))
    assert validate_resp.ok

    compile_payload = api_models.CompileRequest(
        spec=workflow_spec,
        options=api_models.CompileOptions(emit_graph_preview=True, emit_type_env=True),
    )
    compile_resp = services.compile_spec(db_user, compile_payload)
    assert compile_resp.ok
    assert compile_resp.artifacts.type_env is not None
    assert compile_resp.artifacts.graph_preview is not None

    await services.delete_workflow(db_user, created.workflow_id)
    post_delete = await services.list_workflows(db_user)
    assert post_delete.items == []


@pytest.mark.asyncio
async def test_run_lifecycle_from_saved_and_draft(db_user, workflow_spec):
    created = await services.create_workflow(
        db_user,
        api_models.WorkflowCreateRequest(
            name="Runnable workflow",
            description=None,
            spec=workflow_spec,
            tags=[],
        ),
    )

    saved_run = await services.run_saved_workflow(
        db_user,
        created.workflow_id,
        api_models.RunFromWorkflowRequest(inputs={"user_id": TEST_USER_ID}),
    )
    assert saved_run.status == services.WorkflowRunStatus.SUCCEEDED.value

    status = await services.get_run_status(db_user, saved_run.run_id)
    assert status.run_id == saved_run.run_id

    result = await services.get_run_result(db_user, saved_run.run_id)
    assert result.status == services.WorkflowRunStatus.SUCCEEDED.value

    draft_run = await services.run_draft_workflow(
        db_user,
        api_models.RunFromSpecRequest(spec=workflow_spec, inputs={"user_id": TEST_USER_ID}),
    )
    assert draft_run.status == services.WorkflowRunStatus.SUCCEEDED.value


@pytest.mark.asyncio
async def test_expression_helpers(db_user, workflow_spec):
    suggest_request = api_models.ExpressionSuggestRequest(
        spec=workflow_spec,
        cursor_context=api_models.ExpressionCursorContext(
            node_id="set_issue_title",
            field="prompt",
            prefix="${inputs.",
        ),
    )
    suggestions = services.suggest_expression(db_user, suggest_request)
    assert any(s.label == "user_id" for s in suggestions.suggestions)

    typecheck_request = api_models.ExpressionTypecheckRequest(
        spec=workflow_spec,
        expression="${inputs.user_id}",
    )
    type_resp = services.typecheck_expression(db_user, typecheck_request)
    assert type_resp.ok
    assert type_resp.type == {"type": "integer"}


@pytest.mark.asyncio
async def test_registry_helpers(db_user):
    node_types = await services.list_node_types()
    assert node_types.node_types

    tools_resp = await services.list_tools(include_schemas=True)
    assert tools_resp.tools and tools_resp.tools[0].input_schema is not None

    models_resp = await services.list_models()
    assert models_resp.models

    schema_resp = await services.resolve_schema(TEST_SCHEMA_ID)
    assert schema_resp.json_schema["type"] == "object"


# asyncio.run(close_tortoise())