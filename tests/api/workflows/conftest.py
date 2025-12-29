import asyncio
from copy import deepcopy

import pytest
import pytest_asyncio
from tortoise import Tortoise
from workflow_compiler.schema.models import WorkflowSpec

from api.workflows import services
from shared.database.config import TORTOISE_ORM
from shared.database.models import User
from shared.database.workflow_models import WorkflowRecord, WorkflowRun
from tests.api.workflows.shared_data import (
    TEST_SCHEMA_DEFINITION,
    TEST_SCHEMA_ID,
    TEST_USER_ID,
    TEST_WORKFLOW_SPEC_DATA,
    configure_workflow_test_logging,
)


configure_workflow_test_logging()


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def tortoise_db():
    await Tortoise.init(config=TORTOISE_ORM)
    yield


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


@pytest.fixture(scope="session", autouse=True)
def register_test_schema():
    services.compiler.schema_registry.register(TEST_SCHEMA_ID, TEST_SCHEMA_DEFINITION)


@pytest_asyncio.fixture(autouse=True)
async def cleanup_workflow_models():
    await WorkflowRun.all().delete()
    await WorkflowRecord.all().delete()
