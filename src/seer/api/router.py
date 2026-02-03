"""API router for the Seer Agents API."""
from fastapi import APIRouter

from .agents.traces import router as agents_traces_router
from .agents.workflow.router import router as nexus_router
from .forms.router import router as forms_router
from .integrations.router import router as integrations_router
from .knowledge.router import router as knowledge_router
from .models.router import router as models_router
from .subscriptions.router import router as subscriptions_router
from .subscriptions.setup_intent import router as setup_intent_router
from .usage.router import router as usage_router
from .users.settings import router as user_settings_router
from .webhooks.router import router as webhooks_router
from .workflows.router import router as workflows_router

router = APIRouter(prefix="/api")
router.include_router(agents_traces_router)
router.include_router(integrations_router)
router.include_router(knowledge_router)
router.include_router(models_router)
router.include_router(subscriptions_router)
router.include_router(setup_intent_router)
router.include_router(workflows_router)
router.include_router(nexus_router)
router.include_router(webhooks_router)
router.include_router(forms_router)
router.include_router(usage_router)
router.include_router(user_settings_router)
