"""API router for the Seer Agents API."""
from fastapi import APIRouter

from .agents.workflow.router import router as nexus_router
from .forms.router import router as forms_router
from .integrations.router import router as integrations_router
from .knowledge.router import router as knowledge_router
from .models.router import router as models_router
from .subscriptions.router import router as subscriptions_router
from .subscriptions.setup_intent import router as setup_intent_router
from .usage.router import router as usage_router
from .users.profile import router as user_profile_router
from .users.settings import router as user_settings_router
from .webhooks.router import router as webhooks_router
from .workflows.router import router as workflows_router
from .templates.router import router as templates_router
from .browser.router import router as browser_router
from .browser.ws_router import router as browser_session_router
from .browser.recording_router import router as recording_router
from .files.router import router as files_router
from .memory.router import router as memory_router
from .organizations.router import router as organizations_router
from .overage.router import router as overage_router
from .chat.router import router as chat_router
from .collaboration.router import router as collaboration_router
from .public.router import router as public_router
from .dev.router import router as dev_router
from .email_analytics.router import router as email_analytics_router
from .meetings.router import router as meetings_router

router = APIRouter(prefix="/api")
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
router.include_router(user_profile_router)
router.include_router(user_settings_router)
router.include_router(templates_router)
router.include_router(browser_router)
router.include_router(browser_session_router)
router.include_router(recording_router)
router.include_router(files_router)
router.include_router(memory_router)
router.include_router(organizations_router)
router.include_router(overage_router)
router.include_router(chat_router)
router.include_router(collaboration_router)
router.include_router(public_router)
router.include_router(dev_router)
router.include_router(email_analytics_router)
router.include_router(meetings_router)
