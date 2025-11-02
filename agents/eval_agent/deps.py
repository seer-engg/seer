import os

from langgraph_sdk import get_client, get_sync_client
from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

from shared.logger import get_logger
from shared.llm import get_llm


logger = get_logger("eval_agent")


# Local LangGraph store used for eval reflections and metadata persistence
LANGGRAPH_BASE_URL = os.getenv("LANGGRAPH_BASE_URL", "http://127.0.0.1:8002")
LANGGRAPH_CLIENT = get_client(url=LANGGRAPH_BASE_URL)
LANGGRAPH_SYNC_CLIENT = get_sync_client(url=LANGGRAPH_BASE_URL)


# LangSmith client for experiment uploads and remote graph execution
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_CLIENT = Client(api_key=LANGSMITH_API_KEY)


# Evaluation loop thresholds
PASS_THRESHOLD = 0.99
MAX_ATTEMPTS = 3
MIN_ATTEMPTS = 3


# Remote Codex planner URL for follow-up remediation workflows
CODEX_REMOTE_URL = os.getenv("CODEX_REMOTE_URL", "http://127.0.0.1:8003")


# Shared LLM handles
LLM = get_llm(temperature=0.2)
CORRECTNESS_EVALUATOR = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    model="openai:gpt-4.1-mini",
    feedback_key="correctness",
)


