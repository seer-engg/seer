"""constants for the eval agent"""
import os

from langgraph_sdk import get_client, get_sync_client
from langsmith import Client
from shared.llm import get_llm

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Local LangGraph store used for eval reflections and metadata persistence
LANGGRAPH_BASE_URL = os.getenv("LANGGRAPH_BASE_URL", "http://127.0.0.1:8002")
LANGGRAPH_CLIENT = get_client(url=LANGGRAPH_BASE_URL)
LANGGRAPH_SYNC_CLIENT = get_sync_client(url=LANGGRAPH_BASE_URL)


# LangSmith client for experiment uploads and remote graph execution
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_CLIENT = Client(api_key=LANGSMITH_API_KEY)


# Evaluation loop thresholds
N_ROUNDS = 2
N_TEST_CASES = 1 # Number of test cases to generate in 1 round of eval
N_VERSIONS = 2 # total versions of the target agent
EVAL_PASS_THRESHOLD = 0.8


CODEX_REMOTE_URL = os.getenv("CODEX_REMOTE_URL", "http://127.0.0.1:8003")
EVAL_REMOTE_URL = os.getenv("EVAL_REMOTE_URL", "http://127.0.0.1:8002")

# Shared LLM handles
LLM = get_llm(temperature=0.0)
