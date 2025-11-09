"""constants for the eval agent"""
import os

from langgraph_sdk import get_client, get_sync_client
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langsmith import Client
from shared.llm import get_llm


# Assumes Neo4j is running (e.g., free AuraDB instance)
NEO4J_URL = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Client for factual graph operations (Cypher queries)
NEO4J_GRAPH = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USER,
    password=NEO4J_PASS,
)

EMBEDDING_DIMS = 1536 # From your langgraph.json
INDEX_NAME = "eval_reflections"
NODE_LABEL = "EvalReflection"
EMBEDDING_PROPERTY = "embedding"

NEO4J_GRAPH.query(
    f"""
    CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
    FOR (n:{NODE_LABEL})
    ON (n.{EMBEDDING_PROPERTY})
    OPTIONS {{ 
        indexConfig: {{
            `vector.dimensions`: {EMBEDDING_DIMS},
            `vector.similarity_function`: 'cosine'
        }}
    }}
    """
)
print(f"Successfully created or verified Neo4j vector index '{INDEX_NAME}'.")

SCORE_INDEX_NAME = "eval_reflection_score_index"
NEO4J_GRAPH.query(
    f"""
    CREATE RANGE INDEX {SCORE_INDEX_NAME} IF NOT EXISTS
    FOR (n:{NODE_LABEL})
    ON (n.latest_score)
    """
)
print(f"Successfully created or verified Neo4j range index '{SCORE_INDEX_NAME}'.")

# Local LangGraph store used for eval reflections and metadata persistence
LANGGRAPH_BASE_URL = os.getenv("LANGGRAPH_BASE_URL", "http://127.0.0.1:8002")
LANGGRAPH_CLIENT = get_client(url=LANGGRAPH_BASE_URL)
LANGGRAPH_SYNC_CLIENT = get_sync_client(url=LANGGRAPH_BASE_URL)


# LangSmith client for experiment uploads and remote graph execution
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_CLIENT = Client(api_key=LANGSMITH_API_KEY)


# Evaluation loop thresholds
N_ROUNDS = 5
N_TEST_CASES = 3 # Number of test cases to generate in 1 round of eval
N_VERSIONS = 2 # total versions of the target agent
EVAL_PASS_THRESHOLD = 0.8


CODEX_REMOTE_URL = os.getenv("CODEX_REMOTE_URL", "http://127.0.0.1:8003")
EVAL_REMOTE_URL = os.getenv("EVAL_REMOTE_URL", "http://127.0.0.1:8002")

# Shared LLM handles
LLM = get_llm(temperature=0.0)
