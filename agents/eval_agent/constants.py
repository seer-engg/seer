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

# --- NEW: EXPLICITLY CREATE THE VECTOR INDEX ---
# This query is idempotent (IF NOT EXISTS) and ensures the index is ready
# before the vector client tries to connect to it.

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

# --- END NEW INDEX CREATION ---

# Client for vector search (your "reflection" store)
# This will create and manage a vector index inside Neo4j
NEO4J_VECTOR = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
    url=NEO4J_URL,
    username=NEO4J_USER,
    password=NEO4J_PASS,
    index_name="eval_reflections",      # Name of the vector index
    node_label="EvalReflection",        # Graph node label for reflections
    text_node_property="summary",       # Which property stores the text
    embedding_node_property="embedding", # Which property stores the vector
)
# --- END NEW NEO4J CONFIG ---

# Local LangGraph store used for eval reflections and metadata persistence
LANGGRAPH_BASE_URL = os.getenv("LANGGRAPH_BASE_URL", "http://127.0.0.1:8002")
LANGGRAPH_CLIENT = get_client(url=LANGGRAPH_BASE_URL)
LANGGRAPH_SYNC_CLIENT = get_sync_client(url=LANGGRAPH_BASE_URL)


# LangSmith client for experiment uploads and remote graph execution
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_CLIENT = Client(api_key=LANGSMITH_API_KEY)


# Evaluation loop thresholds
N_ROUNDS = 3
N_TEST_CASES = 3 # Number of test cases to generate in 1 round of eval


# Remote Codex planner URL for follow-up remediation workflows
CODEX_REMOTE_URL = os.getenv("CODEX_REMOTE_URL", "http://127.0.0.1:8003")

# Shared LLM handles
LLM = get_llm(temperature=0.0)
