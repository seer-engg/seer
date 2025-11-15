


from langchain_neo4j import Neo4jGraph
from .constants import NEO4J_URL, NEO4J_USER, NEO4J_PASS, INDEX_NAME, NODE_LABEL, EMBEDDING_PROPERTY, EMBEDDING_DIMS, TOOL_NODE_LABEL, TOOL_EMBED_PROP, TOOL_VECTOR_INDEX
from shared.logger import get_logger

logger = get_logger("graph_db.client")

# Client for factual graph operations (Cypher queries)
NEO4J_GRAPH = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USER,
    password=NEO4J_PASS,
)
logger.info(f"Successfully created Neo4j graph client.")

def ensure_eval_reflections_setup():
    """
    Ensure the eval reflections setup is complete.
    """
    # Create the vector index
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

    logger.info(f"Successfully created or verified Neo4j vector index '{INDEX_NAME}'.")

    SCORE_INDEX_NAME = "eval_reflection_score_index"
    NEO4J_GRAPH.query(
        f"""
        CREATE RANGE INDEX {SCORE_INDEX_NAME} IF NOT EXISTS
        FOR (n:{NODE_LABEL})
        ON (n.latest_score)
        """
    )
    logger.info(f"Successfully created or verified Neo4j range index '{SCORE_INDEX_NAME}'.")


def ensure_tool_vector_index():
    """
    Ensure the tool vector index is created.
    """
    NEO4J_GRAPH.query(
        f"""
        CREATE VECTOR INDEX {TOOL_VECTOR_INDEX} IF NOT EXISTS
        FOR (t:{TOOL_NODE_LABEL})
        ON (t.{TOOL_EMBED_PROP})
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {EMBEDDING_DIMS},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
    )

ensure_tool_vector_index()
ensure_eval_reflections_setup()