


from langchain_neo4j import Neo4jGraph

from shared.logger import get_logger
from shared.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, EMBEDDING_DIMS, EVAL_REFLECTIONS_INDEX_NAME, EVAL_REFLECTIONS_NODE_LABEL, EVAL_REFLECTIONS_EMBEDDING_PROPERTY, TOOL_NODE_LABEL, TOOL_EMBED_PROP, TOOL_VECTOR_INDEX

logger = get_logger("graph_db.client")

# Client for factual graph operations (Cypher queries)
NEO4J_GRAPH = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)
logger.info(f"Successfully created Neo4j graph client.")

def ensure_eval_reflections_setup():
    """
    Ensure the eval reflections setup is complete.
    """
    # Create the vector index
    NEO4J_GRAPH.query(
        f"""
        CREATE VECTOR INDEX {EVAL_REFLECTIONS_INDEX_NAME} IF NOT EXISTS
        FOR (n:{EVAL_REFLECTIONS_NODE_LABEL})
        ON (n.{EVAL_REFLECTIONS_EMBEDDING_PROPERTY})
        OPTIONS {{ 
            indexConfig: {{
                `vector.dimensions`: {EMBEDDING_DIMS},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
    )

    logger.info(f"Successfully created or verified Neo4j vector index '{EVAL_REFLECTIONS_INDEX_NAME}'.")

    SCORE_INDEX_NAME = "eval_reflection_score_index"
    NEO4J_GRAPH.query(
        f"""
        CREATE RANGE INDEX {SCORE_INDEX_NAME} IF NOT EXISTS
        FOR (n:{EVAL_REFLECTIONS_NODE_LABEL})
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