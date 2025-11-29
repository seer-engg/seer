
from langchain_neo4j import Neo4jGraph
from shared.config import config
from shared.logger import get_logger

logger = get_logger("graph_db.client")

INDEX_NAME = config.eval_reflections_index_name
NODE_LABEL = config.eval_reflections_node_label
EMBEDDING_PROPERTY = config.eval_reflections_embedding_property

# Client for factual graph operations (Cypher queries)
NEO4J_GRAPH = Neo4jGraph(
    url=config.neo4j_uri,
    username=config.neo4j_username,
    password=config.neo4j_password,
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
                `vector.dimensions`: {config.embedding_dims},
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
        CREATE VECTOR INDEX {config.tool_vector_index} IF NOT EXISTS
        FOR (t:{config.tool_node_label})
        ON (t.{config.tool_embed_prop})
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {config.embedding_dims},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
    )

ensure_tool_vector_index()
ensure_eval_reflections_setup()