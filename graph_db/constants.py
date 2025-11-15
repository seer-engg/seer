import os

# Assumes Neo4j is running (e.g., free AuraDB instance)
NEO4J_URL = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")



EMBEDDING_DIMS = 1536 # From your langgraph.json
INDEX_NAME = "eval_reflections"
NODE_LABEL = "EvalReflection"
EMBEDDING_PROPERTY = "embedding"



TOOL_NODE_LABEL = "MCPTool"
TOOL_EMBED_PROP = "embedding"
TOOL_VECTOR_INDEX = "mcp_tools_index"