"""module for retrieving eval reflections"""
import asyncio
from typing import Any

from shared.logger import get_logger
from langchain_openai import OpenAIEmbeddings
from agents.eval_agent.constants import (
    NEO4J_GRAPH, 
    EVAL_PASS_THRESHOLD, 
    OPENAI_API_KEY
)

logger = get_logger("eval_agent.reflection_store")

_embeddings_client = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


async def graph_rag_retrieval(query: str, agent_name: str, user_id: str, limit: int = 3) -> str:
    """
    Perform GraphRAG using a single, manual Cypher query to bypass library abstractions.
    1. Vector search for relevant "reflections" (hypotheses) that are linked to failures.
    2. Graph search to find all "failing tests" (evidence) linked to them.
    """
    
    # 1. Manually get the embedding
    try:
        embedding = await _embeddings_client.aembed_query(query)
    except Exception as e:
        logger.error(f"Failed to embed query for RAG: {e}")
        return "Failed to process query."

    # 2. Run the single, combined Cypher query
    cypher_query = """
    // 2a. Call the vector index and apply metadata filters
    CALL db.index.vector.queryNodes("eval_reflections", $limit, $embedding) YIELD node, score
    WHERE node.user_id = $user_id 
      AND node.agent_name = $agent_name 
      AND node.latest_score < $pass_threshold
    
    // 2b. Verify it's linked to an *actual failure*
    MATCH (node)-[:GENERATED_FROM]->(res:ExperimentResult {passed: false, user_id: $user_id})
    
    // 2c. Get the evidence for that failure
    MATCH (ex:DatasetExample {user_id: $user_id})-[:WAS_RUN_IN]->(res)
    
    // 2d. Return the formatted data
    RETURN 
        node.summary as reflection_summary,
        node.test_generation_critique as test_generation_critique,
        node.judge_critique as judge_critique,
        collect({
            input: ex.input_message,
            actual: res.actual_output,
            reasoning: res.judge_reasoning,
            score: res.score
        }) as evidence,
        score // Return score for ordering
    ORDER BY score DESC
    """
    
    try:
        graph_result = await asyncio.to_thread(
            NEO4J_GRAPH.query,
            cypher_query, 
            params={
                "embedding": embedding,
                "limit": limit,
                "user_id": user_id,
                "agent_name": agent_name,
                "pass_threshold": EVAL_PASS_THRESHOLD
            }
        )
    except Exception as e:
        logger.error(f"Manual graph_rag_retrieval query failed: {e}")
        return "Error retrieving reflections."
    
    if not graph_result:
        # This is the warning you were seeing!
        # It will now only fire if no reflections *that are linked to failures* are found.
        logger.info(f"reflection_store: Vector search found no reflections linked to FAILED tests.")
        return "No past reflections found."

    # 3. Format for Prompt (same as before)
    rag_context_parts = []
    for data in graph_result:
        summary = data.get("reflection_summary")
        critique = data.get("test_generation_critique")
        judge_critique = data.get("judge_critique")
        evidence_list = data.get("evidence", [])
        
        rag_context_parts.append(f"Insight: {summary}")
        
        if critique:
            rag_context_parts.append(f"Past Test Critique: {critique}")
            
        if judge_critique:
            rag_context_parts.append(f"Past Judge Critique: {judge_critique}")
            
        if evidence_list:
            rag_context_parts.append("Supporting Failed Tests:")
            for ev in evidence_list:
                if ev and ev.get('input'): 
                    rag_context_parts.append(
                        f"  - Input: {ev.get('input')}\n"
                        f"    Actual: {ev.get('actual')}\n"
                        f"    Reasoning: {ev.get('reasoning')}\n"
                        f"    Score: {ev.get('score') or 0.0:.2f}"
                    )
        rag_context_parts.append("-" * 20)
            
    return "\n".join(rag_context_parts) if rag_context_parts else "No relevant reflections with evidence found."


async def get_latest_critique(query: str, agent_name: str, user_id: str) -> str:
    """
    Perform a simple vector search to get the most recent critique.
    This does *not* require a link to a failed test.
    """
    try:
        # 1. Manually get the embedding
        embedding = await _embeddings_client.aembed_query(query)
        
        # 2. Run the manual Cypher query
        cypher_query = """
        CALL db.index.vector.queryNodes("eval_reflections", 1, $embedding) YIELD node, score
        
        // --- THIS IS THE FIX ---
        // Combine all filters into a single WHERE clause
        WHERE node.user_id = $user_id 
          AND node.agent_name = $agent_name
          AND node.judge_critique IS NOT NULL
        // --- END OF FIX ---
        
        RETURN node.judge_critique AS critique, score
        ORDER BY score DESC
        LIMIT 1
        """
        
        result = await asyncio.to_thread(
            NEO4J_GRAPH.query,
            cypher_query,
            params={
                "embedding": embedding,
                "user_id": user_id,
                "agent_name": agent_name
            }
        )
        
        if not result:
            logger.info(f"reflection_store.get_latest_critique: No critiques found.")
            return ""
        
        critique = result[0].get("critique")
        if critique:
            logger.info(f"reflection_store.get_latest_critique: Found critique: {critique[:50]}...")
            return critique
        return ""
        
    except Exception as e:
        logger.warning(f"reflection_store.get_latest_critique: Error retrieving critique: {e}")
        return ""
