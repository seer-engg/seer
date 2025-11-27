"""module for retrieving eval reflections"""
import json
import asyncio
from typing import Any, List, Dict

from shared.logger import get_logger
from langchain_openai import OpenAIEmbeddings
from shared.config import EVAL_PASS_THRESHOLD, OPENAI_API_KEY
from graph_db import NEO4J_GRAPH

logger = get_logger("eval_agent.reflection_store")

_embeddings_client = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


async def _find_relevant_reflections(embedding: List[float], agent_name: str, user_id: str, limit: int) -> List[Dict[str, Any]]:
    """
    Step 2: Find matching Memory nodes using vector search and metadata filters.
    """
    
    logger.info(f"Step 2: Searching vector index with params: user_id='{user_id}', agent_name='{agent_name}', pass_threshold<{EVAL_PASS_THRESHOLD}")

    vector_search_query = """
    MATCH (node:Memory)
    WHERE node.user_id = $user_id 
      AND node.agent_id = $agent_name 
      AND node.score < $pass_threshold
      AND node.embedding IS NOT NULL
      AND node.context = 'eval_agent.reflection'
    WITH node, vector.similarity.cosine(node.embedding, $embedding) AS score
    RETURN 
        node.memory_id as reflection_id, 
        node.observation as reflection_summary, 
        node.metadata as metadata,
        score
    ORDER BY score DESC
    LIMIT $limit
    """

    try:
        results = await asyncio.to_thread(
            NEO4J_GRAPH.query,
            vector_search_query,
            params={
                "embedding": embedding,
                "limit": limit,
                "user_id": user_id,
                "agent_name": agent_name,
                "pass_threshold": EVAL_PASS_THRESHOLD
            }
        )
        logger.info(f"Step 2: Found {len(results)} candidate reflections from vector search.")
        
        # Parse metadata to extract critique
        processed_results = []
        for r in results:
            critique = None
            if r.get("metadata"):
                try:
                    meta = json.loads(r["metadata"])
                    critique = meta.get("test_generation_critique")
                except:
                    pass
            
            processed_results.append({
                "reflection_id": r["reflection_id"],
                "reflection_summary": r["reflection_summary"],
                "test_generation_critique": critique,
                "score": r["score"]
            })
            
        return processed_results
    except Exception as e:
        logger.error(f"Step 2: Vector search query failed: {e}")
        return []


async def _get_evidence_for_reflection(reflection_id: str, user_id: str) -> List[Dict[str, Any]]:
    """
    Step 3: Perform graph traversal to find evidence for a specific reflection.
    """
    graph_traversal_query = """
    MATCH (node:Memory {memory_id: $reflection_id, user_id: $user_id})
    MATCH (node)-[:GENERATED_FROM]->(res:ExperimentResult {passed: false, user_id: $user_id})
    MATCH (ex:DatasetExample {user_id: $user_id})-[:WAS_RUN_IN]->(res)
    RETURN 
        collect({
            input: ex.input_message,
            actual: res.actual_output,
            reasoning: res.judge_reasoning,
            score: res.score
        }) as evidence
    """
    try:
        results = await asyncio.to_thread(
            NEO4J_GRAPH.query,
            graph_traversal_query,
            params={"reflection_id": reflection_id, "user_id": user_id}
        )
        if results and results[0]:
            return results[0].get("evidence", [])
        return []
    except Exception as e:
        logger.error(f"Step 3: Graph traversal query for reflection {reflection_id} failed: {e}")
        return []


async def graph_rag_retrieval(query: str, agent_name: str, user_id: str, limit: int = 3) -> str:
    """
    Perform GraphRAG in distinct steps for easier debugging.
    1. Vector search for relevant "reflections" (hypotheses).
    2. Graph search to find all "failing tests" (evidence) linked to them.
    """
    
    # --- Step 1: Get Query Embedding ---
    try:
        embedding = await _embeddings_client.aembed_query(query)
        logger.info("Step 1: Successfully generated query embedding.")
    except Exception as e:
        logger.error(f"Step 1: Failed to embed query for RAG: {e}")
        return "Failed to process query."

    # --- Step 2: Find Matching Reflections (Vector Search) ---
    # This query just finds the reflection nodes, not their evidence.
    candidate_reflections = await _find_relevant_reflections(
        embedding=embedding,
        agent_name=agent_name,
        user_id=user_id,
        limit=limit
    )
    
    if not candidate_reflections:
        # This is not an error, just no relevant history.
        logger.info(f"Step 2: Vector search found no relevant reflections for user {user_id}, agent {agent_name}.")
        return "No past reflections found."

    # --- Step 3: Find Evidence (Graph Traversal) ---
    rag_context_parts = []
    found_evidence_count = 0
    
    for reflection in candidate_reflections:
        reflection_id = reflection.get("reflection_id")
        summary = reflection.get("reflection_summary")
        critique = reflection.get("test_generation_critique")
        
        if not reflection_id:
            logger.warning("Found reflection node with no ID, skipping.")
            continue
            
        # For each reflection, run a separate query to find its evidence
        evidence_list = await _get_evidence_for_reflection(reflection_id, user_id)
        
        if evidence_list:
            # Only add reflections that are successfully linked to failed evidence
            found_evidence_count += 1
            rag_context_parts.append(f"Insight: {summary}")
            
            if critique:
                rag_context_parts.append(f"Past Test Critique: {critique}")
                
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
        else:
            logger.info(f"Step 3: Reflection {reflection_id} had no verifiable failed evidence, skipping.")

    if not rag_context_parts:
        logger.info(f"Step 3: No reflections had verifiable failed evidence.")
        return "No relevant reflections with evidence found."
    
    logger.info(f"Successfully built RAG context with {found_evidence_count} reflections.")
    return "\n".join(rag_context_parts)


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
        CALL db.index.vector.queryNodes("memory_embeddings", 1, $embedding) YIELD node, score
        
        // Combine all filters into a single WHERE clause
        WHERE node.user_id = $user_id 
          AND node.agent_id = $agent_name
          AND node.context = 'eval_agent.reflection'
        
        RETURN node.metadata as metadata, score
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
        
        metadata_str = result[0].get("metadata")
        critique = ""
        if metadata_str:
            try:
                meta = json.loads(metadata_str)
                critique = meta.get("test_generation_critique", "")
            except:
                pass

        if critique:
            logger.info(f"reflection_store.get_latest_critique: Found critique: {critique[:50]}...")
            return critique
        return ""
        
    except Exception as e:
        logger.warning(f"reflection_store.get_latest_critique: Error retrieving critique: {e}")
        return ""
