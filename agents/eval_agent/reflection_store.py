"""module for retrieving eval reflections"""
from typing import Any
import asyncio

from shared.logger import get_logger
from agents.eval_agent.constants import NEO4J_GRAPH, NEO4J_VECTOR

logger = get_logger("eval_agent.reflection_store")


async def graph_rag_retrieval(query: str, agent_name: str, user_id: str, limit: int = 3) -> str:
    """
    Perform GraphRAG:
    1. Vector search for a relevant "reflection" (hypothesis).
    2. Graph search to find all "failing tests" (evidence) linked to it.
    """
    
    # 1. Vector Search (Find the "Idea")
    similar_reflections = await NEO4J_VECTOR.asimilarity_search_with_score(
        query,
        k=limit,
        filters={"agent_name": agent_name, "user_id": user_id}
    )
    if not similar_reflections:
        logger.warning(f"reflection_store: No similar reflections found for user {user_id}.")
        return "No past reflections found."

    # Get the list of reflection IDs from the vector search
    reflection_ids_to_query = [
        doc.metadata.get("reflection_id") for doc, score in similar_reflections 
        if doc.metadata.get("reflection_id")
    ]
    
    if not reflection_ids_to_query:
        logger.warning("reflection_store: Found reflections but they were missing metadata IDs.")
        return "No past reflections found."

    rag_context_parts = []

    # 2. Graph Retrieval (Find the "Proof")
    cypher_query = """
    UNWIND $reflection_ids as ref_id
    MATCH (ref:EvalReflection {reflection_id: ref_id})
    
    // Ensure we only get reflections linked to actual failures
    MATCH (ref)-[:GENERATED_FROM]->(res:ExperimentResult {passed: false})
    
    MATCH (ex:DatasetExample)-[:WAS_RUN_IN]->(res)
    
    RETURN 
        ref.summary as reflection_summary, 
        collect({
            input: ex.input_message,
            actual: res.actual_output,
            reasoning: res.judge_reasoning,
            score: res.score
        }) as evidence
    """
    
    graph_result = await asyncio.to_thread(
        NEO4J_GRAPH.query,
        cypher_query, 
        params={"reflection_ids": reflection_ids_to_query}
    )
    
    if not graph_result:
        logger.warning(f"reflection_store: Found reflections, but none were linked to FAILED tests.")
        return "Found old reflections, but no new failing evidence."

    # 3. Format for Prompt
    for data in graph_result:
        summary = data.get("reflection_summary")
        evidence_list = data.get("evidence", [])
        
        rag_context_parts.append(f"Insight: {summary}")
        if evidence_list:
            rag_context_parts.append("Supporting Failed Tests:")
            for ev in evidence_list:
                if ev and ev.get('input'): 
                    rag_context_parts.append(
                        f"  - Input: {_truncate(ev.get('input'))}\n"
                        f"    Actual: {_truncate(ev.get('actual'))}\n"
                        f"    Reasoning: {_truncate(ev.get('reasoning'))}\n"
                        f"    Score: {ev.get('score') or 0.0:.2f}"
                    )
        rag_context_parts.append("-" * 20)
            
    return "\n".join(rag_context_parts) if rag_context_parts else "No relevant reflections with evidence found."


def _truncate(text: Any, limit: int = 280) -> str:
    """Helper to keep prompt context small."""
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… (truncated {len(text)} chars)"
