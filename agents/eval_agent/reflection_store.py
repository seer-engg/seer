"""module for storing and retrieving eval reflections"""
from typing import Any, List
import asyncio

from langchain_core.documents import Document
from shared.logger import get_logger
from agents.eval_agent.constants import NEO4J_GRAPH, NEO4J_VECTOR
from agents.eval_agent.models import EvalReflection
from shared.schema import ExperimentResultContext

logger = get_logger("eval_agent.reflection_store")


def persist_reflection(
    agent_name: str, 
    reflection: EvalReflection, 
    evidence_results: List[ExperimentResultContext]
) -> None:
    """
    Store the reflection in both the vector index (for search)
    and the graph (for context).
    """
    
    # 1. Store the reflection summary in the vector index
    # We store the reflection's unique ID in the metadata
    # so we can find it in the graph later.
    doc = Document(
        page_content=reflection.summary,
        metadata={
            "reflection_id": reflection.reflection_id,
            "agent_name": agent_name,
            "latest_score": reflection.latest_score,
            "attempt": reflection.attempt,
        }
    )
    NEO4J_VECTOR.add_documents([doc])
    logger.info(f"reflection_store: Stored reflection {reflection.reflection_id} in vector index.")
        
    # 2. Store the reflection node and link it to its evidence
    # Get thread_ids from all evidence results, pass as parameter
    evidence_thread_ids = [res.thread_id for res in evidence_results]
    
    cypher_query = """
    // 1. Find the new EvalReflection node (created by the vector store)
    // or create it if it wasn't vectorized.
    MERGE (ref:EvalReflection {reflection_id: $reflection.reflection_id})
    ON CREATE SET ref = $reflection
    
    // 2. Unwind the list of thread IDs for the evidence
    WITH ref
    UNWIND $evidence_thread_ids AS thread_id
    
    // 3. Match the ExperimentResult nodes (the evidence)
    MATCH (res:ExperimentResult {thread_id: thread_id})
    
    // 4. Create the link (Hypothesis)-[GENERATED_FROM]->(Evidence)
    MERGE (ref)-[r:GENERATED_FROM]->(res)
    
    RETURN count(r) as links_created
    """
    
    NEO4J_GRAPH.query(
        cypher_query,
        params={
            "reflection": reflection.model_dump(),
            "evidence_thread_ids": evidence_thread_ids
        }
    )
    logger.info(f"reflection_store: Linked reflection {reflection.reflection_id} to its evidence in graph.")
    

async def graph_rag_retrieval(query: str, agent_name: str, limit: int = 3) -> str:
    """
    Perform GraphRAG:
    1. Vector search for a relevant "reflection" (hypothesis).
    2. Graph search to find all "failing tests" (evidence) linked to it.
    """
    
    # 1. Vector Search (Find the "Idea")
    similar_reflections = await NEO4J_VECTOR.asimilarity_search_with_score(
        query,
        k=limit,
        filters={"agent_name": agent_name} # Filter by agent
    )
    if not similar_reflections:
        logger.warning("reflection_store: No similar reflections found in vector index.")
        return "No past reflections found."

    rag_context_parts = []

    # 2. Graph Retrieval (Find the "Proof")
    for doc, score in similar_reflections:
        reflection_id = doc.metadata.get("reflection_id")
        if not reflection_id:
            continue
            
        # This query finds the reflection and all linked failing tests
        cypher_query = """
        MATCH (ref:EvalReflection {reflection_id: $reflection_id})
        
        // Find all results that generated this reflection
        OPTIONAL MATCH (ref)-[:GENERATED_FROM]->(res:ExperimentResult)
        
        // Find the test case for that result
        OPTIONAL MATCH (ex:DatasetExample)-[:WAS_RUN_IN]->(res)
        
        // Filter for *only* the failed tests as evidence
        WHERE res.passed = false
        
        RETURN 
            ref.summary as reflection_summary,
            collect({
                input: ex.input_message,
                actual: res.actual_output,
                reasoning: res.judge_reasoning,
                score: res.score
            }) as evidence
        LIMIT 1 // We only care about the single reflection
        """
        
        graph_result = await asyncio.to_thread(
            NEO4J_GRAPH.query,
            cypher_query, 
            params={"reflection_id": reflection_id}
        )
        
        if not graph_result:
            continue
            
        data = graph_result[0]
        summary = data.get("reflection_summary")
        evidence_list = data.get("evidence", [])
        
        # 3. Format for Prompt
        rag_context_parts.append(f"Insight (Score: {score:.2f}): {summary}")
        if evidence_list:
            rag_context_parts.append("Supporting Evidence (Failed Tests):")
            for ev in evidence_list:
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
