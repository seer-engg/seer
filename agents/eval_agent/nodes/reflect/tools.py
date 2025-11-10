"""Tools for the reflection agent inside the Eval Agent."""
import asyncio
import json
from typing import List, Any
from pydantic import BaseModel
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain_openai import OpenAIEmbeddings 

from agents.eval_agent.constants import NEO4J_GRAPH, OPENAI_API_KEY
from agents.eval_agent.models import EvalReflection, Hypothesis
from shared.schema import ExperimentResultContext
from shared.logger import get_logger


_embeddings_client = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
logger = get_logger("eval_agent.reflection_tools")


class ReflectionToolContext(BaseModel):
    """Context provided to the reflection agent's tool runtime."""
    user_id: str
    agent_name: str
    attempts: int
    latest_results: List[ExperimentResultContext]
    raw_request: str

@tool
async def get_latest_run_results(
    runtime: ToolRuntime[ReflectionToolContext],
) -> Command:
    """
    Gets the results from the test run that just completed.
    This is the primary evidence to start the investigation.
    """
    logger.info("Tool: get_latest_run_results()")
    if not runtime.context:
        raise ValueError("Tool runtime context is missing.")
        
    results = []
    for res in runtime.context.latest_results:
        results.append({
            "example_id": res.dataset_example.example_id,
            "input": res.dataset_example.input_message,
            "passed": res.passed,
            "analysis": res.analysis.model_dump()
        })
    return Command(update={
        "messages": [
            ToolMessage(content=json.dumps(results, indent=2), tool_call_id=runtime.tool_call_id)
        ]
    })

@tool
async def get_historical_test_results(
    example_id: str,
    runtime: ToolRuntime[ReflectionToolContext],
) -> Command:
    """
    Checks for flakiness by retrieving the full pass/fail history
    for a single test case (specified by its example_id).
    """
    logger.info(f"Tool: get_historical_test_results(example_id={example_id})")
    cypher_query = """
    MATCH (ex:DatasetExample {example_id: $example_id})
    MATCH (ex)-[:WAS_RUN_IN]->(res:ExperimentResult)
    RETURN 
        res.passed as passed,
        res.score as score,
        res.completed_at as timestamp
    ORDER BY res.completed_at DESC
    LIMIT 10
    """
    
    results = await asyncio.to_thread(
        NEO4J_GRAPH.query,
        cypher_query,
        params={"example_id": example_id}
    )
    # Convert datetime objects to strings for the LLM
    results_for_llm = [
        {
            "passed": r["passed"],
            "score": r["score"],
            "timestamp": r["timestamp"].isoformat()
        } for r in results
    ]
    return Command(update={
        "messages": [
            ToolMessage(content=json.dumps(results_for_llm, indent=2), tool_call_id=runtime.tool_call_id)
        ]
    })


def persist_reflection(
    user_id: str,
    agent_name: str, 
    reflection: EvalReflection, 
    failed_evidence_results: List[ExperimentResultContext],
    all_latest_results: List[ExperimentResultContext],
) -> None:
    """
    Atomically store the complete EvalReflection object, link it to evidence,
    AND update test case fitness ("culling") in a single, robust transaction.
    """
    
    # 1. Manually generate the embedding
    embedding = _embeddings_client.embed_query(reflection.hypothesis.summary)

    # 2. Extract data for the query
    evidence_thread_ids = [res.thread_id for res in failed_evidence_results]
    
    # Get all test case IDs that were *just run* (passed or failed)
    all_run_example_ids = [
        res.dataset_example.example_id for res in all_latest_results
    ]
    
    # 3. Define the single, atomic Cypher query
    cypher_query = """
    // 1. Create or Merge the main Reflection node
    MERGE (ref:EvalReflection {reflection_id: $reflection_id})
    ON CREATE SET
        ref.summary = $summary,
        ref.embedding = $embedding,
        ref.user_id = $user_id,
        ref.agent_name = $agent_name,
        ref.latest_score = $latest_score,
        ref.attempt = $attempt,
        ref.test_generation_critique = $test_generation_critique
    ON MATCH SET // Update if it already exists
        ref.summary = $summary,
        ref.embedding = $embedding,
        ref.latest_score = $latest_score,
        ref.attempt = $attempt,
        ref.test_generation_critique = $test_generation_critique
    
    // 2. Link it to its evidence (if any)
    WITH ref
    UNWIND $evidence_thread_ids AS thread_id
    
    // Match the result nodes from the *previous* graph step
    MATCH (res:ExperimentResult {thread_id: thread_id, user_id: $user_id})
    
    // Create the link
    MERGE (ref)-[r:GENERATED_FROM]->(res)
    
    // 3. Update fitness ("culling") for ALL tests that were just run
    WITH ref // 'ref' is still in scope
    UNWIND $all_run_example_ids AS ex_id
    
    // Use a subquery to update each example without losing 'ref'
    CALL {
        WITH ex_id // Import only the loop variable
        // The $user_id parameter is available globally in the subquery
        MATCH (ex:DatasetExample {example_id: ex_id, user_id: $user_id})
        
        // Get all historical results for this test
        OPTIONAL MATCH (ex)-[:WAS_RUN_IN]->(hist_res:ExperimentResult {user_id: $user_id})
        WITH ex, hist_res
        ORDER BY hist_res.completed_at DESC
        WITH ex, collect(hist_res) as history
        
        // --- CULLING LOGIC ---
        WITH ex, history,
             CASE
               WHEN size(history) >= 3 AND
                    history[0].passed = true AND
                    history[1].passed = true AND
                    history[2].passed = true
               THEN 'retired'
               ELSE 'active'
             END AS new_status
        
        SET ex.status = new_status
        RETURN count(ex) as updated_count // Complete the subquery
    }
    
    // 'ref' is still in scope here after the CALL/UNWIND
    // We must return a value. We use 'ref' to confirm the reflection node.
    RETURN count(ref) as ref_count
    """
    
    # 4. Execute the query
    result = NEO4J_GRAPH.query(
        cypher_query,
        params={
            "reflection_id": reflection.reflection_id,
            "summary": reflection.hypothesis.summary,
            "embedding": embedding,
            "user_id": user_id,
            "agent_name": agent_name,
            "latest_score": reflection.latest_score,
            "attempt": reflection.attempt,
            "test_generation_critique": reflection.hypothesis.test_generation_critique,
            "evidence_thread_ids": evidence_thread_ids,
            "all_run_example_ids": all_run_example_ids, # Pass in all test IDs
        }
    )
    ref_count = result[0]['ref_count'] if result else 0
    logger.info(f"reflection_store: Atomically stored reflection {reflection.reflection_id} (ref_count: {ref_count}) and updated fitness for {len(all_run_example_ids)} tests.")


@tool
async def save_reflection(
    hypothesis: Hypothesis,
    runtime: ToolRuntime[ReflectionToolContext],
) -> Command:
    """
    Saves the final analysis (the hypothesis) to the graph database
    and concludes the reflection step.
    This is the *final* action the agent should take.
    """
    logger.info(f"Tool: save_reflection(summary={hypothesis.summary})")
    if not runtime.context:
        raise ValueError("Tool runtime context is missing.")

    # --- THIS IS THE "BRIDGE" ---
    # We combine the LLM's hypothesis with the system's metadata
    # to create the full database-ready object.
    
    full_reflection = EvalReflection(
        user_id=runtime.context.user_id,
        hypothesis=hypothesis, 
        # System-generated metadata
        agent_name=runtime.context.agent_name,
        latest_score=round(sum(r.score for r in runtime.context.latest_results) / len(runtime.context.latest_results), 5),
        attempt=runtime.context.attempts,
    )
    
    # Persist the complete EvalReflection object
    await asyncio.to_thread(
        persist_reflection,
        user_id=runtime.context.user_id,
        agent_name=runtime.context.agent_name,
        reflection=full_reflection,
        failed_evidence_results=[r for r in runtime.context.latest_results if not r.passed],
        all_latest_results=runtime.context.latest_results,
    )

    # Return a Command to update the main graph's state
    return Command(update={
        "messages": [
            ToolMessage(content="Reflection saved successfully", tool_call_id=runtime.tool_call_id)
        ],
        "attempts": runtime.context.attempts + 1,
    })
