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
from agents.eval_agent.reflection_store import _truncate
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
    user_expectation: str


def _truncate(text: Any, limit: int = 280) -> str:
    """Helper to keep prompt context small."""
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… (truncated {len(text)} chars)"


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
        # --- MODIFIED: Pass the structured analysis ---
        results.append({
            "example_id": res.dataset_example.example_id,
            "input": _truncate(res.dataset_example.input_message),
            "passed": res.passed,
            "analysis": res.analysis.model_dump() # Pass the full structured analysis
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
    evidence_results: List[ExperimentResultContext]
) -> None:
    """
    Atomically store the complete EvalReflection object AND its links
    in a single, robust transaction.
    This solves a race condition where ExperimentResult nodes were
    not queryable in time for link creation.
    """
    
    # 1. Manually generate the embedding
    embedding = _embeddings_client.embed_query(reflection.hypothesis.summary)

    # 2. Extract data for the query
    evidence_thread_ids = [res.thread_id for res in evidence_results]
    
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
        ref.failure_modes = $failure_modes,
        ref.recommended_tests = $recommended_tests
    ON MATCH SET // Update if it already exists (e.g., if embedding failed first time)
        ref.summary = $summary,
        ref.embedding = $embedding,
        ref.latest_score = $latest_score,
        ref.attempt = $attempt,
        ref.failure_modes = $failure_modes,
        ref.recommended_tests = $recommended_tests
    
    // 2. Link it to its evidence (if any)
    WITH ref
    UNWIND $evidence_thread_ids AS thread_id
    
    // Match the result nodes from the *previous* graph step
    MATCH (res:ExperimentResult {thread_id: thread_id, user_id: $user_id})
    
    // Create the link
    MERGE (ref)-[r:GENERATED_FROM]->(res)
    
    RETURN count(r) as links_created
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
            "failure_modes": reflection.hypothesis.failure_modes,
            "recommended_tests": reflection.hypothesis.recommended_tests,
            "evidence_thread_ids": evidence_thread_ids,
        }
    )
    links = result[0]['links_created'] if result else 0
    logger.info(f"reflection_store: Atomically stored reflection {reflection.reflection_id} and created {links} links.")


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
        latest_score=sum(r.score for r in runtime.context.latest_results) / len(runtime.context.latest_results),
        attempt=runtime.context.attempts,
        # (reflection_id and created_at are set by default)
    )
    
    # Persist the complete EvalReflection object
    await asyncio.to_thread(
        persist_reflection,
        user_id=runtime.context.user_id,
        agent_name=runtime.context.agent_name,
        reflection=full_reflection,
        evidence_results=[r for r in runtime.context.latest_results if not r.passed],
    )

    # Return a Command to update the main graph's state
    return Command(update={
        "messages": [
            ToolMessage(content="Reflection saved successfully", tool_call_id=runtime.tool_call_id)
        ],
        "attempts": runtime.context.attempts + 1,
    })
