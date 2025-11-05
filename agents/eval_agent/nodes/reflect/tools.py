"""Tools for the reflection agent inside the Eval Agent."""
import asyncio
import json
from typing import List, Any
from pydantic import BaseModel
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage

from agents.eval_agent.constants import NEO4J_GRAPH, NEO4J_VECTOR
from agents.eval_agent.models import EvalReflection, Hypothesis
from agents.eval_agent.reflection_store import _truncate
from shared.schema import ExperimentResultContext
from shared.logger import get_logger



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
        results.append({
            "example_id": res.dataset_example.example_id,
            "input": _truncate(res.dataset_example.input_message),
            "passed": res.passed,
            "score": res.score,
            "judge_reasoning": _truncate(res.judge_reasoning),
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
    Store the complete EvalReflection object in the graph
    and its summary in the vector index.
    """
    
    # 1. Store the summary in the vector index
    doc = Document(
        page_content=reflection.hypothesis.summary,
        metadata={
            "user_id": user_id,
            "reflection_id": reflection.reflection_id,
            "agent_name": agent_name,
            "latest_score": reflection.latest_score,
            "attempt": reflection.attempt,
        }
    )
    NEO4J_VECTOR.add_documents([doc])
    logger.info(f"reflection_store: Stored summary for {reflection.reflection_id} in vector index.")
        
    # 2. Store the full reflection node and link it to evidence
    evidence_thread_ids = [res.thread_id for res in evidence_results]
    
    # Extract the primitive lists from the hypothesis
    failure_modes_list = reflection.hypothesis.failure_modes
    recommended_tests_list = reflection.hypothesis.recommended_tests

    cypher_query = """
    MERGE (ref:EvalReflection {reflection_id: $reflection_id})
    SET
        ref.failure_modes = $failure_modes,
        ref.recommended_tests = $recommended_tests,
        ref.user_id = $user_id
        // 'summary' and 'embedding' are already set by add_documents()

    WITH ref
    UNWIND $evidence_thread_ids AS thread_id
    MATCH (res:ExperimentResult {thread_id: thread_id, user_id: $user_id})
    MERGE (ref)-[r:GENERATED_FROM]->(res)
    RETURN count(r) as links_created
    """
    
    NEO4J_GRAPH.query(
        cypher_query,
        params={
            "user_id": user_id,
            "reflection_id": reflection.reflection_id,
            "failure_modes": failure_modes_list,          # Pass the list
            "recommended_tests": recommended_tests_list,  # Pass the list
            "evidence_thread_ids": evidence_thread_ids,
        }
    )
    logger.info(f"reflection_store: Linked reflection {reflection.reflection_id} to evidence for user {user_id}.")


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
