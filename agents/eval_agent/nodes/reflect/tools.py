"""Tools for the reflection agent inside the Eval Agent."""
import json
from typing import List, Any
from pydantic import BaseModel
from langchain_core.tools import tool

from agents.eval_agent.models import Hypothesis
from shared.schema import ExperimentResultContext
from shared.logger import get_logger
from agents.eval_agent.reflexion_factory import get_memory_store # Unified Store Access

logger = get_logger("eval_agent.reflect.tools")

class ReflectionToolContext(BaseModel):
    """Context provided to the reflection agent's tool runtime."""
    user_id: str
    agent_name: str
    attempts: int
    latest_results: List[ExperimentResultContext]
    raw_request: str


def link_evidence_and_update_fitness(
    user_id: str,
    memory_id: str, 
    failed_evidence_results: List[ExperimentResultContext],
    all_latest_results: List[ExperimentResultContext],
) -> None:
    """
    Link the Memory node to its evidence and update test case fitness.
    """
    store = get_memory_store()
    
    evidence_thread_ids = [res.thread_id for res in failed_evidence_results]
    all_run_example_ids = [
        res.dataset_example.example_id for res in all_latest_results
    ]
    
    cypher_query = """
    // 1. Link Memory to Evidence
    MATCH (m:Memory {memory_id: $memory_id})
    WITH m
    UNWIND $evidence_thread_ids AS thread_id
    MATCH (res:ExperimentResult {thread_id: thread_id, user_id: $user_id})
    MERGE (m)-[:GENERATED_FROM]->(res)
    
    // 2. Update Fitness
    WITH m
    UNWIND $all_run_example_ids AS ex_id
    CALL {
        WITH ex_id
        MATCH (ex:DatasetExample {example_id: ex_id, user_id: $user_id})
        OPTIONAL MATCH (ex)-[:WAS_RUN_IN]->(hist_res:ExperimentResult {user_id: $user_id})
        WITH ex, hist_res
        ORDER BY hist_res.completed_at DESC
        WITH ex, collect(hist_res) as history
        
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
        RETURN count(ex) as updated_count
    }
    RETURN count(m) as linked_count
    """
    
    with store.driver.session() as session:
        result = session.run(
            cypher_query,
            params={
                "memory_id": memory_id,
                "user_id": user_id,
                "evidence_thread_ids": evidence_thread_ids,
                "all_run_example_ids": all_run_example_ids,
            }
        )
        record = result.single()
        linked_count = record['linked_count'] if record else 0

    logger.info(f"Linked evidence to Memory {memory_id} (linked_count: {linked_count}) and updated fitness.")


def create_reflection_tools(context: ReflectionToolContext) -> List[Any]:
    """
    Creates tool instances bound to the provided context.
    """
    store = get_memory_store()
    
    @tool
    def get_latest_run_results() -> str:
        """
        Gets the results from the test run that just completed.
        This is the primary evidence to start the investigation.
        """
        logger.info("Tool: get_latest_run_results()")
        
        results = []
        for res in context.latest_results:
            results.append({
                "example_id": res.dataset_example.example_id,
                "input": res.dataset_example.input_message,
                "passed": res.passed,
                "analysis": res.analysis.model_dump()
            })
        return json.dumps(results, indent=2)

    @tool
    def get_historical_test_results(example_id: str) -> str:
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
        
        with store.driver.session() as session:
            results = session.run(
                cypher_query,
                params={"example_id": example_id}
            )
            records = [record.data() for record in results]
        
        results_for_llm = [
            {
                "passed": r["passed"],
                "score": r["score"],
                "timestamp": r["timestamp"].isoformat() if hasattr(r["timestamp"], 'isoformat') else str(r["timestamp"])
            } for r in records
        ]
        return json.dumps(results_for_llm, indent=2)

    @tool
    def save_reflection(hypothesis: Hypothesis) -> str:
        """
        Saves the final analysis (the hypothesis) to the graph database
        and concludes the reflection step.
        This is the *final* action the agent should take.
        """
        logger.info(f"Tool: save_reflection(summary={hypothesis.summary})")

        # Use Shared Store
        # store = get_memory_store() # Already got at top of closure

        # 1. Create and Save Memory with Metadata
        from reflexion.core.memory.models import Memory
        
        memory = Memory(
            agent_id=context.agent_name,
            context="eval_agent.reflection",
            entities=["evaluation", "hypothesis"],
            observation=hypothesis.summary,
            metadata={
                "type": "EvalReflection",
                "user_id": context.user_id,
                "latest_score": round(sum(r.score for r in context.latest_results) / len(context.latest_results), 5) if context.latest_results else 0.0,
                "attempt": context.attempts,
                "test_generation_critique": hypothesis.test_generation_critique
            },
            user_id=context.user_id,
            score=round(sum(r.score for r in context.latest_results) / len(context.latest_results), 5) if context.latest_results else 0.0
        )
        memory_id = store.save(memory)
        
        # 2. Link Evidence & Update Fitness
        link_evidence_and_update_fitness(
            user_id=context.user_id,
            memory_id=memory_id,
            failed_evidence_results=[r for r in context.latest_results if not r.passed],
            all_latest_results=context.latest_results,
        )

        return "Reflection saved successfully. You may now conclude."

    return [get_latest_run_results, get_historical_test_results, save_reflection]
