"""Tools for the reflection agent inside the Eval Agent."""
import json
from typing import List, Any
from pydantic import BaseModel
from langchain_core.tools import tool

from agents.eval_agent.models import Hypothesis
from shared.schema import ExperimentResultContext
from shared.logger import get_logger

logger = get_logger("eval_agent.reflect.tools")

class ReflectionToolContext(BaseModel):
    """Context provided to the reflection agent's tool runtime."""
    user_id: str
    agent_name: str
    attempts: int
    latest_results: List[ExperimentResultContext]
    raw_request: str




def create_reflection_tools(context: ReflectionToolContext) -> List[Any]:
    """
    Creates tool instances bound to the provided context.
    """
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
        Returns empty - no Neo4j, no historical data available.
        """
        logger.info(f"Tool: get_historical_test_results(example_id={example_id})")
        # No Neo4j - return empty list
        return json.dumps([])

    @tool
    def save_reflection(hypothesis: Hypothesis) -> str:
        """
        Saves the final analysis (the hypothesis).
        Now ephemeral - no persistence, just confirms completion.
        Reflection is important to DO, not to persist.
        """
        logger.info(f"Tool: save_reflection(summary={hypothesis.summary})")
        # No persistence - reflection is ephemeral
        # Hypothesis will be extracted from tool call for logging
        return "Reflection completed successfully. Hypothesis analyzed."

    return [get_latest_run_results, get_historical_test_results, save_reflection]
