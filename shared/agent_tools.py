"""Shared tools for all Seer agents"""

import json
import uuid
import httpx
from typing import Dict, Any
from langchain_core.tools import tool
from .error_handling import create_error_response, create_success_response
from .config import get_seer_config
from .a2a_utils import send_a2a_message


@tool
def acknowledge_user(confirmed: bool, details: str = "") -> str:
    """
    Acknowledge user's confirmation response.
    
    Args:
        confirmed: Whether user confirmed
        details: Additional details from user
    
    Returns:
        JSON acknowledgment with metadata
    """
    return json.dumps({
        "action": "CONFIRMATION",
        "confirmed": confirmed,
        "details": details
    })


@tool
def request_confirmation(question: str, test_count: int) -> str:
    """
    Request user confirmation before running tests.
    
    Args:
        question: Question to ask user
        test_count: Number of tests that will be run
    
    Returns:
        JSON metadata about the confirmation request
    """
    return json.dumps({
        "action": "CONFIRMATION_REQUEST",
        "test_count": test_count,
        "question": question
    })


@tool
def summarize_results(passed: int, failed: int, total: int) -> str:
    """
    Summarize test results.
    
    Args:
        passed: Number of tests passed
        failed: Number of tests failed
        total: Total tests
    
    Returns:
        JSON summary of test results
    """
    score = (passed / total * 100) if total > 0 else 0
    return json.dumps({
        "action": "TEST_RESULTS",
        "passed": passed,
        "failed": failed,
        "total": total,
        "score": score,
        "summary": f"{passed}/{total} passed ({score:.0f}%)"
    })


@tool
async def get_test_cases(agent_url: str, agent_id: str) -> str:
    """
    Get test cases for a specific agent from the Orchestrator.
    
    Args:
        agent_url: URL of the target agent (e.g. "http://localhost:2024")
        agent_id: ID of the target agent (e.g. "deep_researcher")

    Returns:
        JSON string with test cases or error message
    """
    try:
        config = get_seer_config()
        orchestrator_response = await send_a2a_message(
            "orchestrator",
            config.orchestrator_port,
            json.dumps({
                "action": "get_eval_suites",
                "payload": {
                    "agent_url": agent_url,
                    "agent_id": agent_id
                }
            }),
            thread_id=config.generate_thread_id("test_cases_request")
        )

        orchestrator_data = json.loads(orchestrator_response)
        if not orchestrator_data.get("success"):
            return create_error_response(f"Failed to get test cases: {orchestrator_data.get('error', 'Unknown error')}")

        suites = orchestrator_data.get("suites", [])
        if not suites:
            return create_error_response(f"No test cases found for {agent_url}/{agent_id}")

        latest_suite = suites[-1]
        test_cases = latest_suite.get("test_cases", [])

        return create_success_response({
            "eval_suite_id": latest_suite.get("suite_id"),
            "test_count": len(test_cases),
            "test_cases": test_cases
        })

    except Exception as e:
        return create_error_response(f"Failed to get test cases: {str(e)}", e)


@tool
async def run_test(target_url: str, target_agent_id: str, test_input: str, thread_id: str = None) -> str:
    """
    Run a single test against the target LangGraph agent via A2A (persistent thread).
    """
    try:
        # Extract port and call A2A using graph name or UUID
        from urllib.parse import urlparse
        parsed = urlparse(target_url)
        port = parsed.port or (2024 if parsed.scheme in ("http", "https") else 2024)
        a2a_resp = await send_a2a_message(
            target_agent_id=target_agent_id,
            target_port=port,
            message=test_input,
            thread_id=thread_id or str(uuid.uuid4())
        )
        return a2a_resp
    except Exception as e:
        return create_error_response(f"Failed to run test via A2A: {str(e)}", e)


@tool
async def store_eval_suite(target_url: str, target_id: str, eval_suite: dict, thread_id: str = None, langgraph_thread_id: str = None) -> str:
    """
    Store eval suite in Orchestrator database.

    Args:
        target_url: URL of the target agent
        target_id: ID of the target agent
        eval_suite: The eval suite data
        thread_id: Thread ID
        langgraph_thread_id: LangGraph thread ID

    Returns:
        JSON string with store result
    """
    try:
        config = get_seer_config()
        orchestrator_response = await send_a2a_message(
            "orchestrator",
            config.orchestrator_port,
            json.dumps({
                "action": "store_eval_suite",
                "payload": {
                    "suite_id": eval_suite.get("id"),
                    "spec_name": eval_suite.get("spec_name"),
                    "spec_version": eval_suite.get("spec_version"),
                    "test_cases": eval_suite.get("test_cases", []),
                    "target_agent_url": target_url,
                    "target_agent_id": target_id,
                    "thread_id": thread_id,
                    "langgraph_thread_id": langgraph_thread_id
                }
            }),
            thread_id=thread_id or config.eval_suite_storage_thread
        )

        orchestrator_data = json.loads(orchestrator_response)
        if not orchestrator_data.get("success"):
            return create_error_response(f"Orchestrator error: {orchestrator_data.get('error')}")

        return create_success_response({
            "eval_suite_id": orchestrator_data.get("suite_id"),
            "message": orchestrator_data.get("message")
        })
    except Exception as e:
        return create_error_response(f"Failed to store eval suite: {str(e)}", e)


@tool
async def store_test_results(suite_id: str, thread_id: str, results: list) -> str:
    """
    Store test results in Orchestrator database.

    Args:
        suite_id: ID of the eval suite
        thread_id: Thread ID
        results: List of test results

    Returns:
        JSON string with store result
    """
    try:
        config = get_seer_config()
        orchestrator_response = await send_a2a_message(
            "orchestrator",
            config.orchestrator_port,
            json.dumps({
                "action": "store_test_results",
                "payload": {
                    "suite_id": suite_id,
                    "thread_id": thread_id,
                    "results": results
                }
            }),
            thread_id=thread_id or config.test_results_storage_thread
        )

        orchestrator_data = json.loads(orchestrator_response)
        if not orchestrator_data.get("success"):
            return create_error_response(f"Orchestrator error: {orchestrator_data.get('error')}")

        return create_success_response({
            "results_count": orchestrator_data.get("results_count"),
            "message": orchestrator_data.get("message")
        })
    except Exception as e:
        return create_error_response(f"Failed to store test results: {str(e)}", e)
