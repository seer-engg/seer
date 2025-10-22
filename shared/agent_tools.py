"""Shared tools for all Seer agents"""

import json
import uuid
import httpx
from langchain_core.tools import tool
from .error_handling import create_error_response, create_success_response
from .messaging import messenger
import os


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
        base = f"http://127.0.0.1:{os.getenv('DATA_SERVICE_PORT', '8500')}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{base}/eval-suites", params={"agent_url": agent_url, "agent_id": agent_id})
            resp.raise_for_status()
            data = resp.json()
        suites = data.get("suites", [])
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
    Run a single test against the target LangGraph agent via LangGraph SDK.
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(target_url)
        port = parsed.port or 2024
        base_url = f"http://127.0.0.1:{port}"
        user_tid = thread_id or str(uuid.uuid4())
        text, remote_tid = await messenger.send(
            user_thread_id=user_tid,
            src_agent="eval_agent",
            dst_agent=target_agent_id,
            base_url=base_url,
            assistant_id=target_agent_id,
            content=test_input
        )
        return create_success_response({"response": text, "thread_id": remote_tid})
    except Exception as e:
        return create_error_response(f"Failed to run test via SDK: {str(e)}", e)


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
        base = f"http://127.0.0.1:{os.getenv('DATA_SERVICE_PORT', '8500')}"
        payload = {
            "suite_id": eval_suite.get("id"),
            "spec_name": eval_suite.get("spec_name"),
            "spec_version": eval_suite.get("spec_version"),
            "test_cases": eval_suite.get("test_cases", []),
            "target_agent_url": target_url,
            "target_agent_id": target_id,
            "thread_id": thread_id,
            "langgraph_thread_id": langgraph_thread_id
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{base}/eval-suites", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return create_success_response({"eval_suite_id": data.get("suite_id"), "message": data.get("message")})
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
        base = f"http://127.0.0.1:{os.getenv('DATA_SERVICE_PORT', '8500')}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{base}/test-results", json={"suite_id": suite_id, "thread_id": thread_id, "results": results})
            resp.raise_for_status()
            data = resp.json()
        return create_success_response({"results_count": data.get("results_count"), "message": data.get("message")})
    except Exception as e:
        return create_error_response(f"Failed to store test results: {str(e)}", e)
