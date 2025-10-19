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
    Run a single test against the target LangGraph agent.
    
    Args:
        target_url: URL of the target agent to test (e.g. http://localhost:2024)
        target_agent_id: Graph/assistant ID for the target (e.g. 'my_agent')
        test_input: Input message to send to the agent
        thread_id: Optional thread ID for context
    
    Returns:
        JSON string with the agent's response
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
    url = f"{target_url}/runs/stream"
    payload = {
        "assistant_id": target_agent_id,
        "input": {
            "messages": [{"role": "user", "content": test_input}]
        },
        "stream_mode": ["values"],
        "config": {"configurable": {"thread_id": thread_id}}
    }
    
    try:
        config = get_seer_config()
        async with httpx.AsyncClient(timeout=config.test_timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            
            # Parse streaming response to get final message
            final_response = ""
            for line in response.text.strip().split('\n'):
                if line.startswith('data: '):
                    data = json.loads(line[6:])  # Skip 'data: ' prefix
                    if 'messages' in data:
                        messages = data['messages']
                        if messages and len(messages) > 0:
                            last_msg = messages[-1]
                            if isinstance(last_msg, dict) and last_msg.get('type') == 'ai':
                                final_response = last_msg.get('content', '')
            
            if final_response:
                return create_success_response({"response": final_response})
            else:
                return create_success_response({"response": "No response received"})
    except Exception as e:
        return create_error_response(f"Failed to run test: {str(e)}", e)


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
