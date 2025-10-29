from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig

from agents.codex.graphs.reviewer.models import ReviewerState
from agents.codex.graphs.reviewer.nodes.test_creator import test_creator_node
from agents.codex.graphs.reviewer.nodes.test_executor import test_executor_node
from shared.logger import get_logger

logger = get_logger('codex.reviewer')


def finalize_node(state: ReviewerState, config: RunnableConfig) -> dict:
    """
    Finalize the test results and prepare final message.
    """
    verdict = state.test_verdict
    
    # Build final review message
    if verdict.all_passed:
        message_content = f"✓ All tests passed! ({verdict.passed_tests}/{verdict.total_tests})\n\n"
        message_content += f"Test Description: {state.test_suite.description if state.test_suite else 'N/A'}\n\n"
        message_content += "Code implementation validated successfully."
    else:
        message_content = f"✗ Tests failed: {verdict.failed_tests}/{verdict.total_tests} tests failed\n\n"
        message_content += f"Test Description: {state.test_suite.description if state.test_suite else 'N/A'}\n\n"
        if verdict.issues:
            message_content += "Issues found:\n"
            for issue in verdict.issues:
                message_content += f"  - {issue}\n"
        message_content += f"\n{verdict.execution_summary}"
    
    logger.info(f"Review finalized - All passed: {verdict.all_passed}")
    
    # Add final message to state
    messages = list(state.messages) if state.messages else []
    messages.append({"role": "system", "content": message_content})
    
    return {
        "messages": messages
    }


def compile_reviewer_graph():
    """
    Build the reviewer graph with test creator and executor nodes.
    
    Flow:
    START -> test_creator -> test_executor -> [conditional]
                                              |
                                              └─> finalize -> END
    
    Future enhancement: Add test_refiner node for iterative test improvement
    """
    workflow = StateGraph(ReviewerState)
    
    # Add nodes
    workflow.add_node("test_creator", test_creator_node)
    workflow.add_node("test_executor", test_executor_node)
    workflow.add_node("finalize", finalize_node)
    
    # Add edges
    workflow.add_edge(START, "test_creator")
    workflow.add_edge("test_creator", "test_executor")
    workflow.add_edge("test_executor", "finalize")
    
    # Finalize goes to END
    workflow.add_edge("finalize", END)
    
    # Compile
    compiled_graph = workflow.compile()
    
    logger.info("Reviewer graph compiled successfully")
    return compiled_graph


graph = compile_reviewer_graph()