"""
Generate a concrete agent spec from the current AgentContext and latest user request,
then **pause for human confirmation/edits** via LangGraph interrupts.

This node:
- Summarises the target agent configuration (GitHub repo/branch, MCP services)
- Proposes a list of functional requirements for the target agent
- Uses LangGraph's `interrupt()` API to ask the human to confirm or edit the spec
- Stores the final functional requirements back on AgentContext.functional_requirements
"""
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agents.eval_agent.models import EvalAgentPlannerState
from shared.logger import get_logger
from shared.schema import AgentContext, GithubContext


logger = get_logger("eval_agent.agent_spec.generate_spec")


class FunctionalRequirementsOutput(BaseModel):
    """Structured output for functional requirements only.

    NOTE: Keep this schema simple to avoid strict-mode / additionalProperties pitfalls.
    """

    functional_requirements: List[str] = Field(
        ...,
        description=(
            "A concise, ordered list of functional requirements for the target agent. "
            "Each item should be an imperative requirement, e.g. "
            "'Automatically triage new GitHub issues into Asana tasks'."
        ),
        min_length=1,
    )


async def _get_latest_user_request(state: EvalAgentPlannerState) -> str:
    """Extract the latest human message content as the user request."""
    last_human = None
    for msg in reversed(state.messages or []):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            last_human = msg
            break
    if last_human is not None and getattr(last_human, "content", None):
        return str(last_human.content)

    # Fallback to raw_request in context, if available
    if state.context and state.context.user_context and state.context.user_context.raw_request:
        return state.context.user_context.raw_request

    return ""


async def _propose_functional_requirements(state: EvalAgentPlannerState) -> List[str]:
    """Use an LLM to propose functional requirements given current context + user request."""
    context: AgentContext = state.context or AgentContext()
    user_request = await _get_latest_user_request(state)

    github_repo = context.github_context.repo_url if context.github_context else "N/A"
    github_branch = context.github_context.branch_name if context.github_context else "main"
    mcp_services = context.mcp_services or []
    agent_name = context.agent_name or "agent"

    instruction = (
        "You are designing the functional requirements for an evaluation target agent.\n\n"
        "Given:\n"
        f"- Agent name: {agent_name}\n"
        f"- User request: {user_request}\n"
        f"- GitHub repo: {github_repo}\n"
        f"- Branch: {github_branch}\n"
        f"- MCP services available: {', '.join(mcp_services) if mcp_services else 'none explicitly configured'}\n\n"
        "Produce a list of concrete functional requirements for this agent. "
        "Focus on what the agent must be able to do end‑to‑end, not implementation details.\n"
        "Requirements should be high‑signal, testable behaviors that we can later evaluate."
    )

    # Use a small, non-reasoning model with structured output to avoid schema issues
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.0).with_structured_output(
        FunctionalRequirementsOutput
    )

    fr_output: FunctionalRequirementsOutput = await llm.ainvoke(instruction)
    return fr_output.functional_requirements


async def generate_target_agent_spec(state: EvalAgentPlannerState) -> Dict[str, Any]:
    """
    Generate and confirm the agent spec with the user.

    Flow:
    1. Read existing AgentContext (github_context, mcp_services, etc.)
    2. Use LLM to propose functional requirements
    3. If interrupt is available, pause for user review/edits via Agent Inbox
    4. Store final functional requirements back onto AgentContext.functional_requirements
    """
    context: AgentContext = state.context or AgentContext()

    # Step 1: propose initial functional requirements (idempotent, temperature=0)
    proposed_requirements = await _propose_functional_requirements(state)

    github_repo = context.github_context.repo_url if context.github_context else "N/A"
    github_branch = context.github_context.branch_name if context.github_context else "main"
    mcp_services = context.mcp_services or []
    agent_name = context.agent_name or "agent"


    final_requirements = list(proposed_requirements)

    # Step 3: persist final spec back into AgentContext
    # If the human edited the repo/branch, reflect that in GithubContext too.
    if github_repo != "N/A":
        github_context = GithubContext(
            repo_url=github_repo,
            branch_name=github_branch,
        )
    else:
        github_context = context.github_context

    updated_context = AgentContext(
        user_context=context.user_context,
        github_context=github_context,
        sandbox_context=context.sandbox_context,
        agent_name=agent_name,
        target_agent_version=context.target_agent_version,
        mcp_services=mcp_services,
        mcp_resources=context.mcp_resources,
        functional_requirements=final_requirements,
        tool_entries=context.tool_entries,
        integrations=context.integrations,
        user_id=context.user_id,
    )

    # Also emit a summary AI message so the user sees the spec in the main thread
    summary_lines = [
        "Here is the current target agent specification:",
        f"- **Agent name**: {agent_name}",
        f"- **GitHub repo**: {github_repo}",
        f"- **Branch**: {github_branch}",
        f"- **MCP services**: {', '.join(mcp_services) if mcp_services else 'none'}",
        "",
        "Functional requirements:",
    ]
    for idx, req in enumerate(final_requirements, 1):
        summary_lines.append(f"{idx}. {req}")

    summary_message = AIMessage(
        content="\n".join(summary_lines),
        additional_kwargs={"internal_thinking": True})

    return {
        "context": updated_context,
        "messages": [summary_message],
    }


