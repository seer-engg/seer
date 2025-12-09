"""Answer informational queries directly."""
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from agents.eval_agent.models import EvalAgentState
from shared.logger import get_logger

logger = get_logger("eval_agent.intent")

EVAL_AGENT_CAPABILITIES = """I am the Seer Eval Agent, an autonomous reliability engineering system for evaluating AI agents.

**What I Can Do:**
1. **Generate Agent Specifications**: Analyze your agent and create structured specs (goals, capabilities, integrations)
2. **Create Test Plans**: Generate test case intents and validation criteria
3. **Evaluate Agents**: Run comprehensive tests against your agent in real sandbox environments
4. **Alignment Workflow**: Ask clarifying questions to ensure test plans match your expectations
5. **Reflection & Analysis**: Analyze test results and provide insights

**How to Use Me:**
- **To generate a test plan**: "Evaluate my agent at https://github.com/owner/repo" or "Create a test plan for my agent"
- **To run tests**: Provide a GitHub URL and I'll provision a sandbox and execute tests
- **Plan-only mode**: I can generate specs and test plans without executing (faster, cheaper)

**What I Need:**
- Agent description or GitHub repository URL
- MCP services your agent uses (e.g., Asana, GitHub, Jira)
- Your agent's goals and capabilities

**Example Requests:**
- "Evaluate my GitHub PR reviewer agent"
- "Create test cases for my Asana task manager agent"
- "Generate an agent spec for my agent at https://github.com/owner/repo"
"""


async def answer_informational_query(state: EvalAgentState) -> dict:
    """Answer informational queries directly."""
    last_human = None
    for msg in reversed(state.messages or []):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            last_human = msg
            break
    
    if not last_human:
        return {
            "messages": [AIMessage(content=EVAL_AGENT_CAPABILITIES)]
        }
    
    # Use LLM to generate contextual answer
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.0)
    
    prompt = f"""You are the Seer Eval Agent. Answer the user's informational question about your capabilities.

**Your Capabilities:**
{EVAL_AGENT_CAPABILITIES}

**User's Question:**
{last_human.content}

Provide a helpful, concise answer. If they're asking about capabilities, explain what you can do. If they're asking how to use you, provide clear instructions."""

    response = await llm.ainvoke(prompt)
    logger.info("Answered informational query directly")
    
    return {
        "messages": [AIMessage(content=response.content)]
    }

