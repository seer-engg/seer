# Reflexion Coding Agent

A self-improving coding agent that uses the reflexion pattern to write high-quality, production-ready code through iterative refinement with test-driven evaluation.

## Architecture

The Reflexion Coding Agent implements an iterative refinement loop with three key components specialized for code generation:

### Components

1. **Actor (Coding Agent)** (`actor_node`)
   - Expert software engineer that writes production-ready code
   - Generates clean, well-documented code for user requests
   - Accesses persistent memory containing coding patterns and lessons learned
   - Improves code with each iteration based on test failures and feedback
   - Follows best practices: SOLID, DRY, proper error handling
   - Uses LLM with temperature=0.7 for creative but consistent code
   - **Visible to user**: Yes - this is the code the user receives

2. **Evaluator (Test Engineer - ReAct Agent)** (`evaluator_node`)
   - **Autonomous ReAct agent** with tools for code testing
   - Makes its own decisions about what tests to run
   - **Tools available**:
     - `extract_code_from_response`: Extract code from markdown
     - `execute_code_in_sandbox`: Run code in E2B sandbox
     - `run_test_in_sandbox`: Execute individual unit tests
     - `create_test_summary`: Generate test result summaries
   - **Executes tests in E2B sandbox** - actual code execution, not simulation
   - Iteratively designs and runs tests based on code analysis
   - Tests for correctness, edge cases, error handling, and quality
   - Provides pass/fail verdict based on real execution results
   - Uses LLM with temperature=0.0 for consistent evaluation
   - **Visible to user**: No - runs behind the scenes, logged only
   - **Technology**: LangChain `create_agent` + E2B Code Interpreter

3. **Reflection (Senior Architect)** (`reflection_node`)
   - Analyzes why code failed tests
   - Suggests coding paradigms, patterns, and best practices
   - Provides actionable feedback with code examples
   - Teaches design patterns (Factory, Strategy, etc.), SOLID principles
   - Stores insights in persistent memory for continuous learning
   - Uses LLM with temperature=0.3 for focused analysis
   - **Visible to user**: No - stored in memory, not shown to user

### Flow Diagram

```
START
  ↓
[Actor] ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
  ↓                               │ (stores feedback in memory)
[Evaluator]                       │
  ↓                               │
[Decision Point]                  │
  ├─→ Pass? → [Finalize] → END   │
  └─→ Fail & Attempts < Max? ─ ─ ┘
      │                          ↑
      └─→ [Reflection] ──────────┘
           (behind the scenes)
      │
      └─→ Max Attempts? → [Finalize] → END
```

**What the user sees**: Just a clean conversation with a coding assistant!

```
User: "Write a function to sort a list of dictionaries by a key"
Actor: [First code implementation]
Actor: [Improved code with better edge case handling - if tests failed]
Actor: [Final production-ready code]
```

Behind the scenes:
- **Evaluator ReAct agent** autonomously designs and executes tests
- Uses tools to run code in E2B sandbox
- Makes decisions about which tests to run based on code analysis
- Real code execution with actual test results (not simulated)
- Reflection suggests coding paradigms to fix failures
- All feedback stored in persistent memory for continuous learning

### Decision Logic

After evaluation, the system decides:
- **If evaluator passes** → Finalize and return success
- **If max attempts reached** → Finalize with best effort response
- **Otherwise** → Generate reflection feedback, store in memory, and loop back to Actor

## State Schema

```python
ReflexionState = {
    "messages": list[BaseMessage],        # Conversation history (user & agent messages)
    "current_attempt": int,               # Current iteration count
    "max_attempts": int,                  # Maximum iterations allowed
    "evaluator_verdict": Verdict | None,  # Latest evaluation result
    "success": bool,                      # Whether evaluation passed
    "memory_key": str                     # Key for persistent memory store
}
```

## Memory Layer

The memory layer is implemented using **LangGraph's persistent Store API**, which stores reflection feedback **across threads**. This allows the actor to learn from past interactions, not just within a single conversation.

### How It Works:

1. **Persistent Storage**: Reflections are stored in a persistent memory store using namespace `("reflexion", "feedback")`
2. **Cross-Thread Access**: Actor can retrieve reflections from previous conversations using `memory_key`
3. **Continuous Learning**: Each failed iteration generates reflection feedback that's stored for future use
4. **Memory Retrieval**: Actor queries the store at the start of each attempt to incorporate learned lessons

### Memory Flow:

1. User sends message → Actor retrieves past reflections for this `memory_key`
2. Actor generates response using conversation history + reflection memory
3. Evaluator judges the response
4. If failed, Reflection analyzes and stores feedback in persistent memory
5. Loop continues with Actor accessing updated memory

### Memory Key:

The `memory_key` identifies which memory pool to use. Examples:
- `"user_123"` - User-specific memory
- `"domain_ml"` - Domain-specific memory (e.g., all ML questions)
- `"default"` - Global memory pool

## User Experience

From the end user's perspective, the reflexion coding agent appears as an **expert coding assistant**. They:

1. Request code for a specific task
2. Receive production-ready code from the Actor
3. If the code needs refinement, they see improved versions naturally

**The user never sees**:
- Unit test creation and execution
- Test failures or passing verdicts
- Coding paradigm suggestions from Reflection
- Retry attempt counting
- Internal quality scores

**Example conversation from user's view**:
```
User: "Write a Python function to merge overlapping intervals"

Actor: "Here's a function to merge overlapping intervals:

def merge_intervals(intervals):
    '''Merge overlapping intervals and return consolidated list'''
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        if current[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], current[1]))
        else:
            merged.append(current)
    
    return merged

This handles overlapping intervals by sorting first, then merging..."

# Behind the scenes: 
# - Evaluator generates tests: [], [(1,2)], [(1,3),(2,6)], edge cases
# - Executes in E2B sandbox
# - Tests fail: missing validation for invalid intervals
# - Actual error: "AttributeError: 'str' object has no attribute '__getitem__'"
# - Reflection suggests: "Add input validation pattern with type checking"

Actor: "Here's the improved version with better error handling:

def merge_intervals(intervals):
    '''Merge overlapping intervals with input validation'''
    if not intervals:
        return []
    
    # Validate input
    if not all(isinstance(i, tuple) and len(i) == 2 for i in intervals):
        raise ValueError('Each interval must be a tuple of 2 elements')
    
    if not all(i[0] <= i[1] for i in intervals):
        raise ValueError('Invalid interval: start must be <= end')
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        if current[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], current[1]))
        else:
            merged.append(current)
    
    return merged

This version adds proper validation and handles edge cases..."

# Behind the scenes: 
# - All tests executed in E2B sandbox
# - 5/5 tests passed ✓
# - Code meets requirements
```

The user sees natural code evolution, getting progressively better implementations without seeing the internal testing and reflection process.

## Usage

### Basic Usage

```python
from agents.reflexion.graph import graph
from langchain_core.messages import HumanMessage

# Configure the agent with store for persistent memory
config = {
    "configurable": {
        "thread_id": "unique_thread_id",
        # "store": your_store_instance  # Required for persistent memory
    }
}

# Prepare initial state - uses message chain
initial_state = {
    "messages": [HumanMessage(content="Your query here")],
    "current_attempt": 1,
    "max_attempts": 3,  # Try up to 3 times
    "memory_key": "user_123"  # Key for persistent memory
}

# Run the agent
result = await graph.ainvoke(initial_state, config=config)

# Access results
print(f"Success: {result['success']}")
print(f"Messages: {len(result['messages'])}")
print(f"Score: {result['evaluator_verdict']['score']}")
```

### Example Script

See `example_usage.py` for a complete working example.

### Running the Example

```bash
cd /home/ubuntu/lokesh/seer
python -m agents.reflexion.example_usage
```

## Configuration

### Max Attempts

Control iteration limit by setting `max_attempts` in initial state:

```python
initial_state = {
    "max_attempts": 5  # Try up to 5 iterations
}
```

### Persistent Memory Store

For cross-thread persistent memory, configure a store in the config:

```python
from langgraph.checkpoint.postgres import PostgresSaver

# When deploying with LangGraph API, the store is automatically provided
config = {
    "configurable": {
        "thread_id": "user_thread_123",
        "store": store_instance  # LangGraph API provides this automatically
    }
}
```

**Note**: When deployed via LangGraph API, the store is automatically available and configured. No manual setup needed!

### Memory Key Strategy

Choose a memory key strategy based on your use case:

```python
# User-specific memory - each user has their own reflection pool
"memory_key": f"user_{user_id}"

# Domain-specific memory - shared learning within a domain
"memory_key": f"domain_{topic}"  # e.g., "domain_python", "domain_ml"

# Global memory - all reflections shared
"memory_key": "global"
```

### LLM Temperature Settings

Temperatures are optimized for each component:
- **Actor**: 0.7 (creative responses)
- **Evaluator**: 0.0 (consistent judgment)
- **Reflection**: 0.3 (focused analysis)

Modify in `graph.py` if needed:

```python
llm = get_llm(temperature=0.7)  # Adjust as needed
```

## Code Evaluation Criteria

The Evaluator creates unit tests and judges code on:

1. **Correctness**: Does the code produce correct outputs for all test cases?
2. **Completeness**: Does it handle all requirements from the user?
3. **Edge Cases**: Empty inputs, null values, boundary conditions
4. **Error Handling**: Graceful handling of invalid inputs and exceptions
5. **Code Quality**: Clean, readable, well-documented code
6. **Best Practices**: Follows language-specific standards (PEP 8 for Python)
7. **Security**: No vulnerabilities (injection, validation issues)
8. **Performance**: Reasonable efficiency for the task

### Test Cases Covered:
- **Happy path**: Normal, expected inputs
- **Edge cases**: Empty, null, boundary values (0, -1, max)
- **Error scenarios**: Invalid types, malformed inputs
- **Performance**: Large datasets (when applicable)
- **Security**: Input validation, sanitization

## Integration

### As a Standalone Service

The reflexion agent can be deployed as a LangGraph service:

```bash
cd agents/reflexion
langgraph dev
```

Default port: Configure in deployment settings

### With Other Agents

The reflexion agent can be integrated with the orchestrator or other agents via inter-agent communication:

```python
# From orchestrator or another agent
response = await messenger.send(
    src_agent="orchestrator",
    dst_agent="reflexion",
    content=user_query,
    state_update={
        "max_attempts": 3
    }
)
```

## Prompts

All prompts are defined in `prompts.py`:
- `ACTOR_PROMPT`: Guides the Actor's response generation
- `EVALUATOR_PROMPT`: Defines evaluation criteria
- `REFLECTION_PROMPT`: Structures feedback generation

Customize these prompts to adjust behavior.

## Logging

The agent uses the shared logger from `shared.logger`:

```python
logger = get_logger('reflexion_agent')
```

Logs include:
- Attempt numbers
- Verdict results
- Reflection feedback counts
- Success/failure status

## Best Practices

1. **Start with 3 attempts**: Balance between code quality and cost
2. **Clear coding requests**: Specific requirements → better generated code
   - Good: "Write a function to merge sorted lists with duplicates removed"
   - Bad: "Write a sorting function"
3. **Monitor scores**: Track code quality improvement across iterations
4. **Review reflections**: Learn what coding patterns help most
5. **Memory key strategy**: 
   - User-specific for personalized coding style
   - Domain-specific for language/framework best practices
6. **Adjust temperatures**: 
   - Actor 0.7 for creative solutions
   - Evaluator 0.0 for consistent testing
   - Reflection 0.3 for focused feedback

## Troubleshooting

### Code doesn't improve across iterations
- Check that the store is properly configured in config
- Verify Actor is retrieving memory (check logs for "Retrieved X reflections")
- Ensure `memory_key` is consistent across invocations
- Review reflection feedback quality - are suggestions actionable?
- Check if reflection includes specific coding paradigms and patterns

### Memory not persisting across threads
- Confirm store is configured in the `configurable` section of config
- Check that `memory_key` is being set correctly
- Verify LangGraph API deployment has store enabled
- Check logs for "No store available" warnings
- Test memory retrieval with a simple query

### Evaluator too strict (all code fails)
- Review test cases being created - are they reasonable?
- Check if edge cases are too obscure or impractical
- Adjust EVALUATOR_PROMPT in `prompts.py` to balance thoroughness
- Consider language/framework-specific nuances
- Lower the passing threshold for initial iterations

### Evaluator too lenient (bad code passes)
- Strengthen test case requirements in EVALUATOR_PROMPT
- Add more edge case categories (security, performance)
- Increase focus on error handling and validation
- Review actual test failures that should have been caught

### Generated code has recurring bugs
- Check reflection memory - are lessons being stored?
- Verify Actor is reading and applying memory context
- Improve REFLECTION_PROMPT to provide more specific code examples
- Use more specific `memory_key` for the problem domain
- Add explicit coding pattern suggestions to reflections

### Max attempts reached frequently
- Increase `max_attempts` (3-5) for complex coding tasks
- Simplify evaluation criteria for first iterations
- Improve reflection feedback specificity with code snippets
- Check if Actor is actually seeing and incorporating feedback
- Consider breaking complex tasks into smaller functions

## Future Enhancements

Potential improvements:
- [x] ~~Persistent memory across sessions~~ ✅ Implemented via Store API
- [ ] Different evaluation strategies (multiple judges, voting)
- [ ] Fine-tuned models for each component
- [ ] Parallel actor attempts with selection
- [ ] Adaptive max_attempts based on query complexity
- [ ] Memory summarization for long reflexion histories
- [ ] Memory pruning/cleanup strategies
- [ ] Similarity-based memory retrieval (semantic search)
- [ ] Memory analytics and insights dashboard

## Persistence

The reflexion agent uses LangGraph API's built-in persistence layer. When deployed via LangGraph:
- State is automatically persisted between invocations
- Thread IDs track conversation history
- No custom checkpointer needed (handled by the platform)

For local testing without LangGraph API, you may need to add a checkpointer manually in your test scripts.

## E2B Sandbox Integration

The Reflexion Coding Agent uses **E2B Code Interpreter** for secure, isolated code execution:

### How It Works:

1. **Agent Initialization**: Evaluator ReAct agent receives the evaluation task
2. **Code Extraction**: Agent uses `extract_code_from_response` tool to get code
3. **Code Loading**: Agent uses `execute_code_in_sandbox` to verify code loads
4. **Test Design & Execution**: Agent autonomously:
   - Decides what tests are needed
   - Writes test code with assertions
   - Calls `run_test_in_sandbox` for each test
   - Observes results and adjusts testing strategy
5. **Iterative Testing**: Agent can run multiple tests, analyze failures, add more tests
6. **Verdict Generation**: Agent provides final verdict based on all test results

### Configuration:

Set the `E2B_API_KEY` environment variable:

```bash
# In .env file
E2B_API_KEY=your_e2b_api_key_here
```

### Benefits:

- **Autonomous Testing**: ReAct agent decides its own testing strategy
- **Real Execution**: No simulation - code actually runs in E2B
- **Secure**: Isolated sandbox environment
- **Accurate**: Catches real runtime errors, not just logic issues
- **Comprehensive**: Agent iteratively tests until satisfied
- **Adaptive**: Agent can add more tests based on initial results
- **Fast**: E2B provides quick sandbox provisioning

### Example ReAct Agent Test Execution:

```python
# Agent thinks: "I need to test this merge_intervals function"

# Agent action 1: Extract code
code = agent.use_tool("extract_code_from_response", response)

# Agent action 2: Load code  
result = agent.use_tool("execute_code_in_sandbox", code)
# Observation: Code loaded successfully

# Agent thinks: "Let me test with empty input first"
# Agent action 3: Run test
test1 = agent.use_tool("run_test_in_sandbox", 
    "result = merge_intervals([]); assert result == []",
    "test_empty")
# Observation: PASSED ✓

# Agent thinks: "Now test overlapping intervals"
# Agent action 4: Run test
test2 = agent.use_tool("run_test_in_sandbox",
    "result = merge_intervals([(1,3),(2,6)]); assert result == [(1,6)]",
    "test_overlap")
# Observation: FAILED - AssertionError: Expected [(1,6)], got [(1,3), (2,6)]

# Agent thinks: "Found a bug, let me test more edge cases..."
# Agent continues testing autonomously...
```

## Dependencies

- **LangGraph**: Graph orchestration and state management
- **LangChain**: LLM integration and message handling  
- **E2B Code Interpreter** (`e2b-code-interpreter==2.2.0`): Secure code execution sandbox
  - Import: `from e2b_code_interpreter import Sandbox`
  - Usage: `sandbox.run_code(code)`
- **Pydantic**: Structured outputs and validation
- **Shared modules**: `llm`, `logger` from project's `shared/`

## License

Part of the Seer project.

