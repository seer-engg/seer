import json
from typing import Dict, List, Optional, Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ConfigDict

from agents.eval_agent.constants import (
    LLM,
    N_TEST_CASES,
)
from agents.eval_agent.models import EvalAgentPlannerState
from shared.logger import get_logger
from shared.resource_utils import format_resource_hints
from shared.tools import ToolEntry
from shared.tools.schema_formatter import format_tool_schemas_for_llm
from shared.schema import (
    DatasetExample,
)

logger = get_logger("eval_agent.plan.generate_evals")


class _AgentTestGenerationOutput(BaseModel):
    """The final output from the test generation agent."""
    model_config = ConfigDict(extra="forbid")
    dataset_examples: List[DatasetExample] = Field(
        description=f"A list of {N_TEST_CASES} generated test cases."
    )

AGENTIC_GENERATOR_SYSTEM_PROMPT = """### PROMPT: TEST_CASE_GENERATOR_AGENT (EVAL_AGENT) ###
You are an expert adversarial QA agent using **3-PHASE TESTING ARCHITECTURE**.

**CRITICAL**: Tests MUST work on an EMPTY CANVAS. Create ALL test data from scratch in Phase 1.

**1. CONTEXT:**
* **System Goal:** {system_goal_description}
* **Agent Weaknesses (from past reflections):** {reflections_text}
* **Recently Run Tests (Do Not Repeat):** {prev_dataset_examples}
* **Available Resources:** {resource_hints}

**2. AVAILABLE TOOLS & SCHEMAS (Your "Contract"):**
Here is the "contract" for each tool. Your generated actions MUST adhere to these schemas.
{formatted_tool_schemas}

**3. YOUR TASK:**
Generate {n_tests} `DatasetExample` objects using **3-PHASE TESTING**.

**3-PHASE TEST STRUCTURE:**

**Phase 1: PROVISION (provision_actions)**
- Create ALL test data (PRs, labels, tasks, etc.)
- Assign resources to variables for Phase 3
- NO assertions in this phase
- Example: Create PR, add labels, create Asana task

**Phase 2: INVOKE (input_message)**
- Human-readable scenario sent to agent
- Example: "A PR was merged that mentions Asana tickets, but one of its labels was deleted. Please sync."
- The agent will be invoked automatically with this message

**Phase 3: ASSERT (assert_actions)**
- Verify final state changes
- Use assertions (assert_field + assert_expected)
- Check if resources were updated correctly
- Example: Verify Asana task was marked complete

**MATHEMATICAL GUARANTEE**: 
- Phase 1 MUST create everything Phase 2's scenario requires
- Phase 3 MUST verify observable changes in the system
- NEVER assume existing data exists

For each test case:
1.  **Reason:** What weakness are you exploiting? What edge case?
2.  **Design 3 Phases:**
    * Phase 1: What resources to create?
    * Phase 2: What scenario to send to agent?
    * Phase 3: What to verify?
3.  **Create the Test Case:** Build a `DatasetExample` with:
    * `example_id`: Leave as empty string "" (will be auto-generated)
    * `reasoning`: Why this test is valuable and what it targets
    * `input_message`: Human-readable scenario for Phase 2
    * `expected_output`: An `ExpectedOutput` object with:
      - `provision_actions`: List of actions to create test data
      - `assert_actions`: List of actions to verify final state
      - `expected_actions`: (optional) What agent SHOULD do
    * `status`: "active"

**Requirements:**
- Tool names MUST match exactly from the available tools list
- Params should be provided as dict/object with all required fields
- Use [var:variable_name.field] to reference previous action outputs
- Use [resource:resource_name.field] to reference pre-provisioned resources
- EVERY test MUST have provision_actions and assert_actions
- NO angle bracket placeholders (<owner>, <repo>) - use [var:...] or [resource:...]
- Generate {n_tests} diverse, high-quality test cases
- Return as `_AgentTestGenerationOutput` with your `dataset_examples` list
"""

def _create_constrained_schemas(available_tool_names: List[str], tool_entries: Dict[str, ToolEntry]):
    """
    Create dynamic schemas with individual models for each tool.
    
    This generates a separate Pydantic model for each tool with its actual parameters
    as direct fields (not nested under 'params'). This allows OpenAI's strict mode
    to work properly while preventing tool hallucinations.
    """
    from typing import Literal, Union
    from pydantic import create_model
    
    # Add system.wait to valid tools
    all_tools = available_tool_names + ["system.wait"]
    
    # Create individual model for each tool
    tool_models = []
    
    for tool_name in all_tools:
        if tool_name == "system.wait":
            # Special case for system.wait
            tool_model = create_model(
                f"Action_{tool_name.replace('.', '_').replace('-', '_')}",
                __config__=ConfigDict(extra="forbid"),
                tool=(Literal[tool_name], Field(description=f"Tool name: {tool_name}")),
                seconds=(int, Field(description="Number of seconds to wait")),
                assign_to_var=(str, Field(default="", description="Variable name (or empty)")),
                assert_field=(str, Field(default="", description="Field to check (or empty)")),
                assert_expected=(str, Field(default="", description="Expected value (or empty)")),
            )
            tool_models.append(tool_model)
            continue
        
        # Get tool schema from entries
        tool_entry = tool_entries.get(tool_name.lower())
        if not tool_entry or not tool_entry.pydantic_schema:
            # Fallback: create model with no specific params if schema missing
            logger.warning("No schema found for tool %s, skipping", tool_name)
            continue
        
        schema = tool_entry.pydantic_schema
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        # Build field definitions for this tool
        field_defs = {
            "tool": (Literal[tool_name], Field(description=f"Tool name: {tool_name}")),
            "assign_to_var": (str, Field(default="", description="Variable name (or empty)")),
            "assert_field": (str, Field(default="", description="Field to check (or empty)")),
            "assert_expected": (str, Field(default="", description="Expected value (or empty)")),
        }
        
        # Add each parameter as a direct field
        for param_name, param_schema in properties.items():
            param_type = param_schema.get("type", "string")
            param_desc = param_schema.get("description", "")
            is_required = param_name in required
            
            # Map JSON schema types to Python types
            # Note: For strict mode, we must avoid Any in arrays/dicts where possible
            if param_type == "string":
                py_type = str
            elif param_type == "integer":
                py_type = int
            elif param_type == "number":
                py_type = float
            elif param_type == "boolean":
                py_type = bool
            elif param_type == "array":
                # Check if items schema specifies a type
                items_schema = param_schema.get("items", {})
                items_type = items_schema.get("type")
                
                if items_type == "string":
                    py_type = List[str]
                elif items_type == "integer":
                    py_type = List[int]
                elif items_type == "object":
                    py_type = List[Dict[str, Any]]
                else:
                    # Fallback: use List[str] as safest default for strict mode
                    # This works for most Asana/GitHub params which are string arrays
                    py_type = List[str]
            elif param_type == "object":
                # Dict type
                py_type = Dict[str, str]  # Safer default for strict mode
            else:
                # Fallback to string for unknown types (strict mode compatible)
                py_type = str
            
            # Create field with or without default
            if is_required:
                field_defs[param_name] = (py_type, Field(description=param_desc))
            else:
                # Optional field - use None as default
                field_defs[param_name] = (Optional[py_type], Field(default=None, description=param_desc))
        
        # Create the model dynamically
        model_name = f"Action_{tool_name.replace('.', '_').replace('-', '_')}"
        tool_model = create_model(
            model_name,
            __config__=ConfigDict(extra="forbid"),
            **field_defs
        )
        tool_models.append(tool_model)
    
    # Create Union of all tool models
    if len(tool_models) == 0:
        raise ValueError("No valid tool models created")
    elif len(tool_models) == 1:
        ConstrainedActionStep = tool_models[0]
    else:
        ConstrainedActionStep = Union[tuple(tool_models)]  # type: ignore
    
    # Create constrained ExpectedOutput
    class ConstrainedExpectedOutput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        
        provision_actions: Optional[List[ConstrainedActionStep]] = Field(
            None,
            description="Phase 1: Actions to create test data"
        )
        expected_actions: Optional[List[ConstrainedActionStep]] = Field(
            None,
            description="Phase 2: Expected tool calls the agent should make"
        )
        assert_actions: List[ConstrainedActionStep] = Field(
            ...,
            description="Phase 3: Actions to verify final state - REQUIRED"
        )
    
    # Create constrained DatasetExample
    class ConstrainedDatasetExample(BaseModel):
        model_config = ConfigDict(extra="forbid")
        
        example_id: str = Field(
            default="",
            description="Leave empty - will be auto-generated"
        )
        reasoning: str = Field(
            ...,
            description="Why is this test important? What will it test?"
        )
        input_message: str = Field(
            ...,
            description="Input message for the agent"
        )
        expected_output: ConstrainedExpectedOutput = Field(...)
        status: str = Field(default="active")
    
    # Create constrained output wrapper
    class ConstrainedOutput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        dataset_examples: List[ConstrainedDatasetExample] = Field(...)
    
    return ConstrainedOutput


async def _invoke_agentic_llm(
    raw_request: str,
    reflections_text: str,
    prev_dataset_examples: str, 
    available_tool_names: List[str],
    tool_entries: Dict[str, ToolEntry],
    resource_hints: str,
    n_tests: int,
) -> List[DatasetExample]:
    
    logger.info("plan.test-llm: Starting test generation for %d tests...", n_tests)    
    # Format tool schemas using shared formatter
    formatted_tool_schemas = format_tool_schemas_for_llm(tool_entries, available_tool_names)
    
    system_prompt = AGENTIC_GENERATOR_SYSTEM_PROMPT.format(
        n_tests=n_tests,
        system_goal_description=raw_request,
        reflections_text=reflections_text,
        prev_dataset_examples=prev_dataset_examples,
        resource_hints=resource_hints,
        formatted_tool_schemas=formatted_tool_schemas,
    )

    # Use structured output with CONSTRAINED tool names
    logger.info("Invoking test generator with constrained structured output...")
    
    try:
        # Create dynamic schema that constrains tool names to available tools
        # Each tool gets its own Pydantic model with exact parameters
        ConstrainedOutput = _create_constrained_schemas(available_tool_names, tool_entries)
        
        # Create a structured LLM with constrained schema
        # Use STRICT MODE now that we have proper flat schemas for each tool
        structured_llm = LLM.with_structured_output(ConstrainedOutput, method="json_schema", strict=True)
        
        # Single LLM call with system prompt + user message
        output = await structured_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Please generate {n_tests} test cases.")
        ])
        
        # Convert constrained models back to regular DatasetExample objects
        dataset_examples = []
        for constrained_example in output.dataset_examples:
            # Convert to dict
            example_dict = constrained_example.model_dump()
            
            # Convert flat tool parameters to nested params structure
            # LLM returns: {"tool": "X", "owner": "...", "repo": "...", "assign_to_var": "..."}
            # We need: {"tool": "X", "params": "{\"owner\": \"...\", \"repo\": \"...\"}", "assign_to_var": "..."}
            def convert_to_nested_params(actions_list):
                if not actions_list:
                    return actions_list
                
                converted = []
                for action in actions_list:
                    # Extract tool and metadata fields
                    tool = action.get('tool', '')
                    assign_to_var = action.get('assign_to_var', '')
                    assert_field = action.get('assert_field', '')
                    assert_expected = action.get('assert_expected', '')
                    
                    # Extract all other fields as params
                    params = {}
                    for key, value in action.items():
                        if key not in ['tool', 'assign_to_var', 'assert_field', 'assert_expected']:
                            # Only include non-None values
                            if value is not None:
                                params[key] = value
                    
                    # Build new action structure
                    converted_action = {
                        'tool': tool,
                        'params': json.dumps(params),  # Serialize params dict to JSON string
                        'assign_to_var': assign_to_var,
                        'assert_field': assert_field,
                        'assert_expected': assert_expected,
                    }
                    converted.append(converted_action)
                
                return converted
            
            if example_dict.get('expected_output'):
                eo = example_dict['expected_output']
                if eo.get('provision_actions'):
                    eo['provision_actions'] = convert_to_nested_params(eo['provision_actions'])
                if eo.get('expected_actions'):
                    eo['expected_actions'] = convert_to_nested_params(eo['expected_actions'])
                if eo.get('assert_actions'):
                    eo['assert_actions'] = convert_to_nested_params(eo['assert_actions'])
            
            example = DatasetExample(**example_dict)
            
            # Generate UUID if example_id is missing or empty
            if not example.example_id or example.example_id == "":
                example.example_id = str(uuid4())
            example.status = "active"
            
            dataset_examples.append(example)
            
        return dataset_examples[:n_tests]
        
    except Exception as e:
        logger.error("Agentic test generator failed: %s", e)
        raise e

async def agentic_eval_generation(state: EvalAgentPlannerState) -> dict:
    agent_name = state.context.github_context.agent_name
    
    # Get just the inputs from the most recent run
    previous_inputs = [res.dataset_example.input_message for res in state.latest_results]
    
    resource_hints = format_resource_hints(state.context.mcp_resources)

    logger.info("Using 'agentic' (structured output) test generation.")
    dataset_examples = await _invoke_agentic_llm(
        raw_request=state.context.user_context.raw_request,
        reflections_text=state.reflections_text,
        prev_dataset_examples=json.dumps(previous_inputs, indent=2),
        available_tool_names=state.available_tools,
        tool_entries=state.tool_entries,
        resource_hints=resource_hints,
        n_tests=N_TEST_CASES,
    )

    logger.info("plan.generate: produced %d tests (agent=%s)", len(dataset_examples), agent_name)
    return {
        "dataset_examples": dataset_examples,
    }
