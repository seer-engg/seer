import asyncio
import json
from uuid import uuid4
from typing import List, Optional, Dict, Any

from graph_db import NEO4J_GRAPH
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ConfigDict
from agents.eval_agent.constants import N_TEST_CASES
from agents.eval_agent.models import EvalAgentPlannerState

from shared.resource_utils import format_resource_hints
from shared.logger import get_logger
from shared.schema import DatasetExample


logger = get_logger("eval_agent.plan.genetic_eval_generation")

COMMON_INSTRUCIONS = """\n\n
Create one new `DatasetExample` using **3-PHASE TESTING ARCHITECTURE**.

**CRITICAL**: Tests MUST work on an EMPTY CANVAS. Create ALL test data from scratch.

**Available Tools:**
{formatted_tool_schemas}

**Available Resources:**
{resource_hints}

**CRITICAL - Variable Syntax:**
- Reference previous action output: [var:variable_name.field]
- Reference pre-provisioned resource: [resource:resource_name.field]
- NEVER use angle brackets: <owner>, <repo>, <pr_number> will FAIL
- Example array access: [var:prs.data.pull_requests.0.number]

**Action Fields (all required):**
- `tool`: Exact tool name from list above
- `params`: Dict/object with ALL required params
- `assign_to_var`: Variable name (or "")
- `assert_field`: Field to check (or "")
- `assert_expected`: Expected value as string (or "")

**3-PHASE TEST STRUCTURE:**

**Phase 1: PROVISION (provision_actions)** - Create test data
- Create PRs, labels, tasks, etc.
- Assign resources to variables
- NO assertions in this phase

**Phase 2: INVOKE (input_message)** - Agent is invoked automatically
- Human-readable scenario describing what needs to be done
- Example: "A PR was merged that mentions Asana tickets, but one of its labels was deleted. Please sync."

**Phase 3: ASSERT (assert_actions)** - Verify final state
- Check if resources were updated correctly
- Use assertions (assert_field + assert_expected)
- Verify observable state changes

**MATHEMATICAL GUARANTEE**: 
- Phase 1 MUST create everything Phase 2's scenario requires
- Phase 3 MUST verify observable changes (not just "did agent call tool X")
- NEVER assume existing data - create it!

**Example (3-phase testing):**
{{
  "reasoning": "Tests handling of deleted GitHub labels in PR-Asana sync",
  "input_message": "A PR was merged that mentions Asana tickets, but one of its labels was deleted. Please sync the Asana tasks.",
  "expected_output": {{
    "provision_actions": [
      {{
        "tool": "GITHUB_CREATE_PULL_REQUEST",
        "params": "{{\\"title\\": \\"Fix bug\\", \\"base\\": \\"main\\", \\"head\\": \\"bugfix\\", \\"body\\": \\"Fixes ASANA-123\\", \\"owner\\": \\"[resource:github_owner]\\", \\"repo\\": \\"[resource:github_repo]\\"}}",
        "assign_to_var": "test_pr",
        "assert_field": "",
        "assert_expected": ""
      }},
      {{
        "tool": "GITHUB_ADD_LABELS_TO_AN_ISSUE",
        "params": "{{\\"owner\\": \\"[resource:github_owner]\\", \\"repo\\": \\"[resource:github_repo]\\", \\"issue_number\\": \\"[var:test_pr.number]\\", \\"labels\\": [\\"bug\\"]}}",
        "assign_to_var": "",
        "assert_field": "",
        "assert_expected": ""
      }},
      {{
        "tool": "ASANA_CREATE_TASK",
        "params": "{{\\"name\\": \\"ASANA-123\\", \\"workspace_gid\\": \\"[resource:asana_workspace]\\"}}",
        "assign_to_var": "asana_task",
        "assert_field": "",
        "assert_expected": ""
      }}
    ],
    "assert_actions": [
      {{
        "tool": "ASANA_GET_TASK",
        "params": "{{\\"task_gid\\": \\"[var:asana_task.gid]\\"}}",
        "assign_to_var": "",
        "assert_field": "completed",
        "assert_expected": "true"
      }}
    ]
  }}
}}
"""



MUTATION_PROMPT = """The agent PASSED this test - make it harder.

**Parent Test:**
{parent_test_json}

**Known Agent Weaknesses:**
{reflections_text}

Mutate the parent test to exploit weaknesses. Add edge cases, delays, or unexpected sequences.
""" + COMMON_INSTRUCIONS

CROSSOVER_PROMPT = """Both parents found bugs - combine their failure modes.

**Parent 1:**
{parent_1_json}

**Parent 2:**
{parent_2_json}

**Known Agent Weaknesses:**
{reflections_text}

Create a hybrid test that combines both parents' attack vectors.
""" + COMMON_INSTRUCIONS

NEW_TEST_PROMPT = """Create a novel, complex test to make the agent fail.

**Goal:**
{raw_request}

**Known Weaknesses:**
{reflections_text}

**Don't Repeat:**
{prev_dataset_examples}

Attack vectors: race conditions, invalid data, rapid updates, edge cases.
""" + COMMON_INSTRUCIONS


def _create_constrained_schema_genetic(available_tool_names: List[str], tool_entries: Dict[str, Any]):
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
    
    # Create constrained output wrapper (single example for genetic)
    class ConstrainedTestGenerationOutput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        dataset_example: ConstrainedDatasetExample = Field(...)
    
    return ConstrainedTestGenerationOutput


async def genetic_eval_generation(state: EvalAgentPlannerState) -> dict:
    logger.info("Starting genetic test generation with constrained tool names")

    tool_names = list(dict.fromkeys(state.available_tools or [])) + ["system.wait"]
    
    # Create constrained schema that prevents tool hallucinations
    # Each tool gets its own Pydantic model with exact parameters
    ConstrainedOutput = _create_constrained_schema_genetic(state.available_tools or [], state.tool_entries)
    
    # Setup LLM with constrained schema
    # Use STRICT MODE now that we have proper flat schemas for each tool
    llm = ChatOpenAI(
        model="gpt-5-codex",
        use_responses_api=True,
        output_version="responses/v1",
        reasoning={"effort": "low"},
    ).with_structured_output(ConstrainedOutput, method="json_schema", strict=True)
    
    # TODO: Add make this simliar to the agentic_eval_generation.py
    formatted_schemas = []
    resource_hints = format_resource_hints(state.context.mcp_resources)
    
    prev_inputs = [r.dataset_example.input_message for r in state.latest_results]
    
    new_generation: List[DatasetExample] = []
    
    async def get_parents(passed: bool, k: int = 1) -> List[dict]:
        """Get parent tests from Neo4j."""
        result = await asyncio.to_thread(
            NEO4J_GRAPH.query,
            """
            MATCH (ex:DatasetExample {user_id: $user_id, agent_name: $agent_name, status: 'active'})
            MATCH (ex)-[:WAS_RUN_IN]->(res:ExperimentResult {passed: $passed})
            RETURN ex.reasoning as reasoning, ex.input_message as input_message, 
                   ex.expected_output as expected_output, ex.example_id as example_id
            LIMIT $k
            """,
            params={
                "user_id": state.context.user_context.user_id,
                "agent_name": state.context.github_context.agent_name,
                "passed": passed,
                "k": k
            }
        )
        parsed = []
        for r in result:
            try:
                r_dict = dict(r)
                r_dict['expected_output'] = json.loads(r_dict['expected_output'])
                parsed.append(r_dict)
            except Exception:
                continue
        return parsed

    async def generate(prompt_template: str, **kwargs):
        """Generate a test from a prompt."""
        prompt = prompt_template.format(
            formatted_tool_schemas=formatted_schemas,
            resource_hints=resource_hints,
            reflections_text=state.reflections_text,
            **kwargs
        )
        output = await llm.ainvoke(prompt)
        
        # Convert constrained model back to regular DatasetExample
        example_dict = output.dataset_example.model_dump()
        
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
        new_generation.append(example)

    # Get history
    fit_parents = await get_parents(passed=False, k=2)
    unfit_parents = await get_parents(passed=True, k=1)
    
    # Generate tests
    if len(fit_parents) == 2 and len(new_generation) < N_TEST_CASES:
        await generate(CROSSOVER_PROMPT, parent_1_json=json.dumps(fit_parents[0]), 
                      parent_2_json=json.dumps(fit_parents[1]))
    
    if unfit_parents and len(new_generation) < N_TEST_CASES:
        await generate(MUTATION_PROMPT, parent_test_json=json.dumps(unfit_parents[0]))
    
    while len(new_generation) < N_TEST_CASES:
        await generate(NEW_TEST_PROMPT, raw_request=state.context.user_context.raw_request,
                      prev_dataset_examples=json.dumps(prev_inputs))

    # Finalize
    for example in new_generation:
        if not example.example_id:
            example.example_id = str(uuid4())
        example.status = "active"

    logger.info("Generated %d tests", len(new_generation))
    return {"dataset_examples": new_generation[:N_TEST_CASES]}
