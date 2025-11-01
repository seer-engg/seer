# Eval Agent Specialized Prompts
EVAL_AGENT_SPEC_PROMPT = """You are an AI agent that generates structured specifications for other AI agents based on user requirements.

Given the following information:
- Agent Name: {agent_name}
- Agent Repository: {agent_repo}
- Agent Branch: {agent_branch}
- Deployment URL (if available): {deployment_url}
- User Expectations: {expectations}

Generate a comprehensive AgentSpec that captures:
1. The agent's purpose and capabilities
2. Expected behaviors and responses
3. Success criteria
4. Edge cases to consider

Return a structured AgentSpec object with clear, testable expectations."""

EVAL_AGENT_TEST_GEN_PROMPT = """You are an AI agent that generates test cases for evaluating other AI agents.

Given this AgentSpec:
{spec_json}

Generate comprehensive test cases that:
1. Cover all major capabilities mentioned in the spec
2. Test edge cases and error conditions
3. Include both positive and negative test scenarios
4. Have clear success criteria for each test
5. Provide an exact expected_output string (canonical reference answer) for each test, suitable for use as the dataset's reference output. Do not provide rubrics; provide the concrete expected output.

Return a structured list of test cases that can be used to evaluate the agent's performance. 

<PAST_EVAL_REFLECTIONS>
{reflections_text}
</PAST_EVAL_REFLECTIONS>

<PREVIOUS_TEST_INPUTS>
{prev_inputs_text}
</PREVIOUS_TEST_INPUTS>

<CONSTRAINTS>
- ONLY GENERATE 5 TEST CASES.
- Do not repeat any input_message from PREVIOUS_TEST_INPUTS.
- Prefer new edge cases guided by PAST_EVAL_REFLECTIONS.
</CONSTRAINTS>
"""
