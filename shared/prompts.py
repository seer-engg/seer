"""
System prompts for agents

NOTE: Agent-specific system prompts are now defined in their respective graph.py files:
- Customer Success: agents/customer_success/graph.py (SYSTEM_PROMPT)
- Eval Agent: agents/eval_agent/graph.py (SYSTEM_PROMPT)

This file contains prompts used by the agent tools (spec generation, test generation, judging).
"""


# Prompts for Eval Agent Tools

EVAL_AGENT_SPEC_PROMPT = """User's expectations for their agent:
{expectations}

Agent being evaluated:
- Name: {agent_name}
- URL: {agent_url}

Convert these expectations into a structured specification with:
1. Clear description of what the agent does
2. List of specific expectations with priority levels (must/should/nice-to-have)

Return ONLY valid JSON matching this schema:
{{
  "name": "agent_name",
  "version": "1.0.0",
  "description": "...",
  "expectations": [
    {{"description": "...", "context": "...", "priority": "must"}},
    ...
  ]
}}"""


EVAL_AGENT_TEST_GEN_PROMPT = """Generate comprehensive test cases for this agent specification:

{spec_json}

For each expectation, create 2-3 test cases that:
1. Test the happy path
2. Test edge cases
3. Test error handling

Each test should have:
- expectation_ref: Which expectation is being tested
- input_message: What to send to the agent
- expected_behavior: How the agent should respond
- success_criteria: Specific criteria for judging success

Return ONLY valid JSON:
{{
  "test_cases": [
    {{
      "expectation_ref": "...",
      "input_message": "...",
      "expected_behavior": "...",
      "success_criteria": "..."
    }},
    ...
  ]
}}"""


EVAL_AGENT_JUDGE_PROMPT = """Evaluate if the agent met the success criteria.

Test Case:
- Input: {input_message}
- Expected Behavior: {expected_behavior}
- Success Criteria: {success_criteria}

Actual Agent Output:
{actual_output}

Judge the response:
1. Did it address the core intent?
2. Is the response appropriate and helpful?
3. Does it match the expected behavior?

Respond with ONLY valid JSON:
{{
  "passed": true/false,
  "score": 0.0 to 1.0,
  "reasoning": "detailed explanation"
}}"""
