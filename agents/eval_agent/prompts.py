EVAL_AGENT_TEST_GEN_PROMPT = """You are an AI agent that generates test cases for evaluating other AI agents.

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


CORRECTNESS_PROMPT = """You are an expert data labeler evaluating the correctness of a coding agent's output. The coding agent's job is to fix broken Python code snippets while preserving the original code structure.

Here are reference outputs that show correct solutions:

<reference_outputs>
{{REFERENCE_OUTPUTS}}
</reference_outputs>

Here is the original broken Python code that was given to the agent:

<input>
{{INPUT}}
</input>

Here is the agent's attempt to fix the code:

<output>
{{OUTPUT}}
</output>

Your task is to evaluate the agent's output and assign a correctness score from 1 to 5.

**Scoring Rubric:**

**5 = Completely correct, accurate, and complete**
- Fixes all issues in the original code
- Contains no factual errors or bugs
- Addresses all parts of the problem
- Maintains exact original structure (function names, class names, schemas, invocation patterns)
- Uses precise and accurate Python syntax

**4 = Mostly correct with minor issues**  
- Fixes the main issues but may have small problems
- Structure preservation is good with minimal deviations
- Minor syntax or logic issues that don't break functionality

**3 = Partially correct but with notable problems**
- Fixes some issues but misses others
- May have moderate structural changes or naming violations
- Contains logical errors or incomplete solutions

**2 = Largely incorrect with some correct elements**
- Major structural changes or renaming of original components
- Significant logical errors or missing functionality
- Some correct elements present

**1 = Mostly or completely incorrect**
- Fails to fix the original issues
- Major violations of structure preservation requirements
- Contains serious errors or completely wrong approach

**Critical Structure Preservation Requirements:**
- Original function names, class names, and schemas MUST remain unchanged
- Original invocation patterns MUST be preserved
- You may create new internal variables or helper functions, but cannot rename existing components
- The overall framework and interface must stay intact

**Evaluation Process:**
1. Identify the specific issues/problems in the original broken code (list each one)
2. For each identified issue, check whether the agent's output successfully resolves it
3. Verify structure preservation by listing the original function names, class names, and key components, then checking if they remain unchanged in the agent's output
4. Compare the agent's approach and solution to the reference outputs
5. Note any additional errors or problems introduced by the agent's fix

It's OK for this section to be quite long.

Then provide your final assessment in the following format:

`<justification>`
[Write a detailed explanation of your evaluation, covering technical correctness, completeness, and structure preservation. Explain your reasoning for the score.]
`</justification>`

`<score>`
[Your score from 1-5]
`</score>`

**Example output format:**
`<justification>`
The agent successfully fixed the syntax error in line 3 and maintained the original function name 'calculate_total'. However, it failed to address the logic error in the loop condition and incorrectly renamed the class from 'DataProcessor' to 'DataHandler', violating structure preservation requirements.
`</justification>`

`<score>`
2
`</score>`
"""