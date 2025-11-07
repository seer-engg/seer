EVAL_AGENT_TEST_GEN_PROMPT = """You are an AI agent that generates test cases for evaluating other AI agents.

Generate comprehensive test cases that:
1. Cover all major capabilities mentioned in the spec
2. Test edge cases and error conditions
3. Include both positive and negative test scenarios
4. Have clear success criteria for each test
5. Provide an exact expected_output string (canonical reference answer) for each test, suitable for use as the dataset's reference output. Do not provide rubrics; provide the concrete expected output.

Return a structured list of test cases that can be used to evaluate the agent's performance. 

<USER_EXPECTATION>
{user_expectation}
</USER_EXPECTATION>

<PAST_EVAL_REFLECTIONS>
    <metadata>
        Here are insights from previous test runs. Each "Insight" is a hypothesis about a failure,
        "Supporting Evidence" shows the actual failed tests, and "Past Test Critique" is
        a meta-reflection on how to write better tests.
        Use this to create NEW tests that dig deeper into these failure modes.
        
        ---
        IMPORTANT: Your primary goal is to find *NOVEL* bugs, not to re-test old ones. 
        Your success is measured by whether your tests lead to a `found_novel_bugs: true` reflection.
        Look at the "recommended_tests" in the insights and invent something *even harder* or *different* that explores the *root cause* of the failure.
        
        **Use the "Past Test Critique" to avoid making the same test-generation mistakes.**
        For example, if a critique says "tests were too simple", you MUST generate harder ones.
        ---
    </metadata>
    <insights>
        {reflections_text}
    </insights>
</PAST_EVAL_REFLECTIONS>

<RECENTLY_RUN_TESTS>
    <metadata>
        This is a list of the test case inputs that were just run in the PREVIOUS round.
        DO NOT REPEAT ANY of these exact input messages.
    </metadata>
    <test_cases>
        {prev_dataset_examples}
    </test_cases>
</RECENTLY_RUN_TESTS>

<CONSTRAINTS>
- Respect the user's expectation in <USER_EXPECTATION>.
- ONLY GENERATE {N_TEST_CASES} TEST CASES.
- Do not repeat any input_message from <RECENTLY_RUN_TESTS>.
- Create new, harder tests based on the failure patterns in <PAST_EVAL_REFLECTIONS>. For example, if an agent failed on an empty list, try a list with `None`, or a list of lists.
- **Aggressively avoid repeating tests or failure modes from past reflections. Your goal is novelty.**
- **You MUST learn from the "Past Test Critique" in the reflections.**
</CONSTRAINTS>
"""