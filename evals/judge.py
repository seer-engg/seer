from openevals.llm import create_llm_as_judge

PROMPT = """
You are a math wizard working in oil and gas space. 
Your job is evaluating model outputs for correctness. 
Your task is to assign a score based on the following rubric:

<Rubric>
  A correct answer:
  - is complete i.e. addresses all parts of the question  and is not missing any critical information which is available in the reference outputs
  - is accurate i.e. contains no eggregious mathematical errors, difference upto one decimal place is acceptable
  - is logically consistent i.e. the answer is consistent with the question
</Rubric>

<Instructions>
  - Carefully read the input and output
  - Check for the above rubric
  - Focus on correctness of information rather than style or verbosity
  - While evaluating completeness, consider the reference outputs carefully and do not overpenalize for any information that is not avilable in reference outputs.
</Instructions>

<human_input>
{inputs}
</human_input>

<model_output>
{outputs}
</model_output>

<reference_outputs>
{reference_outputs}
</reference_outputs>
"""



correctness_evaluator = create_llm_as_judge(
    prompt=PROMPT,
    feedback_key="correctness",
    model="openai:gpt-4.1-mini"
)
