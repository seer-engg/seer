import json
import hashlib
import uuid
import traceback
import os
from datetime import datetime

from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langgraph.types import Command
from langchain.agents import create_agent
from langsmith import Client, evaluate
from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk import get_sync_client
from langchain.agents.middleware import TodoListMiddleware
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

from agents.eval_agent.prompts import (
    EVAL_AGENT_PROMPT,
    EVAL_AGENT_SPEC_PROMPT,
    EVAL_AGENT_TEST_GEN_PROMPT_WITHOUT_REFLECTIONS,
)
from agents.eval_agent.models import AgentSpec, TargetAgentConfig, GeneratedTests, EvalAgentState, GeneratedTestCase
from shared.llm import get_llm
from shared.logger import get_logger
from shared.error_handling import create_success_response, create_error_response


# Get logger for eval agent
logger = get_logger('eval_agent')
_LLM = get_llm(temperature=0.0)
_LANGSMITH_CLIENT = Client(api_key=os.getenv("LANGSMITH_API_KEY"))


@tool
def think(thought: str) -> str:
    """
    Think tool for eval_agent: log internal reflection; no external side effects.
    """
    logger.info(f"THINK: {thought}")
    return json.dumps({"success": True, "thought": thought})


@tool
def generate_evals(runtime: ToolRuntime) -> Command:
    """
    should take TargetAgentConfig from state and generate evals
    Single-step eval generation: parse request, build internal spec, generate tests,
    cache suite in state, and return a compact columnar table view.
    """
    target_agent_config = runtime.state.get("target_agent_config")
    agent_name = target_agent_config.graph_name
    agent_url = target_agent_config.url
    expectations = target_agent_config.expectations

    # Build AgentSpec internally (not shown to user)
    spec_llm = _LLM.with_structured_output(AgentSpec)
    spec_obj: AgentSpec = spec_llm.invoke(
        EVAL_AGENT_SPEC_PROMPT.format(
            expectations=expectations,
            agent_name=agent_name,
            agent_url=agent_url,
        )
    )

    # Generate tests
    generated: GeneratedTests = _LLM.with_structured_output(GeneratedTests).invoke(
        EVAL_AGENT_TEST_GEN_PROMPT_WITHOUT_REFLECTIONS.format(spec_json=spec_obj.model_dump_json())
    )
    assert len(generated.test_cases) == 5, f"Generated {len(generated.test_cases)} test cases, expected 5"

    # Build deterministic test cases and table view
    test_cases: list[dict] = []
    for idx, tc in enumerate(generated.test_cases):
        content_hash = hashlib.md5(f"{tc.expectation_ref}{tc.input_message}".encode()).hexdigest()[:8]
        test_id = f"{idx+1}_{content_hash}"
        test_cases.append({
            "id": test_id,
            "expectation_ref": tc.expectation_ref,
            "input_message": tc.input_message,
            "expected_behavior": tc.expected_behavior,
            "success_criteria": tc.success_criteria,
            "expected_output": getattr(tc, "expected_output", None) or "",
        })

    return Command(update={
        "test_cases": test_cases,
        "messages": [ToolMessage(content=f"Generated {len(test_cases)} test cases for {agent_name}", tool_call_id=runtime.tool_call_id)]
    })


@tool
def parse_eval_request(runtime: ToolRuntime) -> Command:
    """
    Extract agent_name, agent_url, and expectations from the user's message.
    Also updates the state with the extracted information.
    Returns JSON string with these fields.
    """
    messages = runtime.state["messages"]
    human_msgs = [m for m in messages if m.__class__.__name__ == "HumanMessage"]
    last_human_msg = human_msgs[-1]

    if not last_human_msg:
        raise ValueError("No human message found")

    extractor = _LLM.with_structured_output(TargetAgentConfig)
    instruction = (
        "Extract graph_name, url, and expectations from the user's latest message.\n\n"
        "url should begin with http:// or https://"
        "IMPORTANT - graph_name is the name of the graph (NOT ASSISTANT ID WHICH IS A HEX STRING) to evaluate"
    )
    target_agent_config: TargetAgentConfig = extractor.invoke(f"{instruction}\n\nUSER:\n{last_human_msg.content}")

    # RULE - if a tool call is updating the state via command, it must append a ToolMessage with the tool_call_id to the messages list
    return Command(update={
        "messages": [ToolMessage(content=json.dumps(target_agent_config.model_dump()), tool_call_id=runtime.tool_call_id)],
        "target_agent_config": target_agent_config,
    })


@tool
def create_langsmith_dataset(runtime: ToolRuntime) -> Command:
    """Create or read a LangSmith dataset and store its name in state for later use."""
    # If we've already created/selected a dataset for this run, reuse it
    existing = runtime.state.get("dataset_name")
    if existing:
        payload = json.dumps({"dataset_name": existing, "reused": True})
        return Command(update={
            "dataset_name": existing,
            "messages": [ToolMessage(content=payload, tool_call_id=tool_call_id)]
        })

    agent_id = runtime.state.get("target_agent_config").graph_name.replace("/", "_")
    date_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"seer_eval_{agent_id}_{date_tag}"

    try:
        _ = _LANGSMITH_CLIENT.create_dataset(name)
    except Exception:
        raise Exception(f"Failed to create dataset: {name}\n{traceback.format_exc()}")

    # Persist dataset name in state for reuse and acknowledge with ToolMessage
    payload = json.dumps({"dataset_name": name})
    return Command(update={
        "dataset_name": name,
        "messages": [ToolMessage(content=payload, tool_call_id=runtime.tool_call_id)]
    })


@tool
def test_cases_to_langsmith_dataset(runtime: ToolRuntime = None) -> str:
    """Upsert examples into LangSmith dataset from EvalAgentState's test_cases."""
    try:
        test_cases = runtime.state.get("test_cases")
        assert test_cases, f"No cached test cases found in state: {runtime.state}"

        dataset_name = runtime.state.get("dataset_name")
        assert dataset_name, f"No dataset name found in state: {runtime.state}"

        dataset = _LANGSMITH_CLIENT.read_dataset(dataset_name=dataset_name)

        examples = [
            {
                "inputs": {"question": tc.get("input_message", "")},
                "outputs": {"answer": (tc.get("expected_output") or tc.get("expected_behavior", ""))},
            }
            for tc in test_cases
        ]
        if examples:
            _LANGSMITH_CLIENT.create_examples(dataset_id=dataset.id, examples=examples)
        return create_success_response({"dataset_name": dataset_name, "examples": len(examples)})
    except Exception as e:
        return create_error_response(f"Failed to upsert examples: {str(e)}\n{traceback.format_exc()}", e)


# Define an LLM-as-a-judge evaluator to evaluate correctness of the output
def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="openai:o3-mini",
        feedback_key="correctness",
    )
    eval_result = evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    )
    logger.info(f"evaluator result: {eval_result}")
    return eval_result


@tool
def run_langsmith_evaluation(runtime: ToolRuntime = None) -> str:
    """Run LangSmith evaluation over the given dataset, calling the target agent for responses.
    Note: For direct feedback and scores, prefer using `compute_local_evaluation`.
    """
    try:
        target_agent_config = runtime.state.get("target_agent_config")
        target_graph_name = target_agent_config.graph_name
        target_url = target_agent_config.url
        dataset_name = runtime.state.get("dataset_name")

        # graph runs (for example, calls made with .invoke() or .stream()) are stateless
        # create a thread for persistence
        sync_client = get_sync_client(url=target_url)
        thread = sync_client.threads.create()
        thread_cfg = {"configurable": {"thread_id": thread["thread_id"]}}

        # define the remote graph
        remote_graph = RemoteGraph(target_graph_name, url=target_url, client=_LANGSMITH_CLIENT, sync_client=sync_client, distributed_tracing=True)

        # define a runnable function that will be used to evaluate the remote graph
        def persistent_runnable(input_dict: dict) -> dict:
            """
            input will be dict with 'question' key
            output should be dict with 'answer' key
            """
            # logger.info(f"target agent input: {input_dict}")
            question = input_dict.get('question', '')
            result = remote_graph.invoke({"messages": [{"role": "user", "content": question}]}, config=thread_cfg)
            # logger.info(f"target agent response: {result}")
            answer = result.get("messages", [{}])[-1].get("content", "")
            logger.info(f"evaluator input: {question}, answer: {answer}")
            return {"answer": answer}

        # generate a unique experiment prefix which makes it easy to find the experiment in LangSmith
        experiment_prefix = f"seer-{target_graph_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        results = evaluate(
            persistent_runnable,
            data=dataset_name,
            evaluators=[correctness_evaluator],
            experiment_prefix=experiment_prefix,
            client=_LANGSMITH_CLIENT,
            upload_results=True,
        )

        # Block until the evaluation is complete
        results.wait()

        logger.info(f"evaluation results: {results.to_pandas()}")

        return create_success_response({
            "dataset_name": dataset_name,
            "experiment_name": getattr(results, "experiment_name", experiment_prefix),
        })
    except Exception as e:
        return create_error_response(f"Failed to run LangSmith evaluation: {str(e)}\n{traceback.format_exc()}", e)


@tool
def index_conversation(namespace_prefix: str = "conversations", runtime: ToolRuntime = None) -> str:
    """Index the eval agent conversation into the local LangGraph store for semantic search."""
    try:
        # Collect conversation messages
        messages = runtime.state.get("messages") or []
        graph_name = runtime.state.get("target_agent_config").graph_name
        namespace = f"{namespace_prefix}/eval_agent/{graph_name}"

        client = get_sync_client(url="http://127.0.0.1:8002")

        # Upsert each user/ai message individually
        added = 0
        for msg in messages:
            try:
                role = getattr(msg, "type", None) or getattr(msg, "role", "")
                content = getattr(msg, "content", "")
                if not content:
                    continue
                key = uuid.uuid4().hex
                value = {"role": role, "content": content}
                # Best-effort store put
                client.store.put_item(namespace=["eval_agent", "conversations", graph_name], key=key, value=value)
                added += 1
            except Exception:
                continue

        return create_success_response({"namespace": namespace, "messages_indexed": added})
    except Exception as e:
        return create_error_response(f"Failed to index conversation: {str(e)}\n{traceback.format_exc()}", e)


@tool
def compute_local_evaluation(runtime: ToolRuntime) -> Command:
    """Perform local evaluation of the target agent and report the score."""
    try:
        target_agent_config = runtime.state.get("target_agent_config")
        test_cases = runtime.state.get("test_cases")

        if not target_agent_config or not test_cases:
            raise ValueError("Missing target_agent_config or test_cases in state.")

        target_graph_name = target_agent_config.graph_name
        target_url = target_agent_config.url

        # Ensure dataset name and experiment name
        dataset_name = runtime.state.get("dataset_name")
        if not dataset_name:
            agent_id = target_graph_name.replace("/", "_")
            date_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
            dataset_name = f"seer_eval_{agent_id}_{date_tag}"
            # Try creating dataset, but don't fail if it exists
            try:
                _LANGSMITH_CLIENT.create_dataset(dataset_name)
            except Exception:
                logger.warning(f"Failed to create dataset {dataset_name}, it might already exist.")

        experiment_name = f"seer-local-eval-{target_graph_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        sync_client = get_sync_client(url=target_url)
        thread = sync_client.threads.create()
        thread_cfg = {"configurable": {"thread_id": thread["thread_id"]}}
        remote_graph = RemoteGraph(target_graph_name, url=target_url, client=_LANGSMITH_CLIENT, sync_client=sync_client, distributed_tracing=True)

        scores = []
        for tc_dict in test_cases:
            tc = GeneratedTestCase(**tc_dict) # Re-parse into object for type safety
            question = tc.input_message
            expected = tc.expected_output or tc.expected_behavior
            try:
                result = remote_graph.invoke({"messages": [{"role": "user", "content": question}]}, config=thread_cfg)
                answer = result.get("messages", [{}])[-1].get("content", "")
            except Exception as e:
                logger.error(f"Error invoking remote graph: {e}")
                answer = ""

            try:
                eval_result = correctness_evaluator(
                    inputs={"question": question},
                    outputs={"answer": answer},
                    reference_outputs={"answer": expected},
                )
                score = eval_result.get("score", 0.0) # Extract score directly
            except Exception as e:
                logger.error(f"Error running correctness evaluator: {e}")
                score = 0.0
            scores.append(score)

        mean_score = sum(scores) / max(len(scores), 1)

        # ToolMessage for internal state/traceability
        tool_msg_content = json.dumps({
            "dataset_name": dataset_name,
            "experiment_name": experiment_name,
            "score": mean_score
        })
        tool_message = ToolMessage(content=tool_msg_content, tool_call_id=runtime.tool_call_id)

        # AIMessage for user-facing summary
        user_summary = f"Final evaluation complete: score={mean_score:.2f} (0–1 scale). Dataset=`{dataset_name}`, Experiment=`{experiment_name}`."
        ai_message = AIMessage(content=user_summary)

        return Command(update={
            "dataset_name": dataset_name,
            "experiment_name": experiment_name,
            "score": float(mean_score),
            "messages": [tool_message, ai_message]
        })

    except Exception as e:
        msg = create_error_response(f"Failed to compute local evaluation: {str(e)}\n{traceback.format_exc()}", e)
        return Command(update={"messages": [ToolMessage(content=msg, tool_call_id=runtime.tool_call_id)]})


def build_graph():
    tools = [
        generate_evals,
        parse_eval_request,
        create_langsmith_dataset,
        test_cases_to_langsmith_dataset,
        run_langsmith_evaluation,
        index_conversation,
        think,
        compute_local_evaluation,
    ]
    return create_agent(
        model=_LLM,
        tools=tools,
        system_prompt=EVAL_AGENT_PROMPT,
        middleware=[
            TodoListMiddleware(),
        ],
        state_schema=EvalAgentState,
        debug=True,
    )

graph = build_graph()
