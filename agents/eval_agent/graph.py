import requests
import json
import hashlib
import uuid
import traceback
import os
from datetime import datetime, timezone

from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langgraph.types import Command
from langchain.agents import create_agent
from langsmith import Client
from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk import get_sync_client
from langchain.agents.middleware import TodoListMiddleware
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from openevals.types import EvaluatorResult

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
_CORRECTNESS_EVALUATOR = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    model="openai:gpt-4.1-mini",
    feedback_key="correctness",
)

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
    """Perform local evaluation of the target agent and uploads the experiment to LangSmith"""
    try:
        target_agent_config = runtime.state.get("target_agent_config")
        test_cases = runtime.state.get("test_cases")

        if not target_agent_config or not test_cases:
            raise ValueError("Missing target_agent_config or test_cases in state.")

        target_graph_name = target_agent_config.graph_name
        target_url = target_agent_config.url
        agent_id = target_graph_name.replace("/", "_")
        date_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
        dataset_name = f"seer_eval_{agent_id}_{date_tag}"
        experiment_name = f"seer-local-eval-{target_graph_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        sync_client = get_sync_client(url=target_url)
        thread = sync_client.threads.create()
        thread_cfg = {"configurable": {"thread_id": thread["thread_id"]}}
        remote_graph = RemoteGraph(
            target_graph_name,
            url=target_url,
            client=_LANGSMITH_CLIENT,
            sync_client=sync_client,
            distributed_tracing=True,
        )

        experiment_start_time = datetime.now(timezone.utc)
        scores: list[float] = []
        results_payload: list[dict] = []
        earliest_start = experiment_start_time
        latest_end = experiment_start_time

        for tc_dict in test_cases:
            tc = GeneratedTestCase(**tc_dict)  # Re-parse into object for type safety
            question = tc.input_message
            expected = tc.expected_output or tc.expected_behavior

            row_id = uuid.uuid4().hex
            run_start = datetime.now(timezone.utc)
            try:
                result = remote_graph.invoke({"messages": [{"role": "user", "content": question}]}, config=thread_cfg)
                answer = result.get("messages", [{}])[-1].get("content", "")
            except Exception as e:
                logger.error(f"Error invoking remote graph: {e}")
                answer = ""
            run_end = datetime.now(timezone.utc)

            eval_result: EvaluatorResult = _CORRECTNESS_EVALUATOR(
                inputs={"question": question},
                outputs={"answer": answer},
                reference_outputs={"answer": expected},
            )
            score = eval_result["score"]
            scores.append(score)
            evaluator_comment = eval_result.get('comment', '')

            results_payload.append({
                "row_id": row_id,
                "inputs": {"question": question},
                "expected_outputs": {"answer": expected},
                "actual_outputs": {"answer": answer},
                "evaluation_scores": [
                    {
                        "key": "correctness",
                        "score": score,
                        "comment": evaluator_comment,
                    }
                ],
                "start_time": run_start.isoformat(),
                "end_time": run_end.isoformat(),
                "run_name": target_graph_name,
                "run_metadata": {
                    "expectation_ref": tc.expectation_ref,
                    "success_criteria": tc.success_criteria,
                },
            })

            if run_start < earliest_start:
                earliest_start = run_start
            if run_end > latest_end:
                latest_end = run_end

        mean_score = sum(scores) / max(len(scores), 1)
        experiment_end_time = max(latest_end, datetime.now(timezone.utc))

        api_key = os.getenv("LANGSMITH_API_KEY")
        if not api_key:
            raise ValueError("LANGSMITH_API_KEY environment variable is required for experiment upload.")

        api_base = os.getenv("LANGSMITH_API_URL", "https://api.smith.langchain.com")
        endpoint = f"{api_base.rstrip('/')}/api/v1/datasets/upload-experiment"

        experiment_description = (
            "Evaluation uploaded by Seer eval_agent via compute_local_evaluation."
        )

        upload_body = {
            "experiment_name": experiment_name,
            "experiment_description": experiment_description,
            "dataset_name": dataset_name,
            "experiment_start_time": earliest_start.isoformat(),
            "experiment_end_time": experiment_end_time.isoformat(),
            "experiment_metadata": {
                "target_graph_name": target_graph_name,
                "target_url": target_url,
            },
            "summary_experiment_scores": [
                {
                    "key": "mean_correctness",
                    "score": mean_score,
                    "comment": "Average correctness score across generated tests.",
                }
            ],
            "results": results_payload,
        }

        try:
            response = requests.post(
                endpoint,
                json=upload_body,
                headers={"x-api-key": api_key},
                timeout=60,
            )
        except Exception as request_error:
            raise RuntimeError(f"Failed to upload experiment to LangSmith: {request_error}") from request_error

        if not response.ok:
            raise RuntimeError(
                f"LangSmith upload failed with status {response.status_code}: {response.text}"
            )

        response_data = response.json()
        dataset_info = response_data.get("dataset", {})
        experiment_info = response_data.get("experiment", {})

        uploaded_dataset_name = dataset_info.get("name", dataset_name)
        uploaded_experiment_name = experiment_info.get("name", experiment_name)

        # ToolMessage for internal state/traceability
        tool_msg_content = json.dumps({
            "dataset_name": uploaded_dataset_name,
            "experiment_name": uploaded_experiment_name,
            "score": mean_score,
            "rows": len(results_payload),
        })
        tool_message = ToolMessage(content=tool_msg_content, tool_call_id=runtime.tool_call_id)

        # AIMessage for user-facing summary
        user_summary = (
            "Final evaluation uploaded: score="
            f"{mean_score:.2f} (0–1 scale). Dataset=`{uploaded_dataset_name}`, "
            f"Experiment=`{uploaded_experiment_name}`."
        )
        ai_message = AIMessage(content=user_summary)

        return Command(update={
            "dataset_name": uploaded_dataset_name,
            "experiment_name": uploaded_experiment_name,
            "score": float(mean_score),
            "messages": [tool_message, ai_message],
        })

    except Exception as e:
        msg = create_error_response(f"Failed to compute local evaluation: {str(e)}\n{traceback.format_exc()}", e)
        return Command(update={"messages": [ToolMessage(content=msg, tool_call_id=runtime.tool_call_id)]})


def build_graph():
    tools = [
        generate_evals,
        parse_eval_request,
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
