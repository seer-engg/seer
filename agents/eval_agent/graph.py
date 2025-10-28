import json
import hashlib
import uuid
from typing import Annotated, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langchain.tools import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call

from langchain_core.runnables import RunnableConfig
from shared.schemas import AgentSpec, TestResult, Eval as EvalSchema, AgentExpectation, TargetConfig
from agents.eval_agent.prompts import (
    EVAL_AGENT_PROMPT,
    EVAL_AGENT_SPEC_PROMPT,
    EVAL_AGENT_TEST_GEN_PROMPT,
    EVAL_AGENT_JUDGE_PROMPT,
)
from shared.llm import get_llm
from shared.logger import get_logger
from shared.messaging import messenger
from shared.error_handling import create_success_response, create_error_response
from urllib.parse import urlparse
from datetime import datetime
import os

# LangSmith / OpenAI
from langsmith import Client as LangSmithClient
from langsmith import wrappers as ls_wrappers
import openai
from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk import get_sync_client


# Get logger for eval agent
logger = get_logger('eval_agent')


# Typed state DTOs for short-term memory (serialized Eval)
class TargetConfigDTO(TypedDict, total=False):
    url: str
    port: int
    assistant_id: str
    github_url: str


class AgentExpectationDTO(TypedDict, total=False):
    if_condition: str
    then_behavior: str
    priority: str


class EvalDTO(TypedDict, total=False):
    id: str
    target: TargetConfigDTO
    expectations: list[AgentExpectationDTO]
    status: str
    spec_summary: str
    suite_preview: list[str]
    suite_id: str
    created_at: str
    metadata: dict


class EvalAgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    current_eval: EvalDTO
    latest_eval_suite: dict  # cached suite with test_cases for LangSmith examples
    langsmith_dataset_name: str


class EvalRequest(BaseModel):
    agent_name: str = Field(description="Name/ID of the agent to evaluate")
    agent_url: str = Field(description="Base URL where the agent is hosted")
    expectations: str = Field(description="User's natural language expectations")


class GeneratedTestCase(BaseModel):
    expectation_ref: str
    input_message: str
    expected_behavior: str
    success_criteria: str
    expected_output: Optional[str] = None


class GeneratedTests(BaseModel):
    test_cases: list[GeneratedTestCase]


class JudgeVerdict(BaseModel):
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str


@tool
def think(thought: str) -> str:
    """
    Think tool for eval_agent: log internal reflection; no external side effects.
    """
    logger.info(f"THINK: {thought}")
    return json.dumps({"success": True, "thought": thought})


@tool
def get_current_eval(runtime: ToolRuntime) -> str:
    """
    Return the latest Eval JSON stored in state (or empty string).
    """
    data = runtime.state.get("current_eval")
    return json.dumps(data) if data else ""


@tool
def set_current_eval(eval_json: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    Persist Eval JSON into state as current_eval and acknowledge via ToolMessage.
    Validates against shared Eval schema before storing.
    """
    try:
        eval_obj = EvalSchema.model_validate_json(eval_json)
        payload = eval_obj.model_dump()
    except Exception:
        # Store raw payload to aid debugging if validation fails
        try:
            payload = json.loads(eval_json)
        except Exception:
            payload = {"raw": eval_json}
    return Command(update={
        "current_eval": payload,
        "messages": [ToolMessage(content="Stored current_eval", tool_call_id=tool_call_id)]
    })


@tool
def propose_eval(eval_input_json: str, max_preview: int = 5, tool_call_id: Annotated[str, InjectedToolCallId] = None) -> Command:
    """
    Propose an Eval (draft) from TargetConfig + AgentExpectation list.
    Returns Eval JSON with status='draft', spec_summary, and suite_preview (no suite generated yet).
    """
    try:
        data = json.loads(eval_input_json)
    except Exception:
        data = {}

    # Normalize to EvalSchema fields
    try:
        # If already an Eval, validate; else construct
        if all(k in data for k in ["target", "expectations"]):
            eval_obj = EvalSchema.model_validate(data)
        else:
            target = TargetConfig.model_validate(data.get("target", {}))
            # If assistant_id is missing but agent_name is provided at top-level, set it
            maybe_agent_name = (data.get("agent_name") or "").strip()
            if not getattr(target, "assistant_id", None) and maybe_agent_name:
                try:
                    target.assistant_id = maybe_agent_name
                except Exception:
                    pass
            expectations_list = data.get("expectations", [])
            expectations: list[AgentExpectation] = []
            for item in expectations_list:
                if isinstance(item, dict):
                    expectations.append(AgentExpectation.model_validate(item))
                else:
                    # Fallback: treat strings as then_behavior only
                    expectations.append(AgentExpectation(if_condition="", then_behavior=str(item)))
            eval_obj = EvalSchema(target=target, expectations=expectations, status="draft")

        # Build a compact expectations string for summary and preview
        exp_lines = []
        for e in eval_obj.expectations:
            ic = (e.if_condition or "").strip()
            tb = (e.then_behavior or "").strip()
            exp_lines.append(f"IF {ic or '*'} THEN {tb}")
        expectations_text = "\n".join(exp_lines)

        # Create a concise spec summary via LLM
        llm = get_llm(temperature=0.0)
        summary_prompt = (
            "Summarize the following target and expectations in 6-8 bullet points for evaluation planning.\n"
            f"Target URL: {eval_obj.target.url}\n"
            f"Assistant ID: {eval_obj.target.assistant_id or 'unknown'}\n"
            f"Expectations (IF-THEN):\n{expectations_text}\n"
            "Focus on capabilities, constraints, and key success signals."
        )
        spec_summary = llm.invoke(summary_prompt).content or ""

        # Preview candidate test inputs using existing test-gen prompt with limited count
        # Build a lightweight AgentSpec input
        spec_llm = get_llm().with_structured_output(AgentSpec)
        spec_input = {
            "expectations": expectations_text,
            "agent_name": eval_obj.target.assistant_id or "target_agent",
            "agent_url": eval_obj.target.url,
        }
        spec_obj: AgentSpec = spec_llm.invoke(
            EVAL_AGENT_SPEC_PROMPT.format(
                expectations=spec_input["expectations"],
                agent_name=spec_input["agent_name"],
                agent_url=spec_input["agent_url"],
            )
        )

        # Generate a small set of tests for preview only
        gen_llm = get_llm().with_structured_output(GeneratedTests)
        generated: GeneratedTests = gen_llm.invoke(
            EVAL_AGENT_TEST_GEN_PROMPT.format(spec_json=spec_obj.model_dump_json())
        )
        preview_cases = generated.test_cases[: max(1, int(max_preview))]
        suite_preview = [tc.input_message for tc in preview_cases]

        # Prepare draft Eval
        eval_obj.spec_summary = spec_summary
        eval_obj.suite_preview = suite_preview
        eval_obj.status = "draft"
        payload = eval_obj.model_dump()
        return Command(update={
            "current_eval": payload,
            "messages": [ToolMessage(content=json.dumps(payload), tool_call_id=tool_call_id)]
        })
    except Exception as e:
        # Return error as ToolMessage to keep tool call pairing valid
        msg = create_error_response(f"Failed to propose eval: {str(e)}", e)
        return Command(update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]})


@tool
def approve_eval(eval_json: str, max_tests: int = 3, tool_call_id: Annotated[str, InjectedToolCallId] = None) -> Command:
    """
    Approve a draft Eval, generate tests and return updated Eval with suite_id and suite_preview.
    """
    try:
        eval_obj = EvalSchema.model_validate_json(eval_json)
        if eval_obj.status not in ("draft", "approved"):
            raise ValueError("Eval status must be 'draft' or 'approved' to proceed")

        # Build expectations text for spec generation
        exp_lines = []
        for e in eval_obj.expectations:
            ic = (e.if_condition or "").strip()
            tb = (e.then_behavior or "").strip()
            exp_lines.append(f"IF {ic or '*'} THEN {tb}")
        expectations_text = "\n".join(exp_lines)

        # Generate AgentSpec
        spec_llm = get_llm().with_structured_output(AgentSpec)
        spec_obj: AgentSpec = spec_llm.invoke(
            EVAL_AGENT_SPEC_PROMPT.format(
                expectations=expectations_text,
                agent_name=eval_obj.target.assistant_id or "target_agent",
                agent_url=eval_obj.target.url,
            )
        )

        # Generate tests and capture suite_id + preview
        gen_llm = get_llm().with_structured_output(GeneratedTests)
        generated: GeneratedTests = gen_llm.invoke(
            EVAL_AGENT_TEST_GEN_PROMPT.format(spec_json=spec_obj.model_dump_json())
        )

        # Build preview and fake suite id by reusing our deterministic id builder
        limited_cases = generated.test_cases[: max(1, int(max_tests))]
        inputs_preview: list[str] = [tc.input_message for tc in limited_cases]

        # Mirror generate_tests' naming for compatibility
        spec_name = getattr(spec_obj, "name", None) or "target_agent"
        suite_id = f"eval_{spec_name}_{uuid.uuid4().hex[:8]}"

        # Update eval
        eval_obj.suite_preview = inputs_preview
        eval_obj.suite_id = suite_id
        eval_obj.status = "approved"
        payload = eval_obj.model_dump()
        return Command(update={
            "current_eval": payload,
            "messages": [ToolMessage(content=json.dumps(payload), tool_call_id=tool_call_id)]
        })
    except Exception as e:
        msg = create_error_response(f"Failed to approve eval: {str(e)}", e)
        return Command(update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]})


@tool
async def run_test(test_input: str, config: RunnableConfig, runtime: ToolRuntime) -> str:
    """
    Run a single test against the target LangGraph agent via LangGraph SDK.
    ALWAYS reads TargetConfig (url/port/assistant_id) from short-term state (current_eval.target).
    Any provided target_url/target_agent_id args are ignored.
    """
    try:
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        # Load strictly from state
        ce = runtime.state.get("current_eval") or {}
        tgt = ce.get("target") or {}
        url = (tgt.get("url") or "").strip()
        agent_id = (tgt.get("assistant_id") or "").strip()
        port = tgt.get("port")

        def normalize_base_url(u: str, p: int | None) -> str:
            if p:
                return f"http://127.0.0.1:{p}"
            if "://" in (u or ""):
                parsed = urlparse(u)
                scheme = parsed.scheme or "http"
                host = parsed.hostname or "127.0.0.1"
                pp = parsed.port or (443 if scheme == "https" else 80)
                return f"{scheme}://{host}:{pp}"
            if not u:
                return "http://127.0.0.1:80"
            if u.isdigit():
                return f"http://127.0.0.1:{u}"
            if ":" in u:
                host, p2 = u.split(":", 1)
                host = (host or "127.0.0.1").strip()
                p2 = (p2 or "80").strip()
                return f"http://{host}:{p2}"
            return f"http://{u}:80"

        base_url = normalize_base_url(url, port)

        if not agent_id:
            return create_error_response("Missing target_agent_id (assistant_id) in args or state")

        # Prefer RemoteGraph when available; fall back to SDK messenger
        try:
            remote_graph = RemoteGraph(agent_id, url=base_url)
            result = await remote_graph.ainvoke({
                "messages": [{"role": "user", "content": test_input}]
            })
            # Extract last AI message content if present
            text = ""
            try:
                messages = result.get("messages", []) if isinstance(result, dict) else []
                if messages and isinstance(messages, list):
                    last = messages[-1]
                    text = (last.get("content", "") if isinstance(last, dict) else "")
            except Exception:
                text = ""
            if not text:
                # Fallback to str(result) if structure unexpected
                text = str(result)
            return create_success_response({"response": text, "thread_id": thread_id, "base_url": base_url, "assistant_id": agent_id})
        except Exception:
            text, remote_tid = await messenger.send(
                user_thread_id=thread_id,
                src_agent="eval_agent",
                dst_agent=agent_id,
                base_url=base_url,
                assistant_id=agent_id,
                content=test_input
            )
            return create_success_response({"response": text, "thread_id": remote_tid, "base_url": base_url, "assistant_id": agent_id})
    except Exception as e:
        return create_error_response(f"Failed to run test via SDK: {str(e)}", e)


@tool
def parse_eval_request(user_text: str) -> str:
    """
    Extract agent_name, agent_url, and expectations from the user's message.
    Returns JSON string with these fields.
    """
    extractor = get_llm(temperature=0.0).with_structured_output(EvalRequest)
    instruction = (
        "Extract agent_name, agent_url, and expectations from the user's latest message.\n\n"
        "IMPORTANT - agent_name is the assistant_id used by LangGraph."
    )
    req: EvalRequest = extractor.invoke(f"{instruction}\n\nUSER:\n{user_text}")
    return json.dumps(req.model_dump())


@tool
def generate_spec(input_json: str) -> str:
    """
    Generate an AgentSpec from expectations, agent_name, and agent_url.
    input_json must include: expectations, agent_name, agent_url.
    Returns AgentSpec as JSON.
    """
    try:
        data = json.loads(input_json)
    except Exception:
        data = {}
    llm = get_llm().with_structured_output(AgentSpec)
    prompt = EVAL_AGENT_SPEC_PROMPT.format(
        expectations=data.get("expectations", ""),
        agent_name=data.get("agent_name", ""),
        agent_url=data.get("agent_url", ""),
    )
    spec: AgentSpec = llm.invoke(prompt)
    return spec.model_dump_json()


@tool
def generate_tests(spec_json: str, max_tests: int = 3, tool_call_id: Annotated[str, InjectedToolCallId] = None) -> Command:
    """
    Generate test cases from AgentSpec JSON. Returns JSON with eval_suite and eval_context.
    Also caches the latest suite in state for LangSmith dataset creation.
    Defaults to generating up to `max_tests` unless the user explicitly asks for a different number.
    """
    llm = get_llm().with_structured_output(GeneratedTests)
    generated: GeneratedTests = llm.invoke(EVAL_AGENT_TEST_GEN_PROMPT.format(spec_json=spec_json))

    # Determine spec name if provided
    try:
        spec_obj = AgentSpec.model_validate_json(spec_json)
        spec_name = spec_obj.name
        spec_version = spec_obj.version
    except Exception:
        spec_name = "target_agent"
        spec_version = "1.0.0"

    # Build test case IDs deterministically
    test_cases: list[dict] = []
    inputs_preview: list[str] = []
    limited_cases = generated.test_cases[: max(1, int(max_tests))]
    for idx, tc in enumerate(limited_cases):
        content_hash = hashlib.md5(f"{tc.expectation_ref}{tc.input_message}".encode()).hexdigest()[:8]
        test_id = f"{spec_name}_{idx+1}_{content_hash}"
        test_cases.append({
            "id": test_id,
            "expectation_ref": tc.expectation_ref,
            "input_message": tc.input_message,
            "expected_behavior": tc.expected_behavior,
            "success_criteria": tc.success_criteria,
            "expected_output": getattr(tc, "expected_output", None) or "",
        })
        inputs_preview.append(f"[{idx}] {tc.input_message}")

    eval_suite = {
        "id": f"eval_{spec_name}_{uuid.uuid4().hex[:8]}",
        "spec_name": spec_name,
        "spec_version": spec_version,
        "test_cases": test_cases,
    }

    eval_context = (
        "EVAL_CONTEXT\n"
        f"spec_name: {spec_name}\n"
        f"test_count: {len(test_cases)}\n"
        "test_inputs (indexed):\n"
        + "\n".join(inputs_preview)
    )

    payload = json.dumps({"eval_suite": eval_suite, "eval_context": eval_context})
    return Command(update={
        "latest_eval_suite": eval_suite,
        "messages": [ToolMessage(content=payload, tool_call_id=tool_call_id)]
    })


@tool
def judge_result(input_json: str) -> str:
    """
    Judge the latest run_test output against a provided test case.
    Input JSON: {"input_message","expected_behavior","success_criteria","actual_output","test_case_id"}
    Returns JSON TestResult fields and a concise verdict.
    """
    try:
        data = json.loads(input_json)
    except Exception:
        data = {}
    judge_llm = get_llm().with_structured_output(JudgeVerdict)
    prompt = EVAL_AGENT_JUDGE_PROMPT.format(
        input_message=data.get("input_message", ""),
        expected_behavior=data.get("expected_behavior", ""),
        success_criteria=data.get("success_criteria", ""),
        actual_output=data.get("actual_output", ""),
    )
    verdict: JudgeVerdict = judge_llm.invoke(prompt)
    result = TestResult(
        test_case_id=data.get("test_case_id", ""),
        input_sent=data.get("input_message", ""),
        actual_output=data.get("actual_output", ""),
        expected_behavior=data.get("expected_behavior", ""),
        passed=verdict.passed,
        score=verdict.score,
        judge_reasoning=verdict.reasoning,
    )
    return json.dumps({
        "passed": verdict.passed,
        "score": verdict.score,
        "reasoning": verdict.reasoning,
        "test_result": result.model_dump(),
    })


@wrap_tool_call
async def handle_tool_errors(request, handler):
    try:
        return await handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"],
        )

# -----------------------
# LangSmith integration
# -----------------------

@tool
def create_langsmith_dataset(name: str | None = None, runtime: ToolRuntime = None) -> str:
    """Create or read a LangSmith dataset and store its name in state for later use."""
    try:
        # Derive deterministic name from current_eval and suite if available
        if not name:
            ce = runtime.state.get("current_eval") or {}
            tgt = ce.get("target") or {}
            agent_id = (tgt.get("assistant_id") or "target_agent").replace("/", "_")
            suite_id = (ce.get("suite_id") or (runtime.state.get("latest_eval_suite") or {}).get("id"))
            if suite_id:
                name = f"seer_eval_{agent_id}_{suite_id}"
            else:
                date_tag = datetime.utcnow().strftime("%Y%m%d")
                name = f"seer_eval_{agent_id}_{date_tag}"

        client = LangSmithClient()
        try:
            _ = client.read_dataset(dataset_name=name)
        except Exception:
            _ = client.create_dataset(name)

        # Persist dataset name in state for reuse
        return Command(update={
            "langsmith_dataset_name": name
        })
    except Exception as e:
        return create_error_response(f"Failed to create/read LangSmith dataset: {str(e)}", e)


@tool
def upsert_examples_from_suite(dataset_name: str | None = None, runtime: ToolRuntime = None) -> str:
    """Upsert examples into LangSmith dataset from cached latest_eval_suite."""
    try:
        suite = runtime.state.get("latest_eval_suite") or {}
        test_cases = suite.get("test_cases") or []
        if not test_cases:
            return create_error_response("No cached test cases found. Please run generate_tests first.")

        ce = runtime.state.get("current_eval") or {}
        tgt = ce.get("target") or {}
        agent_id = (tgt.get("assistant_id") or "target_agent").replace("/", "_")
        if not dataset_name:
            dataset_name = runtime.state.get("langsmith_dataset_name")
        if not dataset_name:
            suite_id = (ce.get("suite_id") or suite.get("id"))
            if suite_id:
                dataset_name = f"seer_eval_{agent_id}_{suite_id}"
            else:
                date_tag = datetime.utcnow().strftime("%Y%m%d")
                dataset_name = f"seer_eval_{agent_id}_{date_tag}"

        client = LangSmithClient()
        try:
            dataset = client.read_dataset(dataset_name=dataset_name)
        except Exception:
            dataset = client.create_dataset(dataset_name)

        examples = [
            {
                "inputs": {"question": tc.get("input_message", "")},
                "outputs": {"answer": (tc.get("expected_output") or tc.get("expected_behavior", ""))},
            }
            for tc in test_cases
        ]
        if examples:
            client.create_examples(dataset_id=dataset.id, examples=examples)
        return create_success_response({"dataset_name": dataset_name, "examples": len(examples)})
    except Exception as e:
        return create_error_response(f"Failed to upsert examples: {str(e)}", e)


@tool
def run_langsmith_evaluation(dataset_name: str | None = None, experiment_prefix: str | None = None, runtime: ToolRuntime = None) -> str:
    """Run LangSmith evaluation over the given dataset, calling the target agent for responses."""
    try:
        ce = runtime.state.get("current_eval") or {}
        tgt = ce.get("target") or {}
        url = (tgt.get("url") or "").strip()
        agent_id = (tgt.get("assistant_id") or "").strip()
        port = tgt.get("port")
        if not agent_id:
            return create_error_response("Missing target assistant_id in current_eval.target")

        def normalize_base_url(u: str, p: int | None) -> str:
            if p:
                return f"http://127.0.0.1:{p}"
            if "://" in (u or ""):
                parsed = urlparse(u)
                scheme = parsed.scheme or "http"
                host = parsed.hostname or "127.0.0.1"
                pp = parsed.port or (443 if scheme == "https" else 80)
                return f"{scheme}://{host}:{pp}"
            if not u:
                return "http://127.0.0.1:80"
            if u.isdigit():
                return f"http://127.0.0.1:{u}"
            if ":" in u:
                host, p2 = u.split(":", 1)
                host = (host or "127.0.0.1").strip()
                p2 = (p2 or "80").strip()
                return f"http://{host}:{p2}"
            return f"http://{u}:80"

        base_url = normalize_base_url(url, port)

        # Default dataset name: prefer stored one, then deterministic
        if not dataset_name:
            dataset_name = runtime.state.get("langsmith_dataset_name")
        if not dataset_name:
            suite = runtime.state.get("latest_eval_suite") or {}
            suite_id = (ce.get("suite_id") or suite.get("id"))
            if suite_id:
                dataset_name = f"seer_eval_{agent_id or 'target_agent'}_{suite_id}"
            else:
                date_tag = datetime.utcnow().strftime("%Y%m%d")
                dataset_name = f"seer_eval_{agent_id or 'target_agent'}_{date_tag}"

        client = LangSmithClient()
        try:
            dataset = client.read_dataset(dataset_name=dataset_name)
        except Exception:
            return create_error_response(f"Dataset '{dataset_name}' not found. Please create and upsert examples first.")

        # Wrap OpenAI for LangSmith tracing
        openai_client = ls_wrappers.wrap_openai(openai.OpenAI())

        eval_instructions = (
            "You are an expert professor specialized in grading answers for correctness against the reference answer."
        )

        def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
            user_content = (
                f"You are grading the following question:\n{inputs.get('question','')}\n"
                f"Here is the real answer:\n{reference_outputs.get('answer','')}\n"
                f"You are grading the following predicted answer:\n{outputs.get('response','')}\n"
                "Respond with CORRECT or INCORRECT:\nGrade:"
            )
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": eval_instructions},
                    {"role": "user", "content": user_content},
                ],
            ).choices[0].message.content
            return response == "CORRECT"

        def concision(outputs: dict, reference_outputs: dict) -> bool:
            ref = reference_outputs.get("answer", "")
            pred = outputs.get("response", "")
            try:
                return int(len(pred.split()) <= 2 * max(1, len(ref.split())))
            except Exception:
                return True

        # Target function to call the agent under test
        async def call_target(question: str) -> str:
            # Prefer RemoteGraph; fall back to SDK messenger
            try:
                remote_graph = RemoteGraph(agent_id, url=base_url)
                result = await remote_graph.ainvoke({
                    "messages": [{"role": "user", "content": question}]
                })
                msg = ""
                try:
                    messages = result.get("messages", []) if isinstance(result, dict) else []
                    if messages and isinstance(messages, list):
                        last = messages[-1]
                        msg = (last.get("content", "") if isinstance(last, dict) else "")
                except Exception:
                    msg = ""
                return msg or str(result)
            except Exception:
                text, _remote_tid = await messenger.send(
                    user_thread_id=uuid.uuid4().hex,
                    src_agent="eval_agent",
                    dst_agent=agent_id,
                    base_url=base_url,
                    assistant_id=agent_id,
                    content=question,
                )
                return text

        def ls_target(inputs: dict) -> dict:
            # Synchronous wrapper that runs the async call
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                resp = loop.run_until_complete(call_target(inputs.get("question", "")))
            finally:
                loop.close()
            return {"response": resp}

        if not experiment_prefix:
            experiment_prefix = f"{agent_id or 'target_agent'}-{datetime.utcnow().strftime('%Y%m%d')}"

        results = client.evaluate(
            ls_target,
            data=dataset_name,
            evaluators=[concision, correctness],
            experiment_prefix=experiment_prefix,
        )

        return create_success_response({
            "dataset_name": dataset_name,
            "experiment_name": getattr(results, "experiment_name", experiment_prefix),
        })
    except Exception as e:
        return create_error_response(f"Failed to run LangSmith evaluation: {str(e)}", e)


@tool
def index_conversation(namespace_prefix: str = "conversations", runtime: ToolRuntime = None) -> str:
    """Index the eval agent conversation into the local LangGraph store for semantic search."""
    try:
        # Determine self URL from env (default 8002)
        port = os.getenv("EVAL_AGENT_PORT", "8002").strip()
        base_url = f"http://127.0.0.1:{port}"

        # Collect conversation messages
        messages = runtime.state.get("messages") or []
        ce = runtime.state.get("current_eval") or {}
        tgt = ce.get("target") or {}
        agent_id = (tgt.get("assistant_id") or "unknown").replace("/", "_")
        namespace = f"{namespace_prefix}/eval_agent/{agent_id}"

        client = get_sync_client(url=base_url)

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
                client.store.put(namespace=namespace, key=key, value=value)
                added += 1
            except Exception:
                continue

        return create_success_response({"namespace": namespace, "messages_indexed": added})
    except Exception as e:
        return create_error_response(f"Failed to index conversation: {str(e)}", e)

def build_graph():
    model = get_llm(temperature=0.0)
    tools = [
        propose_eval,
        approve_eval,
        get_current_eval,
        set_current_eval,
        parse_eval_request,
        generate_spec,
        generate_tests,
        # LangSmith path
        create_langsmith_dataset,
        upsert_examples_from_suite,
        run_langsmith_evaluation,
        index_conversation,
        # Ad-hoc tools
        run_test,
        judge_result,
        think,
    ]
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=EVAL_AGENT_PROMPT,
        middleware=[handle_tool_errors],
        state_schema=EvalAgentState,
    )



# NOTE: graph is created after all tools are defined
graph = build_graph()