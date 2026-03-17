# pylint: disable=too-many-lines
# Reason: Orchestration module with multiple service classes; refactoring deferred
"""
Services for workflow chat endpoint to reduce complexity.
"""
import asyncio
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from seer.api.core.errors import VALIDATION_PROBLEM, raise_problem
from seer.config import config
from seer.logger import get_logger
from seer.utilities.langfuse_tracing import merge_nexus_langfuse_callbacks
from seer.utilities.message_sanitizer import sanitize_tool_call_id

from .services import (
    create_chat_session,
    get_chat_session,
    get_chat_session_by_thread_id,
)

try:
    import psycopg
except ImportError:
    psycopg = None

logger = get_logger(__name__)


class SessionService:
    """Handles chat session creation and retrieval logic."""

    @staticmethod
    async def get_or_create_session(
        workflow,
        user,
        thread_id: Optional[str] = None,
        session_id: Optional[int] = None,
    ) -> Tuple[Any, str, int]:
        """
        Get or create a chat session.

        Returns:
            Tuple of (session, thread_id, session_id)
        """
        session = None

        if thread_id:
            session = await get_chat_session_by_thread_id(thread_id, workflow)
            if session:
                session_id = session.id
            else:
                raise_problem(
                    type_uri=VALIDATION_PROBLEM,
                    title="Session not found",
                    detail=f"Chat session not found for thread_id: {thread_id}",
                    status=404,
                )
        elif session_id:
            session = await get_chat_session(session_id, workflow)
            if session:
                thread_id = session.thread_id
            else:
                raise_problem(
                    type_uri=VALIDATION_PROBLEM,
                    title="Session not found",
                    detail=f"Chat session not found for session_id: {session_id}",
                    status=404,
                )
        else:
            # Create new session only when no identifiers provided
            thread_id = f"workflow-{workflow.workflow_id}-{uuid.uuid4().hex}"
            session = await create_chat_session(
                workflow=workflow,
                user=user,
                thread_id=thread_id,
            )
            session_id = session.id

        return session, thread_id, session_id


class CheckpointerHealthService:
    """Handles checkpointer connection health and reconnection."""

    @staticmethod
    def is_connection_error(error: Exception) -> bool:
        """Check if an error is a connection-related error."""
        return (
            (psycopg and isinstance(error, psycopg.OperationalError))
            or isinstance(error, (ConnectionError, EOFError))
            or "connection is closed" in str(error).lower()
            or "ssl syscall error" in str(error).lower()
        )

    @staticmethod
    async def get_checkpointer_with_reconnect(reconnect_func) -> Optional[Any]:
        """
        Attempt to get or recreate checkpointer.

        Args:
            reconnect_func: Async function to recreate checkpointer

        Returns:
            Checkpointer instance or None
        """
        try:
            return await reconnect_func()
        except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Recovery mechanism, catch all to prevent cascading failures
            logger.error("Failed to reconnect checkpointer: %s", e)
            return None


class IncompleteToolCallDetector:
    """Detects incomplete tool calls in message lists."""

    @staticmethod
    def extract_tool_call_ids(message: Any) -> Set[str]:
        """Extract tool call IDs from a message."""
        tool_call_ids = set()

        if isinstance(message, AIMessage) and hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                tool_call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tool_call_id:
                    tool_call_ids.add(sanitize_tool_call_id(tool_call_id))

        return tool_call_ids

    @staticmethod
    def extract_tool_response_ids(messages: List[Any]) -> Set[str]:
        """Extract tool response IDs from a list of messages."""
        tool_response_ids = set()

        for m in messages:
            if isinstance(m, ToolMessage):
                tool_call_id = getattr(m, "tool_call_id", None)
                if tool_call_id:
                    tool_response_ids.add(sanitize_tool_call_id(tool_call_id))
            elif isinstance(m, dict) and m.get("type") == "tool":
                tool_call_id = m.get("tool_call_id")
                if tool_call_id:
                    tool_response_ids.add(sanitize_tool_call_id(tool_call_id))

        return tool_response_ids

    @classmethod
    def has_incomplete_tool_calls(cls, messages: List[Any]) -> bool:
        """
        Check if message list has incomplete tool calls.

        Returns:
            True if incomplete tool calls detected
        """
        for i, msg in enumerate(messages):
            if not isinstance(msg, AIMessage) or not hasattr(msg, "tool_calls") or not msg.tool_calls:
                continue

            tool_call_ids = cls.extract_tool_call_ids(msg)
            if not tool_call_ids:
                continue

            following_msgs = messages[i + 1: i + 1 + len(tool_call_ids) * 2]
            tool_response_ids = cls.extract_tool_response_ids(following_msgs)

            if tool_call_ids - tool_response_ids:
                logger.warning("Found incomplete tool calls. Missing responses for: %s", tool_call_ids - tool_response_ids)
                return True

        return False


class IncompleteToolCallRecoveryService:
    """Handles recovery from incomplete tool call states."""

    @staticmethod
    def _extract_checkpoint_tool_call_ids(chk_msg: Any) -> Set[str]:
        """Extract tool call IDs from checkpoint message."""
        tool_call_ids = set()
        tool_calls = None
        msg_type = None

        if isinstance(chk_msg, AIMessage):
            msg_type = "ai"
            tool_calls = getattr(chk_msg, "tool_calls", None)
        elif isinstance(chk_msg, dict):
            msg_type = chk_msg.get("type") or chk_msg.get("role", "")
            tool_calls = chk_msg.get("tool_calls")

        if msg_type in ("ai", "assistant") and tool_calls:
            for tc in tool_calls:
                tool_call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tool_call_id:
                    tool_call_ids.add(sanitize_tool_call_id(tool_call_id))

        return tool_call_ids

    @staticmethod
    def _extract_checkpoint_response_ids(messages: List[Any]) -> Set[str]:
        """Extract tool response IDs from checkpoint messages."""
        response_ids = set()

        for m in messages:
            if isinstance(m, ToolMessage):
                tool_call_id = getattr(m, "tool_call_id", None)
                if tool_call_id:
                    response_ids.add(sanitize_tool_call_id(tool_call_id))
            elif isinstance(m, dict):
                m_type = m.get("type") or m.get("role", "")
                if m_type == "tool":
                    tool_call_id = m.get("tool_call_id")
                    if tool_call_id:
                        response_ids.add(sanitize_tool_call_id(tool_call_id))

        return response_ids

    @classmethod
    def find_safe_checkpoint(cls, checkpoints: List[Any]) -> Optional[Any]:
        """
        Find the last checkpoint without incomplete tool calls.

        Args:
            checkpoints: List of checkpoint tuples

        Returns:
            Safe checkpoint tuple or None
        """
        for checkpoint_tuple in reversed(checkpoints[:-1]):
            checkpoint_messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", [])
            has_incomplete = False

            for j, chk_msg in enumerate(checkpoint_messages):
                chk_tool_call_ids = cls._extract_checkpoint_tool_call_ids(chk_msg)
                if not chk_tool_call_ids:
                    continue

                chk_following = checkpoint_messages[j + 1: j + 1 + len(chk_tool_call_ids) * 2]
                chk_response_ids = cls._extract_checkpoint_response_ids(chk_following)

                if chk_tool_call_ids - chk_response_ids:
                    has_incomplete = True
                    break

            if not has_incomplete:
                return checkpoint_tuple

        return None

    @staticmethod
    async def delete_thread(checkpointer: Any, thread_id: str) -> None:
        """Delete a thread from checkpointer."""
        if hasattr(checkpointer, "adelete_thread"):
            await checkpointer.adelete_thread(thread_id)
        else:
            await asyncio.to_thread(checkpointer.delete_thread, thread_id)


class InterruptHandler:
    """Handles interrupt detection and extraction."""

    @staticmethod
    def extract_interrupt_from_result(result: Any) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Extract interrupt data from agent result.

        Returns:
            Tuple of (interrupt_required, interrupt_data)
        """
        if not isinstance(result, dict):
            return False, None

        interrupt_data = None

        if "__interrupt__" in result:
            interrupts = result["__interrupt__"]
            if isinstance(interrupts, list) and len(interrupts) > 0:
                first_interrupt = interrupts[0]
                if hasattr(first_interrupt, "value"):
                    interrupt_data = (
                        first_interrupt.value
                        if isinstance(first_interrupt.value, dict)
                        else {"value": first_interrupt.value}
                    )
                elif isinstance(first_interrupt, dict):
                    interrupt_data = first_interrupt.get("value", first_interrupt)
                else:
                    interrupt_data = {"value": str(first_interrupt)}
            elif isinstance(interrupts, dict):
                interrupt_data = interrupts
            else:
                interrupt_data = {"value": str(interrupts)}
            return True, interrupt_data

        if "interrupt" in result:
            interrupt_data = result["interrupt"] if isinstance(result["interrupt"], dict) else {"value": result["interrupt"]}
            return True, interrupt_data

        return False, None

    @staticmethod
    async def extract_interrupt_from_state(agent: Any, config_dict: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Extract interrupt data from agent state.

        Returns:
            Tuple of (interrupt_required, interrupt_data)
        """
        try:
            current_state = await agent.aget_state(config_dict)
            # LangGraph 1.x stores interrupts in StateSnapshot.interrupts (tuple of Interrupt objects)
            interrupts = getattr(current_state, "interrupts", None)
            if interrupts:
                first_interrupt = interrupts[0]
                if hasattr(first_interrupt, "value"):
                    interrupt_data = (
                        first_interrupt.value
                        if isinstance(first_interrupt.value, dict)
                        else {"value": first_interrupt.value}
                    )
                elif isinstance(first_interrupt, dict):
                    interrupt_data = first_interrupt.get("value", first_interrupt)
                else:
                    interrupt_data = {"value": str(first_interrupt)}
                return True, interrupt_data
        except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Interrupt check is non-critical, log and proceed
            logger.debug("Could not check state for interrupts: %s", e)

        return False, None


# ---------------------------------------------------------------------------
# LangGraph astream_events helpers (shared by ChatOrchestrator and chat tasks)
# ---------------------------------------------------------------------------

async def _handle_tool_event(
    event_type: str,
    event_name: str,
    event_data: Dict[str, Any],
    publisher: Any,
    stream_event_type: Any,
) -> None:
    """Publish TOOL_START or TOOL_END to the stream publisher."""
    if event_type == "on_tool_start":
        await publisher.publish(stream_event_type.TOOL_START, {
            "tool_name": event_name,
            "input_preview": str(event_data.get("input", {}))[:200],
        })
    elif event_type == "on_tool_end":
        await publisher.publish(stream_event_type.TOOL_END, {
            "tool_name": event_name,
            "output_preview": str(event_data.get("output", ""))[:200],
        })


async def _handle_chat_model_end(
    event_data: Dict[str, Any],
    publisher: Any,
    stream_event_type: Any,
) -> Optional[str]:
    """Handle on_chat_model_end. Returns final_content if this is the terminal LLM call."""
    output = event_data.get("output")
    if output is None:
        return None
    tool_calls = getattr(output, "tool_calls", None) or []
    content = getattr(output, "content", "") or ""
    if tool_calls and content:
        await publisher.publish(stream_event_type.AI_MESSAGE, {"content": str(content)})
    elif content:
        return str(content)
    return None


def _handle_chain_end(event_name: str, event_data: Dict[str, Any]) -> Optional[List[Any]]:
    """Handle on_chain_end LangGraph. Returns captured messages list or None."""
    if event_name != "LangGraph":
        return None
    output = event_data.get("output", {})
    if isinstance(output, dict):
        return output.get("messages", [])
    return None


async def process_stream_event(
    event: Dict[str, Any],
    publisher: Any,
    stream_event_type: Any,
) -> Tuple[Optional[str], Optional[List[Any]]]:
    """
    Process a single LangGraph astream_events v2 event.

    Returns:
        Tuple of (final_content, all_messages) — either may be None.
    """
    event_type = event.get("event", "")
    event_name = event.get("name", "")
    event_data = event.get("data", {})

    if event_type in ("on_tool_start", "on_tool_end"):
        await _handle_tool_event(event_type, event_name, event_data, publisher, stream_event_type)
        return None, None
    if event_type == "on_chat_model_end":
        return await _handle_chat_model_end(event_data, publisher, stream_event_type), None
    if event_type == "on_chain_end":
        return None, _handle_chain_end(event_name, event_data)
    return None, None


class ChatOrchestrator:
    """Orchestrates agent invocation with health checks and recovery."""

    def __init__(  # pylint: disable=too-many-positional-arguments # Reason: Service orchestrator with multiple dependencies, requires refactoring to config object
        self,
        agent: Any,
        checkpointer: Any,
        health_service: CheckpointerHealthService,
        detector: IncompleteToolCallDetector,
        recovery_service: IncompleteToolCallRecoveryService,
        reconnect_func,
    ):
        """Initialize ChatOrchestrator."""
        self.agent = agent
        self.checkpointer = checkpointer
        self.health_service = health_service
        self.detector = detector
        self.recovery_service = recovery_service
        self.reconnect_func = reconnect_func

    async def invoke_with_timeout(
        self,
        messages: Any,
        config_dict: Dict[str, Any],
        # pylint: disable=unused-argument # Reason: Legacy parameter for compatibility
        thread_id_context=None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Invoke agent with timeout."""
        # pylint: disable=import-outside-toplevel # Reason: Avoids circular import with agents module
        from seer.agents.nexus import _current_thread_id

        thread_id = config_dict.get("configurable", {}).get("thread_id")
        token = _current_thread_id.set(thread_id) if thread_id else None
        timeout_seconds = timeout if timeout is not None else float(config.nexus_chat_timeout_seconds)

        recursion_limit = config_dict.get("recursion_limit", 25)
        logger.info("Invoking agent thread=%s recursion_limit=%d timeout=%ds", thread_id or 'unknown', recursion_limit, int(timeout_seconds))

        config_with_langfuse = merge_nexus_langfuse_callbacks(config_dict)

        try:
            return await asyncio.wait_for(
                self.agent.ainvoke(messages, config=config_with_langfuse),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.error("Agent invocation timed out after %ss for thread %s", timeout_seconds, thread_id or 'unknown')
            raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Request timeout",
                detail="Request timed out. The agent took too long to respond.",
                status=504,
            )
        finally:
            if token is not None:
                _current_thread_id.reset(token)

    async def _check_for_incomplete_tool_calls(self, config_dict: Dict[str, Any]) -> bool:
        """Check if current state has incomplete tool calls."""
        from seer.api.agents.checkpointer import get_checkpointer_with_retry  # pylint: disable=import-outside-toplevel # Reason: Avoids circular import with checkpointer module

        thread_id = config_dict.get("configurable", {}).get("thread_id")
        logger.debug("Checking checkpointer health for thread %s", thread_id)

        for attempt in range(2):
            try:
                checkpointer = await get_checkpointer_with_retry()
                if checkpointer is None:
                    logger.warning("Checkpointer unavailable, proceeding without state check")
                    return False
                current_state = await self.agent.aget_state(config_dict)
                messages = current_state.values.get("messages", [])
                return self.detector.has_incomplete_tool_calls(messages)
            except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Resilient state checking
                if attempt == 0 and self.health_service.is_connection_error(e):
                    logger.warning("Connection error during state check: %s, reconnecting...", e)
                    try:
                        await self.reconnect_func()
                        continue
                    except Exception:  # pylint: disable=broad-exception-caught # Reason: Recovery fallback
                        pass
                logger.warning("Error checking state for incomplete tool calls: %s. Proceeding.", e)
                return False
        return False

    async def _find_safe_checkpoint_config(
        self, config_dict: Dict[str, Any], thread_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Find a safe checkpoint to resume from, or None if unavailable."""
        from seer.api.agents.checkpointer import get_checkpointer_with_retry  # pylint: disable=import-outside-toplevel # Reason: Avoids circular import

        checkpointer = await get_checkpointer_with_retry()
        if not checkpointer:
            return None

        try:
            checkpoints = await asyncio.wait_for(self._list_checkpoints(checkpointer, config_dict), timeout=10.0)
        except asyncio.TimeoutError:
            logger.error("Checkpoint listing timed out for thread %s", thread_id)
            return None

        safe = self.recovery_service.find_safe_checkpoint(checkpoints)
        if safe:
            cp_id = safe.config["configurable"]["checkpoint_id"]
            logger.info("Found safe checkpoint: %s", cp_id)
            return {"configurable": {"thread_id": thread_id, "checkpoint_id": cp_id}}
        return None

    async def _delete_thread_with_fallback(self, thread_id: str) -> None:
        """Delete thread state, falling back to reconnect on failure."""
        from seer.api.agents.checkpointer import get_checkpointer  # pylint: disable=import-outside-toplevel # Reason: Avoids circular import

        try:
            cp = await get_checkpointer()
            if cp:
                await self.recovery_service.delete_thread(cp, thread_id)
        except Exception:  # pylint: disable=broad-exception-caught # Reason: Recovery fallback
            try:
                cp = await self.reconnect_func()
                if cp:
                    await self.recovery_service.delete_thread(cp, thread_id)
            except Exception:  # pylint: disable=broad-exception-caught # Reason: Final fallback
                pass

    async def _recover_from_incomplete_state(
        self,
        user_msg: HumanMessage,
        config_dict: Dict[str, Any],
        thread_id: str,
    ) -> Dict[str, Any]:
        """Recover from incomplete tool call state by finding safe checkpoint or restarting."""
        logger.warning("Incomplete tool calls detected in thread %s, attempting recovery...", thread_id)

        try:
            safe_config = await self._find_safe_checkpoint_config(config_dict, thread_id)
            if safe_config:
                return await self.invoke_with_timeout({"messages": [user_msg]}, safe_config, None)

            logger.warning("No safe checkpoint found, deleting thread %s and starting fresh", thread_id)
            await self._delete_thread_with_fallback(thread_id)

        except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Recovery must not crash
            logger.error("Error during recovery: %s", e, exc_info=True)
            await self._delete_thread_with_fallback(thread_id)

        return await self.invoke_with_timeout({"messages": [user_msg]}, config_dict, None)

    async def _list_checkpoints(self, checkpointer: Any, config_dict: Dict[str, Any]) -> List[Any]:
        """List checkpoints asynchronously."""
        return [c async for c in checkpointer.alist(config_dict)]

    async def invoke_with_health_checks(
        self,
        user_msg: HumanMessage,
        config_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Invoke agent with health checks and recovery.

        Checks for incomplete tool calls and recovers if needed.
        """
        thread_id = config_dict.get("configurable", {}).get("thread_id")

        if self.checkpointer and thread_id:
            has_incomplete = await self._check_for_incomplete_tool_calls(config_dict)
            if has_incomplete:
                return await self._recover_from_incomplete_state(user_msg, config_dict, thread_id)

        logger.info("Invoking agent for thread %s with checkpointer=%s", thread_id, 'enabled' if self.checkpointer else 'disabled')
        result = await self.invoke_with_timeout({"messages": [user_msg]}, config_dict, None)
        logger.debug("Agent invocation completed for thread %s, checkpoint should be saved automatically by LangGraph", thread_id)
        return result

    async def stream_with_timeout(  # pylint: disable=too-complex # Reason: Timeout/token/stream orchestration across multiple async boundaries
        self,
        messages: Any,
        config_dict: Dict[str, Any],
        publisher: Any,  # StreamPublisher — typed as Any to avoid circular import
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Stream agent events via astream_events, publishing each to Redis.

        Maps LangGraph v2 events:
        - on_tool_start   → TOOL_START
        - on_tool_end     → TOOL_END
        - on_chat_model_end (with tool_calls) → AI_MESSAGE (intermediate)
        - on_chat_model_end (no tool_calls, has content) → captured as final_content

        Returns:
            Dict with "messages" list and "final_content" str (same shape as ainvoke result)
        """
        # pylint: disable=import-outside-toplevel # Reason: Avoids circular import with agents module
        from seer.agents.nexus import _current_thread_id
        from seer.api.agents.workflow.chat_schema import StreamEventType

        thread_id = config_dict.get("configurable", {}).get("thread_id")
        token = _current_thread_id.set(thread_id) if thread_id else None
        timeout_seconds = timeout if timeout is not None else float(config.nexus_chat_timeout_seconds)

        config_with_langfuse = merge_nexus_langfuse_callbacks(config_dict)

        final_content = ""
        all_messages: List[Any] = []

        async def _stream() -> None:
            nonlocal final_content, all_messages
            async for event in self.agent.astream_events(messages, config=config_with_langfuse, version="v2"):
                fc, msgs = await process_stream_event(event, publisher, StreamEventType)
                if fc is not None:
                    final_content = fc
                if msgs is not None:
                    all_messages = msgs

        try:
            await asyncio.wait_for(_stream(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.error("Agent stream timed out after %ss for thread %s", timeout_seconds, thread_id or 'unknown')
            raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Request timeout",
                detail="Request timed out. The agent took too long to respond.",
                status=504,
            )
        finally:
            if token is not None:
                _current_thread_id.reset(token)

        return {"messages": all_messages, "final_content": final_content}

    async def stream_with_health_checks(
        self,
        user_msg: HumanMessage,
        config_dict: Dict[str, Any],
        publisher: Any,  # StreamPublisher — typed as Any to avoid circular import
    ) -> Dict[str, Any]:
        """
        Drop-in streaming replacement for invoke_with_health_checks.

        Health check / recovery paths use ainvoke (rare edge case).
        Main path uses stream_with_timeout for real-time event publishing.
        """
        # pylint: disable=import-outside-toplevel # Reason: Avoids circular import with schema module
        from seer.api.agents.workflow.chat_schema import StreamEventType

        thread_id = config_dict.get("configurable", {}).get("thread_id")

        if self.checkpointer and thread_id:
            has_incomplete = await self._check_for_incomplete_tool_calls(config_dict)
            if has_incomplete:
                # Recovery path — use ainvoke with synthetic events
                await publisher.publish(StreamEventType.AGENT_START, {"recovery": True})
                result = await self._recover_from_incomplete_state(user_msg, config_dict, thread_id)
                return result

        logger.info("Streaming agent for thread %s with checkpointer=%s", thread_id, 'enabled' if self.checkpointer else 'disabled')
        return await self.stream_with_timeout({"messages": [user_msg]}, config_dict, publisher)
