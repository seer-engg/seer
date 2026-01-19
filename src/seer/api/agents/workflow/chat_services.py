"""
Services for workflow chat endpoint to reduce complexity.
"""
import asyncio
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from seer.api.core.errors import VALIDATION_PROBLEM, raise_problem
from seer.logger import get_logger

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
        elif session_id:
            session = await get_chat_session(session_id, workflow)
            thread_id = session.thread_id
        else:
            thread_id = f"workflow-{workflow.workflow_id}-{uuid.uuid4().hex}"
            session = await create_chat_session(
                workflow=workflow,
                user=user,
                thread_id=thread_id,
            )
            session_id = session.id

        if session is None:
            thread_id = thread_id or f"workflow-{workflow.workflow_id}-{uuid.uuid4().hex}"
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
        except Exception as e:
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
                    tool_call_ids.add(tool_call_id)

        return tool_call_ids

    @staticmethod
    def extract_tool_response_ids(messages: List[Any]) -> Set[str]:
        """Extract tool response IDs from a list of messages."""
        tool_response_ids = set()

        for m in messages:
            if isinstance(m, ToolMessage):
                tool_call_id = getattr(m, "tool_call_id", None)
                if tool_call_id:
                    tool_response_ids.add(tool_call_id)
            elif isinstance(m, dict) and m.get("type") == "tool":
                tool_call_id = m.get("tool_call_id")
                if tool_call_id:
                    tool_response_ids.add(tool_call_id)

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
                    tool_call_ids.add(tool_call_id)

        return tool_call_ids

    @staticmethod
    def _extract_checkpoint_response_ids(messages: List[Any]) -> Set[str]:
        """Extract tool response IDs from checkpoint messages."""
        response_ids = set()

        for m in messages:
            if isinstance(m, ToolMessage):
                tool_call_id = getattr(m, "tool_call_id", None)
                if tool_call_id:
                    response_ids.add(tool_call_id)
            elif isinstance(m, dict):
                m_type = m.get("type") or m.get("role", "")
                if m_type == "tool":
                    tool_call_id = m.get("tool_call_id")
                    if tool_call_id:
                        response_ids.add(tool_call_id)

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
            if hasattr(current_state, "interrupt") and current_state.interrupt:
                interrupt = current_state.interrupt
                interrupt_data = None

                if isinstance(interrupt, list) and len(interrupt) > 0:
                    first_interrupt = interrupt[0]
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
                elif isinstance(interrupt, dict):
                    interrupt_data = interrupt
                else:
                    interrupt_data = {"value": interrupt}

                return True, interrupt_data
        except Exception as e:
            logger.debug("Could not check state for interrupts: %s", e)

        return False, None


class ChatOrchestrator:
    """Orchestrates agent invocation with health checks and recovery."""

    def __init__(
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
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        """Invoke agent with timeout."""
        from seer.agents.workflow_agent import (
            _current_thread_id,  # pylint: disable=import-outside-toplevel # Reason: Avoids circular import with agents module
        )

        thread_id = config_dict.get("configurable", {}).get("thread_id")
        token = _current_thread_id.set(thread_id) if thread_id else None

        try:
            return await asyncio.wait_for(
                self.agent.ainvoke(messages, config=config_dict),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.error("Agent invocation timed out after %ss for thread %s", timeout, thread_id or 'unknown')
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
        from seer.api.agents.checkpointer import (
            get_checkpointer_with_retry,  # pylint: disable=import-outside-toplevel # Reason: Avoids circular import with checkpointer module
        )

        thread_id = config_dict.get("configurable", {}).get("thread_id")
        logger.debug("Checking checkpointer health for thread %s", thread_id)

        try:
            checkpointer = await get_checkpointer_with_retry()
            if checkpointer is None:
                logger.warning("Checkpointer unavailable, proceeding without state check")
                return False

            current_state = await self.agent.aget_state(config_dict)
            messages = current_state.values.get("messages", [])
            return self.detector.has_incomplete_tool_calls(messages)

        except (Exception, ConnectionError, EOFError) as e:
            if self.health_service.is_connection_error(e):
                logger.warning("Connection error during state check: %s, attempting reconnection...", e)
                return await self._retry_incomplete_check_after_reconnect(config_dict)
            logger.warning("Error checking state for incomplete tool calls: %s. Proceeding with normal invocation.", e)
            return False

    async def _retry_incomplete_check_after_reconnect(self, config_dict: Dict[str, Any]) -> bool:
        """Retry incomplete tool call check after reconnection."""
        try:
            checkpointer = await self.reconnect_func()
            if not checkpointer:
                logger.warning("Failed to recreate checkpointer, proceeding without state check")
                return False

            current_state = await self.agent.aget_state(config_dict)
            messages = current_state.values.get("messages", [])
            return self.detector.has_incomplete_tool_calls(messages)

        except Exception as retry_error:
            logger.error("State check failed after reconnection: %s", retry_error)
            return False

    async def _recover_from_incomplete_state(
        self,
        user_msg: HumanMessage,
        config_dict: Dict[str, Any],
        thread_id: str,
    ) -> Dict[str, Any]:
        """Recover from incomplete tool call state."""
        from seer.api.agents.checkpointer import (
            get_checkpointer_with_retry,  # pylint: disable=import-outside-toplevel # Reason: Avoids circular import with checkpointer module
        )

        logger.warning("Incomplete tool calls detected in thread %s, attempting recovery...", thread_id)

        if not self.checkpointer:
            logger.error("No checkpointer available for state recovery")
            return await self.invoke_with_timeout({"messages": [user_msg]}, config_dict, None)

        try:
            checkpointer = await get_checkpointer_with_retry()
            if checkpointer is None:
                return await self._delete_thread_and_restart(user_msg, config_dict, thread_id)

            return await self._try_safe_checkpoint_or_restart(user_msg, config_dict, thread_id, checkpointer)

        except (Exception, ConnectionError, EOFError) as e:
            if self.health_service.is_connection_error(e):
                return await self._handle_recovery_connection_error(user_msg, config_dict, thread_id, e)
            return await self._fallback_delete_and_restart(user_msg, config_dict, thread_id, e)

    async def _delete_thread_and_restart(
        self,
        user_msg: HumanMessage,
        config_dict: Dict[str, Any],
        thread_id: str,
    ) -> Dict[str, Any]:
        """Delete thread and restart with fresh state."""
        from seer.api.agents.checkpointer import (
            get_checkpointer,  # pylint: disable=import-outside-toplevel # Reason: Avoids circular import with checkpointer module
        )

        logger.warning("Checkpointer unavailable for checkpoint recovery, deleting thread and starting fresh")
        fresh_checkpointer = await get_checkpointer()
        if fresh_checkpointer:
            await self.recovery_service.delete_thread(fresh_checkpointer, thread_id)
        return await self.invoke_with_timeout({"messages": [user_msg]}, config_dict, None)

    async def _try_safe_checkpoint_or_restart(
        self,
        user_msg: HumanMessage,
        config_dict: Dict[str, Any],
        thread_id: str,
        checkpointer: Any,
    ) -> Dict[str, Any]:
        """Try to resume from safe checkpoint or restart."""
        try:
            checkpoints = await asyncio.wait_for(
                self._list_checkpoints(checkpointer, config_dict),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            logger.error("Checkpoint listing timed out for thread %s", thread_id)
            checkpoints = []

        safe_checkpoint = self.recovery_service.find_safe_checkpoint(checkpoints)

        if safe_checkpoint:
            prev_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": safe_checkpoint.config["configurable"]["checkpoint_id"],
                }
            }
            logger.info("Resuming from safe checkpoint: %s", prev_config['configurable']['checkpoint_id'])
            return await self.invoke_with_timeout({"messages": [user_msg]}, prev_config, None)

        logger.warning("No safe checkpoint found, deleting thread %s and starting fresh", thread_id)
        await self.recovery_service.delete_thread(checkpointer, thread_id)
        return await self.invoke_with_timeout({"messages": [user_msg]}, config_dict, None)

    async def _list_checkpoints(self, checkpointer: Any, config_dict: Dict[str, Any]) -> List[Any]:
        """List checkpoints asynchronously."""
        return [c async for c in checkpointer.alist(config_dict)]

    async def _handle_recovery_connection_error(
        self,
        user_msg: HumanMessage,
        config_dict: Dict[str, Any],
        thread_id: str,
        error: Exception,
    ) -> Dict[str, Any]:
        """Handle connection error during recovery."""
        logger.warning("Connection error during checkpoint recovery: %s, attempting reconnection...", error)
        try:
            checkpointer = await self.reconnect_func()
            if checkpointer:
                await self.recovery_service.delete_thread(checkpointer, thread_id)
                return await self.invoke_with_timeout({"messages": [user_msg]}, config_dict, None)
            logger.error("Failed to recreate checkpointer, proceeding without deletion")
            return await self.invoke_with_timeout({"messages": [user_msg]}, config_dict, None)
        except Exception as reconnect_error:
            logger.error("Error during checkpointer reconnection in recovery: %s", reconnect_error)
            return await self.invoke_with_timeout({"messages": [user_msg]}, config_dict, None)

    async def _fallback_delete_and_restart(
        self,
        user_msg: HumanMessage,
        config_dict: Dict[str, Any],
        thread_id: str,
        error: Exception,
    ) -> Dict[str, Any]:
        """Fallback: delete thread and restart."""
        from seer.api.agents.checkpointer import (
            get_checkpointer,  # pylint: disable=import-outside-toplevel # Reason: Avoids circular import with checkpointer module
        )

        logger.error("Error recovering from incomplete state: %s", error, exc_info=True)
        fresh_checkpointer = await get_checkpointer()
        if fresh_checkpointer:
            await self.recovery_service.delete_thread(fresh_checkpointer, thread_id)
        return await self.invoke_with_timeout({"messages": [user_msg]}, config_dict, None)

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
