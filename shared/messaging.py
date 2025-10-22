"""
LangGraph SDK messaging helper for direct agent-to-agent communication with persistent remote threads.
"""

from typing import Tuple, Optional

from langgraph_sdk import get_client

from seer.shared.logger import get_logger


logger = get_logger('messaging')


class LangGraphMessenger:
    """Thin wrapper around LangGraph SDK with remote thread persistence per user-thread and agent pair."""

    def __init__(self):
        self._clients: dict[str, any] = {}

    def _client(self, base_url: str):
        if base_url not in self._clients:
            self._clients[base_url] = get_client(url=base_url)
        return self._clients[base_url]

    async def _create_remote_thread(self, base_url: str) -> str:
        client = self._client(base_url)
        resp = await client.threads.create()
        return resp["thread_id"]

    async def ensure_remote_thread(self, user_thread_id: str, src_agent: str, dst_agent: str, base_url: str) -> str:
        remote_tid = await self._create_remote_thread(base_url)
        return remote_tid

    async def send(self,
                   user_thread_id: str,
                   src_agent: str,
                   dst_agent: str,
                   base_url: str,
                   assistant_id: str,
                   content: str,
                   remote_thread_id: Optional[str] = None) -> Tuple[str, str]:
        """Send a message using SDK and return (assistant_text, remote_thread_id)."""
        client = self._client(base_url)
        remote_tid = remote_thread_id or await self.ensure_remote_thread(user_thread_id, src_agent, dst_agent, base_url)

        final_text = ""
        try:
            async for chunk in client.runs.stream(
                thread_id=remote_tid,
                assistant_id=assistant_id,
                input={"messages": [{"role": "user", "content": content}]},
                stream_mode="values"
            ):
                if chunk.event == "values":
                    messages = chunk.data.get("messages", [])
                    if messages and messages[-1].get("type") == "ai":
                        text = messages[-1].get("content", "")
                        if text:
                            final_text = text
        except Exception as e:
            logger.warning(f"Remote thread {remote_tid} failed, recreating: {e}")
            remote_tid = await self._create_remote_thread(base_url)
            async for chunk in client.runs.stream(
                thread_id=remote_tid,
                assistant_id=assistant_id,
                input={"messages": [{"role": "user", "content": content}]},
                stream_mode="values"
            ):
                if chunk.event == "values":
                    messages = chunk.data.get("messages", [])
                    if messages and messages[-1].get("type") == "ai":
                        text = messages[-1].get("content", "")
                        if text:
                            final_text = text

        return final_text, remote_tid


messenger = LangGraphMessenger()


