"""
Event Bus Client - Python library for agents to interact with the event bus
"""

import httpx
import asyncio
from typing import Optional, Callable, Any
import time
from datetime import datetime

from .schemas import EventMessage, EventType


class EventBusClient:
    """Client for agents to interact with the event bus"""
    
    def __init__(
        self,
        agent_name: str,
        bus_url: str = "http://127.0.0.1:8000",
        poll_interval: int = 30
    ):
        """
        Initialize event bus client.
        
        Args:
            agent_name: Unique name for this agent
            bus_url: URL of the event bus server
            poll_interval: How long to wait when polling (seconds)
        """
        self.agent_name = agent_name
        self.bus_url = bus_url.rstrip("/")
        self.poll_interval = poll_interval
        self.client = httpx.AsyncClient(timeout=poll_interval + 5)
        self._running = False
        self._message_handlers: list[Callable[[EventMessage], Any]] = []
    
    async def subscribe(self, filters: Optional[dict] = None):
        """Subscribe this agent to the event bus"""
        try:
            response = await self.client.post(
                f"{self.bus_url}/subscribe/{self.agent_name}",
                json={"filters": filters} if filters else {}
            )
            response.raise_for_status()
            print(f"✅ {self.agent_name} subscribed to event bus")
            return response.json()
        except Exception as e:
            print(f"❌ Failed to subscribe {self.agent_name}: {e}")
            raise
    
    async def publish(self, event: EventMessage):
        """Publish a message to the event bus"""
        try:
            response = await self.client.post(
                f"{self.bus_url}/publish",
                json=event.model_dump(mode='json')
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Failed to publish message: {e}")
            raise
    
    async def poll_once(self) -> list[EventMessage]:
        """Poll for messages once (long-polling)"""
        try:
            response = await self.client.get(
                f"{self.bus_url}/poll/{self.agent_name}",
                params={"timeout": self.poll_interval, "batch_size": 10}
            )
            response.raise_for_status()
            data = response.json()
            
            messages = [EventMessage(**msg) for msg in data["messages"]]
            return messages
        except httpx.TimeoutException:
            # Timeout is normal in long-polling, just return empty
            return []
        except Exception as e:
            print(f"⚠️ Poll error for {self.agent_name}: {e}")
            return []
    
    def add_handler(self, handler: Callable[[EventMessage], Any]):
        """
        Add a message handler function.
        
        Handler should be async and take an EventMessage as input.
        Handler is responsible for filtering messages it cares about.
        """
        self._message_handlers.append(handler)
    
    async def start_listening(self):
        """
        Start listening loop (runs forever until stopped).
        Call this in a background task.
        """
        self._running = True
        print(f"👂 {self.agent_name} started listening to event bus...")
        
        while self._running:
            try:
                messages = await self.poll_once()
                
                for message in messages:
                    # Skip messages from self
                    if message.sender == self.agent_name:
                        continue
                    
                    # Call all handlers
                    for handler in self._message_handlers:
                        try:
                            # Call handler (could be sync or async)
                            if asyncio.iscoroutinefunction(handler):
                                await handler(message)
                            else:
                                handler(message)
                        except Exception as e:
                            print(f"⚠️ Handler error in {self.agent_name}: {e}")
                
            except Exception as e:
                print(f"⚠️ Listening error in {self.agent_name}: {e}")
                await asyncio.sleep(5)  # Back off on errors
    
    def stop_listening(self):
        """Stop the listening loop"""
        self._running = False
        print(f"🛑 {self.agent_name} stopped listening")
    
    async def get_history(
        self,
        since: Optional[datetime] = None,
        event_type: Optional[str] = None,
        thread_id: Optional[str] = None,
        limit: int = 100
    ) -> list[EventMessage]:
        """Get message history from the event bus"""
        params = {"limit": limit}
        if since:
            params["since"] = since.isoformat()
        if event_type:
            params["event_type"] = event_type
        if thread_id:
            params["thread_id"] = thread_id
        
        response = await self.client.get(f"{self.bus_url}/history", params=params)
        response.raise_for_status()
        
        data = response.json()
        return [EventMessage(**msg) for msg in data["messages"]]
    
    async def close(self):
        """Close the client connection"""
        self.stop_listening()
        await self.client.aclose()
    
    # Convenience methods for common event types
    
    async def send_message_to_user(
        self,
        content: str,
        thread_id: Optional[str] = None,
        message_type: str = "info"
    ):
        """Send a message to the user (via Customer Success agent)"""
        event = EventMessage(
            event_type=EventType.MESSAGE_TO_USER,
            sender=self.agent_name,
            thread_id=thread_id,
            payload={
                "content": content,
                "message_type": message_type
            }
        )
        await self.publish(event)
    
    async def announce_started(self, agent_type: str, capabilities: list[str]):
        """Announce that this agent has started"""
        event = EventMessage(
            event_type=EventType.AGENT_STARTED,
            sender=self.agent_name,
            payload={
                "agent_name": self.agent_name,
                "agent_type": agent_type,
                "capabilities": capabilities
            }
        )
        await self.publish(event)

