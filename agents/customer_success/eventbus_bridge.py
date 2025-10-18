#!/usr/bin/env python3
"""
Customer Success Agent - Event Bus Bridge
Connects the LangGraph agent to the Event Bus
"""

import asyncio
import httpx
import os
import sys
import json
import uuid
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from event_bus.client import EventBusClient
from event_bus.schemas import EventMessage, EventType


class CustomerSuccessBridge:
    """Bridge between Event Bus and Customer Success LangGraph agent"""
    
    def __init__(self):
        self.langgraph_url = os.getenv("LANGGRAPH_URL", "http://localhost:8001")
        self.event_bus_url = os.getenv("EVENT_BUS_URL", "http://127.0.0.1:8000")
        self.agent_name = "customer_success"
        
        self.event_bus = EventBusClient(self.agent_name, self.event_bus_url)
        self.http_client = httpx.AsyncClient(timeout=60.0)
        
    async def invoke_agent(self, message: str, thread_id: str) -> dict:
        """Invoke the LangGraph agent and return parsed response with tool outputs"""
        url = f"{self.langgraph_url}/runs/stream"
        
        payload = {
            "assistant_id": "customer_success",
            "input": {
                "messages": [{"role": "user", "content": message}]
            },
            "stream_mode": ["values"],
            "config": {"configurable": {"thread_id": thread_id}}
        }
        
        print(f"🔄 Invoking agent: {message[:80]}...", flush=True)
        
        response = await self.http_client.post(url, json=payload)
        response.raise_for_status()
        
        # Parse all messages from stream
        ai_response = ""
        tool_outputs = []
        
        for line in response.text.strip().split('\n'):
            if line.startswith('data: '):
                data = json.loads(line[6:])
                if 'messages' in data:
                    for msg in data['messages']:
                        if msg.get('type') == 'ai':
                            ai_response = msg.get('content', '')
                        elif msg.get('type') == 'tool':
                            # This is a tool output!
                            tool_outputs.append(msg.get('content', ''))
        
        return {
            "ai_response": ai_response,
            "tool_outputs": tool_outputs
        }
    
    async def handle_message_from_user(self, event: EventMessage):
        """Handle MessageFromUser event"""
        print(f"📨 Received MessageFromUser", flush=True)
        
        user_message = event.payload.get("content", "")
        thread_id = event.thread_id or str(uuid.uuid4())
        
        # Invoke agent (LangGraph dev server handles thread persistence automatically)
        result = await self.invoke_agent(user_message, thread_id)
        
        # Check if agent used tools
        if result["tool_outputs"]:
            print(f"🔧 Agent used {len(result['tool_outputs'])} tool(s)", flush=True)
            
            # Track which actions we've already processed to avoid duplicates
            processed_actions = set()
            
            for tool_output in result["tool_outputs"]:
                try:
                    # Parse tool output as JSON
                    data = json.loads(tool_output)
                    action = data.get("action")
                    
                    # Create unique key for deduplication
                    action_key = f"{action}:{thread_id}"
                    
                    # REMOVED: InitialAgentQuery publishing - Eval Agent now listens to MessageFromUser directly
                    # No need for CS to relay/transform messages between user and eval agent
                    
                    if action == "CONFIRMATION" and action_key not in processed_actions:
                        # Publish UserConfirmation
                        await self.event_bus.publish(EventMessage(
                            event_type=EventType.USER_CONFIRMATION,
                            sender=self.agent_name,
                            thread_id=thread_id,
                            payload={
                                "confirmed": data.get("confirmed"),
                                "answer": data.get("details", "")
                            }
                        ))
                        processed_actions.add(action_key)
                        print(f"✉️  Published UserConfirmation", flush=True)
                    elif action == "CONFIRMATION":
                        print(f"⚠️  Skipping duplicate CONFIRMATION", flush=True)
                
                except json.JSONDecodeError:
                    print(f"⚠️  Tool output is not JSON: {tool_output[:50]}...", flush=True)
        
        # Send AI response to user
        if result["ai_response"]:
            await self.event_bus.publish(EventMessage(
                event_type=EventType.MESSAGE_TO_USER,
                sender=self.agent_name,
                thread_id=thread_id,
                payload={
                    "content": result["ai_response"],
                    "message_type": "info"
                }
            ))
            print(f"💬 Sent response to user", flush=True)
    
    async def relay_confirmation_query(self, event: EventMessage):
        """Relay UserConfirmationQuery from Eval Agent to user"""
        print(f"📨 Relaying UserConfirmationQuery to user", flush=True)
        
        question = event.payload.get("question", "")
        thread_id = event.thread_id or str(uuid.uuid4())
        
        # Add to CS agent's thread history for context (but don't generate a response)
        # We use a special format to indicate this is an assistant message from another agent
        url = f"{self.langgraph_url}/threads/{thread_id}/state"
        try:
            response = await self.http_client.get(url)
            if response.status_code == 200:
                # Thread exists, add message to history
                update_url = f"{self.langgraph_url}/threads/{thread_id}/state"
                await self.http_client.post(update_url, json={
                    "values": {
                        "messages": [{"role": "assistant", "content": f"[Eval Agent]: {question}"}]
                    }
                })
                print(f"📝 Added to CS thread history", flush=True)
        except Exception as e:
            print(f"⚠️ Could not update thread history: {e}", flush=True)
        
        # Relay verbatim to user
        await self.event_bus.publish(EventMessage(
            event_type=EventType.MESSAGE_TO_USER,
            sender=self.agent_name,
            thread_id=thread_id,
            payload={
                "content": question,
                "message_type": "question"
            }
        ))
        print(f"✉️  Relayed confirmation query verbatim to user", flush=True)
    
    async def relay_test_results(self, event: EventMessage):
        """Relay TestResultsReady from Eval Agent to user"""
        print(f"📨 Relaying TestResultsReady to user", flush=True)
        
        summary = event.payload.get("summary", "Test results ready")
        thread_id = event.thread_id or str(uuid.uuid4())
        
        # Add to CS agent's thread history for context (but don't generate a response)
        url = f"{self.langgraph_url}/threads/{thread_id}/state"
        try:
            response = await self.http_client.get(url)
            if response.status_code == 200:
                # Thread exists, add message to history
                update_url = f"{self.langgraph_url}/threads/{thread_id}/state"
                await self.http_client.post(update_url, json={
                    "values": {
                        "messages": [{"role": "assistant", "content": f"[Eval Agent]: {summary}"}]
                    }
                })
                print(f"📝 Added test results to CS thread history", flush=True)
        except Exception as e:
            print(f"⚠️ Could not update thread history: {e}", flush=True)
        
        # Relay verbatim to user
        await self.event_bus.publish(EventMessage(
            event_type=EventType.MESSAGE_TO_USER,
            sender=self.agent_name,
            thread_id=thread_id,
            payload={
                "content": summary,
                "message_type": "success"
            }
        ))
        print(f"✉️  Relayed test results verbatim to user", flush=True)
    
    async def start(self):
        """Start the bridge"""
        print("=" * 60)
        print(f"🚀 Customer Success Bridge Starting")
        print(f"   LangGraph: {self.langgraph_url}")
        print(f"   Event Bus: {self.event_bus_url}")
        print("=" * 60)
        
        # Subscribe to Event Bus
        await self.event_bus.subscribe()
        self.event_bus.add_handler(self._handle_event)
        
        # Announce started
        await self.event_bus.announce_started(
            agent_type="customer_success",
            capabilities=["user_interaction", "message_relay"]
        )
        
        print(f"✅ Bridge ready and listening")
        print("=" * 60)
        
        # Start listening
        await self.event_bus.start_listening()
    
    async def _handle_event(self, event: EventMessage):
        """Route events to handlers"""
        if event.event_type == EventType.MESSAGE_FROM_USER:
            await self.handle_message_from_user(event)
        elif event.event_type == EventType.USER_CONFIRMATION_QUERY:
            await self.relay_confirmation_query(event)
        elif event.event_type == EventType.TEST_RESULTS_READY:
            await self.relay_test_results(event)
    
    async def stop(self):
        """Stop the bridge"""
        await self.http_client.aclose()
        await self.event_bus.close()


async def main():
    """Main entry point"""
    bridge = CustomerSuccessBridge()
    
    try:
        await bridge.start()
    except KeyboardInterrupt:
        print("\n🛑 Stopping bridge...")
        await bridge.stop()


if __name__ == "__main__":
    asyncio.run(main())

