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
import re
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from event_bus.client import EventBusClient
from event_bus.schemas import EventMessage, EventType
from shared.database import Database


class CustomerSuccessBridge:
    """Bridge between Event Bus and Customer Success LangGraph agent"""
    
    def __init__(self):
        self.langgraph_url = os.getenv("LANGGRAPH_URL", "http://localhost:8001")
        self.event_bus_url = os.getenv("EVENT_BUS_URL", "http://127.0.0.1:8000")
        self.agent_name = "customer_success"
        
        self.event_bus = EventBusClient(self.agent_name, self.event_bus_url)
        self.http_client = httpx.AsyncClient(timeout=60.0)
        self.db = Database()
    
    def _extract_config_from_message(self, message: str) -> dict:
        """Extract GitHub URL and agent config from user message"""
        config = {
            'github_url': None,
            'agent_host': None,
            'agent_port': None,
            'agent_id': None
        }
        
        # Extract GitHub URL
        github_patterns = [
            r'github\.com/([^\s\)]+)',
            r'https?://github\.com/([^\s\)]+)',
            r'git@github\.com:([^\s\)]+)',
        ]
        for pattern in github_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                github_path = match.group(1).rstrip('/')
                config['github_url'] = f"https://github.com/{github_path}"
                break
        
        # Extract agent host and port
        # Try full URL patterns first
        url_patterns = [
            r'(?:http://)?localhost:(\d+)',
            r'(?:http://)?127\.0\.0\.1:(\d+)',
            r'(?:http://)?([\w\.-]+):(\d+)',
        ]
        for pattern in url_patterns:
            match = re.search(pattern, message)
            if match:
                if len(match.groups()) == 1:
                    # localhost:port or 127.0.0.1:port
                    config['agent_host'] = 'localhost'
                    config['agent_port'] = int(match.group(1))
                else:
                    # host:port
                    config['agent_host'] = match.group(1)
                    config['agent_port'] = int(match.group(2))
                break
        
        # Extract agent ID
        id_patterns = [
            r'\(ID:\s*([^\)]+)\)',
            r'agent[_ ]id[:\s]+([^\s,]+)',
            r'id[:\s]+([a-zA-Z0-9_-]+)',
        ]
        for pattern in id_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                config['agent_id'] = match.group(1).strip()
                break
        
        return config
        
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
        
        # Check if this is the first message in the thread by checking if config exists
        existing_config = await self.db.get_thread_config(thread_id)
        is_first_message = existing_config is None or all(v is None for v in existing_config.values())
        
        # If first message, extract and store GitHub URL and agent config
        if is_first_message:
            print(f"🔍 First message in thread, extracting config...", flush=True)
            config = self._extract_config_from_message(user_message)
            
            # Store config if any values were extracted
            if any(config.values()):
                await self.db.update_thread_config(
                    thread_id=thread_id,
                    github_url=config['github_url'],
                    agent_host=config['agent_host'],
                    agent_port=config['agent_port'],
                    agent_id=config['agent_id']
                )
                print(f"✅ Stored thread config: github={config['github_url']}, "
                      f"agent={config['agent_host']}:{config['agent_port']}, "
                      f"id={config['agent_id']}", flush=True)
            else:
                print(f"⚠️  No config found in message", flush=True)
        
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
        
        # Initialize database
        await self.db.init()
        print(f"✅ Database initialized", flush=True)
        
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
        await self.db.close()


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

