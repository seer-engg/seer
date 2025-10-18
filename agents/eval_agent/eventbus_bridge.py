#!/usr/bin/env python3
"""
Eval Agent - Event Bus Bridge
Connects the LangGraph Eval Agent to the Event Bus
"""

import asyncio
import httpx
import os
import sys
import json
import uuid
import hashlib
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from event_bus.client import EventBusClient
from event_bus.schemas import EventMessage, EventType


class EvalAgentBridge:
    """Bridge between Event Bus and Eval LangGraph agent"""
    
    def __init__(self):
        self.langgraph_url = os.getenv("LANGGRAPH_URL", "http://localhost:8002")
        self.event_bus_url = os.getenv("EVENT_BUS_URL", "http://127.0.0.1:8000")
        self.agent_name = "eval_agent"
        
        self.event_bus = EventBusClient(self.agent_name, self.event_bus_url)
        self.http_client = httpx.AsyncClient(timeout=120.0)
        self.active_threads = {}  # thread_id -> context
        
    async def invoke_agent(self, message: str, thread_id: str) -> dict:
        """Invoke the LangGraph agent and return parsed response with tool outputs"""
        url = f"{self.langgraph_url}/runs/stream"
        
        payload = {
            "assistant_id": "eval_agent",
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
        """Handle MessageFromUser event - Listen directly to user messages"""
        print(f"📨 Received MessageFromUser", flush=True)
        
        user_message = event.payload.get("content", "")
        thread_id = event.thread_id or str(uuid.uuid4())
        
        # Validate thread_id format
        try:
            uuid.UUID(thread_id)
        except ValueError:
            thread_id = str(uuid.UUID(hashlib.md5(thread_id.encode()).hexdigest()))
        
        # Check if this is an evaluation request (heuristic: contains "evaluate", "test", "agent", etc.)
        lower_msg = user_message.lower()
        is_eval_request = any(keyword in lower_msg for keyword in ["evaluate", "test", "check", "assess", "run tests"])
        
        if not is_eval_request:
            print(f"⏭️  Not an eval request, ignoring", flush=True)
            return
        
        print(f"🎯 Detected evaluation request", flush=True)
        
        # Parse the message to extract agent URL, ID, and expectations
        # Simple heuristic parsing (can be improved with NLP)
        import re
        
        # Try to find URL patterns
        url_pattern = r'(?:localhost:|http://localhost:)(\d+)'
        url_match = re.search(url_pattern, user_message)
        target_url = f"http://localhost:{url_match.group(1)}" if url_match else "http://localhost:2024"
        
        # Try to find agent ID
        id_pattern = r'\(ID:\s*([^\)]+)\)'
        id_match = re.search(id_pattern, user_message, re.IGNORECASE)
        target_id = id_match.group(1).strip() if id_match else "agent"
        
        # Store context
        self.active_threads[thread_id] = {
            "target_url": target_url,
            "target_id": target_id,
            "expectations": user_message
        }
        
        # Build message for agent
        message = f"""New evaluation request from user:
- Target agent: {target_url}
- Agent ID: {target_id}
- User message: {user_message}

Please generate a spec and test cases based on the user's expectations."""
        
        # Invoke agent
        result = await self.invoke_agent(message, thread_id)
        
        # Process tool outputs
        eval_suite_data = None
        if result["tool_outputs"]:
            print(f"🔧 Agent used {len(result['tool_outputs'])} tool(s)", flush=True)
            
            # Track which actions we've already processed to avoid duplicates
            processed_actions = set()
            
            for tool_output in result["tool_outputs"]:
                try:
                    data = json.loads(tool_output)
                    
                    # Check if this is a generated test suite (from generate_tests tool)
                    if "test_cases" in data and "id" in data:
                        eval_suite_data = data
                        print(f"📋 Generated eval suite: {data.get('id')}", flush=True)
                        # Persist in thread context for post-confirmation execution
                        try:
                            if thread_id in self.active_threads:
                                self.active_threads[thread_id]["eval_suite"] = data
                        except Exception:
                            pass
                        continue
                    
                    action = data.get("action")
                    
                    # Create unique key for deduplication
                    action_key = f"{action}:{thread_id}"
                    
                    if action == "CONFIRMATION_REQUEST" and action_key not in processed_actions:
                        # Store eval suite in Event Bus and locally before requesting confirmation
                        local_file_path = None
                        evals_markdown = None
                        if eval_suite_data:
                            await self._store_eval_suite(
                                target_url,
                                target_id,
                                eval_suite_data
                            )
                            # Store eval_suite_id in context for later reference
                            self.active_threads[thread_id]["eval_suite_id"] = eval_suite_data.get("id")
                            # Save locally for convenience/auditing
                            try:
                                local_file_path = self._save_eval_suite_local(eval_suite_data)
                                print(f"📝 Saved eval suite locally: {local_file_path}", flush=True)
                                # Keep path in context for fallback loading
                                self.active_threads[thread_id]["local_file_path"] = local_file_path
                            except Exception as e:
                                print(f"⚠️  Failed to save eval suite locally: {e}", flush=True)
                            
                            # Build a human-readable list of all evals (full display)
                            test_cases = eval_suite_data.get("test_cases", [])
                            lines = []
                            for idx, tc in enumerate(test_cases, 1):
                                lines.append(f"{idx}. [{tc.get('id','')}] Expectation: {tc.get('expectation_ref','')}")
                                lines.append(f"   - Input: {tc.get('input_message','')}")
                                lines.append(f"   - Expected: {tc.get('expected_behavior','')}")
                                lines.append(f"   - Criteria: {tc.get('success_criteria','')}")
                            evals_markdown = "\n".join(lines)
                        
                        # Publish UserConfirmationQuery (include full evals in question/context)
                        # Customer Success will relay this to the user
                        test_count = data.get("test_count", 0)
                        question = data.get("question", "Should I proceed?")
                        
                        if evals_markdown:
                            question_text = (
                                f"I've generated {len(eval_suite_data.get('test_cases', []))} test cases.\n\n"
                                f"Here are the generated evals:\n{evals_markdown}\n\n"
                                f"{question}"
                            )
                        else:
                            question_text = f"I've generated {test_count} test cases. {question}"
                        
                        await self.event_bus.publish(EventMessage(
                            event_type=EventType.USER_CONFIRMATION_QUERY,
                            sender=self.agent_name,
                            thread_id=thread_id,
                            payload={
                                "question": question_text,
                                "context": {
                                    "test_count": test_count,
                                    "eval_suite_id": (eval_suite_data or {}).get("id"),
                                    "test_cases": (eval_suite_data or {}).get("test_cases", []),
                                    "local_file_path": local_file_path
                                }
                            }
                        ))
                        processed_actions.add(action_key)
                        print(f"✉️  Published UserConfirmationQuery", flush=True)
                    elif action == "CONFIRMATION_REQUEST":
                        print(f"⚠️  Skipping duplicate CONFIRMATION_REQUEST", flush=True)
                
                except json.JSONDecodeError:
                    print(f"⚠️  Tool output is not JSON: {tool_output[:50]}...", flush=True)
    
    async def _store_eval_suite(self, target_url: str, target_id: str, eval_suite: dict):
        """Store eval suite in Event Bus"""
        try:
            response = await self.http_client.post(
                f"{self.event_bus_url}/evals",
                json={
                    "eval_suite": eval_suite,
                    "target_agent_url": target_url,
                    "target_agent_id": target_id
                }
            )
            response.raise_for_status()
            print(f"✅ Stored eval suite in Event Bus: {eval_suite.get('id')}", flush=True)
        except Exception as e:
            print(f"⚠️  Failed to store eval suite: {e}", flush=True)

    def _save_eval_suite_local(self, eval_suite: dict) -> str:
        """Save eval suite to a local JSON file and return absolute file path"""
        base_dir = Path(__file__).parent / "generated_evals"
        base_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{eval_suite.get('id', 'eval_suite')}.json"
        file_path = base_dir / file_name
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(eval_suite, f, indent=2)
        return str(file_path.resolve())
    
    async def handle_user_confirmation(self, event: EventMessage):
        """Handle UserConfirmation event"""
        print(f"📨 Received UserConfirmation", flush=True)
        
        thread_id = event.thread_id or str(uuid.uuid4())
        
        # Validate thread_id format
        try:
            uuid.UUID(thread_id)
        except ValueError:
            thread_id = str(uuid.UUID(hashlib.md5(thread_id.encode()).hexdigest()))
        
        confirmed = event.payload.get("confirmed", False)
        
        if not confirmed:
            print(f"❌ User declined", flush=True)
            if thread_id in self.active_threads:
                del self.active_threads[thread_id]
            return
        
        print(f"✅ User confirmed, running tests...", flush=True)
        
        # Provide eval suite and target details back to the agent so it can run tests
        ctx = self.active_threads.get(thread_id, {})
        target_url = ctx.get("target_url", "http://localhost:2024")
        target_id = ctx.get("target_id", "agent")
        eval_suite = ctx.get("eval_suite")
        if not eval_suite:
            # Fallback: load from local file path saved earlier
            local_path = ctx.get("local_file_path")
            try:
                if local_path and os.path.exists(local_path):
                    with open(local_path, "r", encoding="utf-8") as f:
                        eval_suite = json.load(f)
            except Exception as e:
                print(f"⚠️  Failed to load eval suite from local file: {e}", flush=True)
        
        if eval_suite:
            message = (
                "User confirmed. Use this eval suite to run tests and then summarize results.\n"
                f"Target URL: {target_url}\n"
                f"Target Agent ID: {target_id}\n"
                "Instructions:\n"
                "- For each test_case in eval_suite.test_cases:\n"
                "  * Call run_test(target_url, target_agent_id, test_case.input_message, same thread_id).\n"
                "  * Then call judge_test_result(test_input, expected_behavior, success_criteria, actual_output).\n"
                "- After all tests, call summarize_results(passed, failed, total) to emit TEST_RESULTS.\n"
                "EVAL_SUITE_JSON:\n" + json.dumps(eval_suite)
            )
        else:
            # Fallback to simple proceed instruction (agent may regenerate tests if needed)
            message = "User confirmed. Please proceed with running the tests and summarizing results."
        
        # Invoke agent
        result = await self.invoke_agent(message, thread_id)
        
        # Process tool outputs
        if result["tool_outputs"]:
            print(f"🔧 Agent used {len(result['tool_outputs'])} tool(s)", flush=True)
            
            # Track which actions we've already processed to avoid duplicates
            processed_actions = set()
            
            for tool_output in result["tool_outputs"]:
                try:
                    data = json.loads(tool_output)
                    action = data.get("action")
                    
                    # Create unique key for deduplication
                    action_key = f"{action}:{thread_id}"
                    
                    if action == "TEST_RESULTS" and action_key not in processed_actions:
                        # Publish TestResultsReady
                        # Customer Success will relay this to the user
                        passed = data.get("passed", 0)
                        failed = data.get("failed", 0)
                        total = data.get("total", 0)
                        score = data.get("score", 0)
                        summary = data.get("summary", "")
                        
                        details_message = f"📊 Evaluation Complete!\n\n**Results:** {summary}\n**Score:** {score:.0f}%\n"
                        if failed > 0:
                            details_message += f"\n⚠️ {failed} test(s) failed."
                        
                        await self.event_bus.publish(EventMessage(
                            event_type=EventType.TEST_RESULTS_READY,
                            sender=self.agent_name,
                            thread_id=thread_id,
                            payload={
                                "summary": details_message,
                                "passed": passed,
                                "failed": failed,
                                "total": total,
                                "overall_score": score,
                                "details": {}
                            }
                        ))
                        processed_actions.add(action_key)
                        print(f"✉️  Published TestResultsReady (CS will relay to user)", flush=True)
                        
                        # Clean up
                        if thread_id in self.active_threads:
                            del self.active_threads[thread_id]
                    elif action == "TEST_RESULTS":
                        print(f"⚠️  Skipping duplicate TEST_RESULTS", flush=True)
                
                except json.JSONDecodeError:
                    print(f"⚠️  Tool output is not JSON: {tool_output[:50]}...", flush=True)
    
    async def start(self):
        """Start the bridge"""
        print("=" * 60)
        print(f"🚀 Eval Agent Bridge Starting")
        print(f"   LangGraph: {self.langgraph_url}")
        print(f"   Event Bus: {self.event_bus_url}")
        print("=" * 60)
        
        # Subscribe to Event Bus
        await self.event_bus.subscribe()
        self.event_bus.add_handler(self._handle_event)
        
        # Announce started
        await self.event_bus.announce_started(
            agent_type="eval_agent",
            capabilities=["spec_generation", "test_generation", "a2a_testing"]
        )
        
        print(f"✅ Bridge ready and listening")
        print("=" * 60)
        
        # Start listening
        await self.event_bus.start_listening()
    
    async def _handle_event(self, event: EventMessage):
        """Route events to handlers"""
        if event.event_type == EventType.MESSAGE_FROM_USER:
            await self.handle_message_from_user(event)
        elif event.event_type == EventType.USER_CONFIRMATION:
            await self.handle_user_confirmation(event)
    
    async def stop(self):
        """Stop the bridge"""
        await self.http_client.aclose()
        await self.event_bus.close()


async def main():
    """Main entry point"""
    bridge = EvalAgentBridge()
    
    try:
        await bridge.start()
    except KeyboardInterrupt:
        print("\n🛑 Stopping bridge...")
        await bridge.stop()


if __name__ == "__main__":
    asyncio.run(main())

