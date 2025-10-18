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
import traceback

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
        
    def _to_plain(self, obj):
        """Convert Pydantic objects or nested structures to plain dicts/lists for JSON."""
        try:
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
        except Exception:
            pass
        if isinstance(obj, list):
            return [self._to_plain(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._to_plain(v) for k, v in obj.items()}
        return obj

    def _collect_known_state_from_obj(self, obj, acc=None):
        """Recursively collect known state keys from arbitrary nested JSON objects."""
        if acc is None:
            acc = {}
        known = {
            "agent_name",
            "agent_url",
            "expectations",
            "spec",
            "eval_suite",
            "current_test_index",
            "passed",
            "failed",
            "total",
            "test_results",
            "messages",
        }
        try:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in known:
                        acc[k] = v
                    # Recurse into dicts/lists
                    if isinstance(v, (dict, list)):
                        self._collect_known_state_from_obj(v, acc)
            elif isinstance(obj, list):
                for item in obj:
                    self._collect_known_state_from_obj(item, acc)
        except Exception:
            pass
        return acc

    def _extract_values_from_sse_obj(self, obj):
        """Attempt to extract state 'values' from various possible SSE shapes."""
        if not isinstance(obj, dict):
            return {}
        # Common shapes
        if isinstance(obj.get("values"), dict):
            return obj.get("values", {})
        if isinstance(obj.get("value"), dict):
            return obj.get("value", {})
        data = obj.get("data")
        if isinstance(data, dict):
            # Nested under data
            if isinstance(data.get("values"), dict):
                return data.get("values", {})
            if isinstance(data.get("value"), dict):
                return data.get("value", {})
            # Some backends may put state under 'state'
            if isinstance(data.get("state"), dict):
                return data.get("state", {})
        # Fallback: pick known state keys anywhere in the object
        return self._collect_known_state_from_obj(obj)

    async def invoke_agent(self, message: str, thread_id: str) -> dict:
        """Invoke the LangGraph agent and return final state values from the values stream."""
        url = f"{self.langgraph_url}/runs/stream"
        
        payload = {
            "assistant_id": "eval_agent",
            "input": {
                "messages": [{"role": "user", "content": message}]
            },
            "stream_mode": ["values"],
            "config": {"configurable": {"thread_id": thread_id}}
        }
        
        print(f"🔄 Invoking agent: {message[:80]}... (thread={thread_id})", flush=True)
        try:
            response = await self.http_client.post(url, json=payload)
            
            print(f"⬅️  Agent HTTP status: {response.status_code}", flush=True)
            response.raise_for_status()
        except Exception as e:
            print(f"❌ Agent HTTP error: {e}", flush=True)
            traceback.print_exc()
            return {"values": {}}
        
        # Capture last known values from the stream (state variables only)
        last_values = {}
        raw = response.text.strip()
        if not raw:
            print("⚠️  Empty response body from agent", flush=True)
        lines = raw.split('\n')
        print(f"📡 SSE lines received: {len(lines)}", flush=True)
        for i, line in enumerate(lines):
            if not line.startswith('data: '):
                continue
            chunk = line[6:]
            try:
                data = json.loads(chunk)
            except Exception as e:
                print(f"⚠️  Failed to parse SSE line #{i}: {e} | chunk={chunk[:120]}...", flush=True)
                continue
            extracted = self._extract_values_from_sse_obj(data)
            if isinstance(extracted, dict) and extracted:
                last_values.update(extracted)
                print(f"🔧 Values update keys: {list(extracted.keys())}", flush=True)
            else:
                # Log shape for debugging
                try:
                    keys = list(data.keys())
                    inner_keys = list(data.get('data', {}).keys()) if isinstance(data.get('data'), dict) else None
                    print(f"ℹ️  SSE obj without values: keys={keys} data.keys={inner_keys}", flush=True)
                except Exception:
                    pass
        
        print(f"✅ Final values keys: {list(last_values.keys())}", flush=True)
        return {"values": last_values}
    
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
        if not isinstance(result, dict):
            print(f"⚠️  Unexpected result type from invoke_agent: {type(result)}", flush=True)
            return
        
        # Process state values only (no messages/tool parsing)
        values = result.get("values", {})
        print(f"🧩 Values keys from agent: {list(values.keys())}", flush=True)
        eval_suite_data = values.get("eval_suite")
        parsed_agent_url = values.get("agent_url", target_url)
        parsed_agent_id = values.get("agent_name", target_id)

        # Normalize to plain Python types
        eval_suite_plain = self._to_plain(eval_suite_data) if eval_suite_data else None

        if eval_suite_plain:
            if not isinstance(eval_suite_plain, dict):
                print(f"⚠️  eval_suite is not a dict after normalization: {type(eval_suite_plain)}", flush=True)
            print(f"📋 Generated eval suite: {eval_suite_plain.get('id')}", flush=True)
            # Persist in thread context for post-confirmation execution
            try:
                if thread_id in self.active_threads:
                    self.active_threads[thread_id]["eval_suite"] = eval_suite_plain
                    self.active_threads[thread_id]["target_url"] = parsed_agent_url
                    self.active_threads[thread_id]["target_id"] = parsed_agent_id
            except Exception:
                pass

            # Store eval suite in Event Bus and locally before requesting confirmation
            local_file_path = None
            evals_markdown = None

            print("🛰️  Storing eval suite in Event Bus...", flush=True)
            await self._store_eval_suite(
                parsed_agent_url,
                parsed_agent_id,
                eval_suite_plain
            )
            # Store eval_suite_id in context for later reference
            self.active_threads[thread_id]["eval_suite_id"] = eval_suite_plain.get("id")
            # Save locally for convenience/auditing
            try:
                local_file_path = self._save_eval_suite_local(eval_suite_plain)
                print(f"📝 Saved eval suite locally: {local_file_path}", flush=True)
                # Keep path in context for fallback loading
                self.active_threads[thread_id]["local_file_path"] = local_file_path
            except Exception as e:
                print(f"⚠️  Failed to save eval suite locally: {e}", flush=True)
                traceback.print_exc()

            # Build a human-readable list of all evals (full display)
            test_cases = eval_suite_plain.get("test_cases", [])
            print(f"🧪 Test cases generated: {len(test_cases)}", flush=True)
            lines = []
            for idx_i, tc in enumerate(test_cases, 1):
                lines.append(f"{idx_i}. [{tc.get('id','')}] Expectation: {tc.get('expectation_ref','')}")
                lines.append(f"   - Input: {tc.get('input_message','')}")
                lines.append(f"   - Expected: {tc.get('expected_behavior','')}")
                lines.append(f"   - Criteria: {tc.get('success_criteria','')}")
            evals_markdown = "\n".join(lines)

            # Publish UserConfirmationQuery (include full evals in question/context)
            test_count = len(test_cases)
            default_question = "Should I proceed to run these tests now?"
            question_text = (
                f"I've generated {test_count} test cases.\n\n"
                f"Here are the generated evals:\n{evals_markdown}\n\n"
                f"{default_question}"
            )

            await self.event_bus.publish(EventMessage(
                event_type=EventType.USER_CONFIRMATION_QUERY,
                sender=self.agent_name,
                thread_id=thread_id,
                payload={
                    "question": question_text,
                    "context": {
                        "test_count": test_count,
                        "eval_suite_id": eval_suite_plain.get("id"),
                        "test_cases": test_cases,
                        "local_file_path": local_file_path
                    }
                }
            ))
            print(f"✉️  Published UserConfirmationQuery (test_count={test_count})", flush=True)
        else:
            print("⚠️  No eval_suite found in values; cannot request confirmation", flush=True)
    
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

    def _save_detailed_results_local(self, thread_id: str, eval_suite_id: str, results: list) -> str:
        """Save per-test detailed results to a local JSON file and return absolute path"""
        base_dir = Path(__file__).parent / "generated_evals"
        base_dir.mkdir(parents=True, exist_ok=True)
        suffix = eval_suite_id or thread_id
        file_name = f"results_{suffix}.json"
        file_path = base_dir / file_name
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({
                "thread_id": thread_id,
                "eval_suite_id": eval_suite_id,
                "results": results
            }, f, indent=2)
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
        print(f"📦 Context present: eval_suite={'yes' if eval_suite else 'no'} url={target_url} id={target_id}", flush=True)
        if not eval_suite:
            # Fallback: load from local file path saved earlier
            local_path = ctx.get("local_file_path")
            try:
                if local_path and os.path.exists(local_path):
                    with open(local_path, "r", encoding="utf-8") as f:
                        eval_suite = json.load(f)
            except Exception as e:
                print(f"⚠️  Failed to load eval suite from local file: {e}", flush=True)
                traceback.print_exc()

        # If we have an eval_suite, run each test by delegating to the judge agent
        if not eval_suite:
            print("⚠️  No eval_suite available; cannot run tests", flush=True)
            return

        # Prepare results output file
        eval_suite_id = ctx.get("eval_suite_id") or eval_suite.get("id")
        results_path = self._save_detailed_results_local(thread_id, eval_suite_id, [])
        print(f"🗂️  Results will be written to: {results_path}", flush=True)

        # Iterate test cases and invoke judge agent for each
        test_cases = eval_suite.get("test_cases", [])
        judge_url = self.langgraph_url
        passed = 0
        failed = 0
        total = len(test_cases)

        for idx, tc in enumerate(test_cases, 1):
            print(f"⚖️  Judging test {idx}/{total}: {tc.get('id')}", flush=True)
            judge_payload = {
                "assistant_id": "judge_agent",
                "input": {
                    "messages": [{
                        "role": "user",
                        "content": json.dumps({
                            "target_url": target_url,
                            "target_agent_id": target_id,
                            "test_case": tc,
                            "results_file": results_path
                        })
                    }]
                },
                "stream_mode": ["values"],
                "config": {"configurable": {"thread_id": str(uuid.UUID(hashlib.md5((thread_id + tc.get('id','')).encode()).hexdigest()))}}
            }

            try:
                resp = await self.http_client.post(f"{judge_url}/runs/stream", json=judge_payload)
                resp.raise_for_status()
                # Inspect final values to tally pass/fail if surfaced
                final_vals = {}
                for line in resp.text.strip().split('\n'):
                    if not line.startswith('data: '):
                        continue
                    try:
                        obj = json.loads(line[6:])
                        vals = obj.get('values', {}) if isinstance(obj, dict) else {}
                        if isinstance(vals, dict):
                            final_vals.update(vals)
                    except Exception:
                        continue
                # Prefer judge state's verdict if present; otherwise, we'll recount from the results file later
                if isinstance(final_vals.get('verdict_passed'), bool):
                    if final_vals.get('verdict_passed'):
                        passed += 1
                    else:
                        failed += 1
            except Exception as e:
                print(f"⚠️  Judge agent call failed for test {tc.get('id')}: {e}", flush=True)
                failed += 1

        # If verdicts weren't all surfaced, derive tallies from the written file
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results = data.get('results', [])
                if results:
                    passed = sum(1 for r in results if r.get('passed'))
                    failed = sum(1 for r in results if not r.get('passed'))
                    total = len(results)
        except Exception as e:
            print(f"⚠️  Failed to read results file for tally: {e}", flush=True)

        score = (passed / total * 100) if total > 0 else 0
        summary = f"{passed}/{total} passed ({score:.0f}%)"
        print(f"🏁 Tally -> passed={passed} failed={failed} total={total} score={score:.2f}", flush=True)

        # Build concise preview for user
        preview_lines = []
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results = data.get('results', [])
                for i, r in enumerate(results[:5], 1):
                    status = "✅" if r.get("passed") else "❌"
                    snippet = (r.get("input_sent") or "")[:60]
                    preview_lines.append(f"{i}. {status} {snippet}")
                if len(results) > 5:
                    preview_lines.append(f"(+{len(results)-5} more) See: {results_path}")
        except Exception:
            pass
        preview = "\n".join(preview_lines)

        details_message = (
            f"📊 Evaluation Complete!\n\n**Results:** {summary}\n**Score:** {score:.0f}%\n\n"
            f"Details preview:\n{preview}\n"
        )
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
                "eval_suite_id": eval_suite_id,
                "total_tests": total,
                "details": {
                    "results_file": results_path,
                }
            }
        ))
        print(f"✉️  Published TestResultsReady (CS will relay to user)", flush=True)
        
        # Clean up
        if thread_id in self.active_threads:
            del self.active_threads[thread_id]
    
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

