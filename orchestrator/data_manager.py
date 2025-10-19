"""Data management module for Orchestrator"""

import uuid
from typing import List, Dict, Any
from datetime import datetime

from shared.database import get_db


class DataManager:
    """Manages all database operations for the Orchestrator"""

    def __init__(self):
        self.db = get_db()

    def register_agent(self, agent_name: str, port: int, assistant_id: str, capabilities: List[str]):
        """Register an agent in the database"""
        self.db.register_subscriber(
            agent_name=agent_name,
            filters={"capabilities": capabilities, "port": port, "assistant_id": assistant_id}
        )
        return {"status": "registered", "agent_name": agent_name}

    def store_eval_suite(self, suite_data: Dict[str, Any]) -> str:
        """Store an evaluation suite"""
        suite_id = suite_data.get("suite_id") or str(uuid.uuid4())

        self.db.save_eval_suite(
            suite_id=suite_id,
            spec_name=suite_data.get("spec_name", ""),
            spec_version=suite_data.get("spec_version", "1.0.0"),
            test_cases=suite_data.get("test_cases", []),
            thread_id=suite_data.get("thread_id"),
            target_agent_url=suite_data.get("target_agent_url"),
            target_agent_id=suite_data.get("target_agent_id"),
            langgraph_thread_id=suite_data.get("langgraph_thread_id")
        )

        return suite_id

    def get_eval_suites(self, agent_url: str = None, agent_id: str = None) -> List[Dict[str, Any]]:
        """Get evaluation suites"""
        return self.db.get_eval_suites(
            target_agent_url=agent_url,
            target_agent_id=agent_id
        )

    def store_test_results(self, suite_id: str, thread_id: str, results: List[Dict[str, Any]]) -> str:
        """Store test results"""
        for result in results:
            result_id = result.get("result_id") or str(uuid.uuid4())
            self.db.save_test_result(
                result_id=result_id,
                suite_id=suite_id,
                thread_id=thread_id,
                test_case_id=result.get("test_case_id"),
                input_sent=result.get("input_sent"),
                actual_output=result.get("actual_output"),
                expected_behavior=result.get("expected_behavior"),
                passed=result.get("passed"),
                score=result.get("score"),
                judge_reasoning=result.get("judge_reasoning")
            )

        return {"status": "stored", "results_count": len(results)}

    def store_conversation_message(self, thread_id: str, sender: str, content: str, role: str = "assistant") -> str:
        """Store a conversation message"""
        message_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        self.db.add_message(
            thread_id=thread_id,
            message_id=message_id,
            timestamp=timestamp,
            role=role,
            sender=sender,
            content=content,
            message_type="conversation"
        )

        return message_id

    def get_conversation_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a thread"""
        return self.db.get_thread_messages(thread_id)

    def get_all_threads(self) -> List[Dict[str, Any]]:
        """Get all conversation threads"""
        return self.db.list_threads()

    def get_test_results(self, suite_id: str = None, thread_id: str = None) -> List[Dict[str, Any]]:
        """Get test results with optional filters"""
        return self.db.get_test_results(suite_id=suite_id, thread_id=thread_id)
