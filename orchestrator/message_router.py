"""Message routing module for Orchestrator"""

import json
from typing import Dict, Any, List
from datetime import datetime

from shared.a2a_utils import send_a2a_message
from shared.error_handling import create_error_response, create_success_response


class MessageRouter:
    """Handles message routing between agents"""

    def __init__(self, data_manager, registered_agents: Dict[str, Dict[str, Any]]):
        self.data_manager = data_manager
        self.registered_agents = registered_agents

    def update_registered_agents(self, agents: Dict[str, Dict[str, Any]]):
        """Update the registered agents dictionary"""
        self.registered_agents = agents

    async def route_user_confirmation(self, confirmation_data: str) -> str:
        """Route user confirmation to eval agent"""
        try:
            data = json.loads(confirmation_data)
            confirmed = data.get("confirmed")
            details = data.get("details")
            thread_id = data.get("thread_id")

            if not all([confirmed is not None, thread_id]):
                return create_error_response("Missing required fields: confirmed, thread_id")

            # Find eval agent
            eval_agent_info = self._find_agent_by_capability("eval")
            if not eval_agent_info:
                return create_error_response("Eval agent not found")

            # Send confirmation to eval agent
            confirmation_message = f"User {'confirmed' if confirmed else 'declined'}: {details}"
            result = await send_a2a_message(
                eval_agent_info["assistant_id"],
                eval_agent_info["port"],
                confirmation_message,
                thread_id
            )

            # Store confirmation in conversation history
            self.data_manager.store_conversation_message(
                thread_id=thread_id,
                sender="user",
                content=confirmation_message,
                role="user"
            )

            return create_success_response({
                "message": f"Confirmation sent to eval agent: {confirmation_message}",
                "result": result
            })
        except Exception as e:
            return create_error_response(f"Failed to handle user confirmation: {str(e)}", e)

    async def route_eval_question(self, question_data: str) -> str:
        """Route eval question to customer success agent"""
        try:
            data = json.loads(question_data)
            question = data.get("question")
            thread_id = data.get("thread_id")

            if not all([question, thread_id]):
                return create_error_response("Missing required fields: question, thread_id")

            # Find customer success agent
            cs_agent_info = self._find_agent_by_capability("user_interaction")
            if not cs_agent_info:
                return create_error_response("Customer Success agent not found")

            # Send question to customer success agent
            question_message = f"Eval Agent Question: {question}"
            result = await send_a2a_message(
                cs_agent_info["assistant_id"],
                cs_agent_info["port"],
                question_message,
                thread_id
            )

            # Store question in conversation history
            self.data_manager.store_conversation_message(
                thread_id=thread_id,
                sender="eval_agent",
                content=question_message,
                role="assistant"
            )

            return create_success_response({
                "message": f"Question forwarded to user: {question}",
                "result": result
            })
        except Exception as e:
            return create_error_response(f"Failed to handle eval question: {str(e)}", e)

    async def route_eval_results(self, results_data: str) -> str:
        """Route eval results to customer success agent"""
        try:
            data = json.loads(results_data)
            results = data.get("results")
            thread_id = data.get("thread_id")

            if not all([results, thread_id]):
                return create_error_response("Missing required fields: results, thread_id")

            # Find customer success agent
            cs_agent_info = self._find_agent_by_capability("user_interaction")
            if not cs_agent_info:
                return create_error_response("Customer Success agent not found")

            # Send results to customer success agent
            results_message = f"Evaluation Results: {results}"
            result = await send_a2a_message(
                cs_agent_info["assistant_id"],
                cs_agent_info["port"],
                results_message,
                thread_id
            )

            # Store results in conversation history
            self.data_manager.store_conversation_message(
                thread_id=thread_id,
                sender="eval_agent",
                content=results_message,
                role="assistant"
            )

            return create_success_response({
                "message": f"Results sent to user: {results}",
                "result": result
            })
        except Exception as e:
            return create_error_response(f"Failed to handle eval results: {str(e)}", e)

    def _find_agent_by_capability(self, capability: str) -> Dict[str, Any]:
        """Find agent by capability"""
        for agent_name, agent_info in self.registered_agents.items():
            capabilities = agent_info.get("capabilities", [])
            if capability in capabilities or any(capability in str(cap).lower() for cap in capabilities):
                return agent_info
        return None
