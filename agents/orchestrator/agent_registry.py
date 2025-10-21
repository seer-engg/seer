"""Agent registration module for Orchestrator (agent-local)"""

from typing import Dict, Any, List
from datetime import datetime

from seer.shared.error_handling import create_error_response, create_success_response


class AgentRegistry:
    """Manages agent registration and status"""

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.registered_agents: Dict[str, Dict[str, Any]] = {}

    def register_agent(self, agent_name: str, port: int, assistant_id: str, capabilities: List[str]) -> str:
        """Register a new agent"""
        try:
            # Store in database
            result = self.data_manager.register_agent(agent_name, port, assistant_id, capabilities)

            # Update in-memory registry
            self.registered_agents[agent_name] = {
                "port": port,
                "assistant_id": assistant_id,
                "capabilities": capabilities,
                "registered_at": datetime.now().isoformat()
            }

            return create_success_response(result)
        except Exception as e:
            return create_error_response(f"Failed to register agent: {str(e)}", e)

    def get_registered_agents(self) -> str:
        """Get all registered agents"""
        try:
            return create_success_response({
                "agents": self.registered_agents,
                "count": len(self.registered_agents)
            })
        except Exception as e:
            return create_error_response(f"Failed to get registered agents: {str(e)}", e)

    def get_agent_status(self) -> str:
        """Get status of all agents"""
        try:
            subscribers = self.data_manager.db.get_subscribers()

            agent_status = {}
            for sub in subscribers:
                agent_name = sub["agent_name"]
                registered_info = self.registered_agents.get(agent_name, {})

                agent_status[agent_name] = {
                    "registered": agent_name in self.registered_agents,
                    "last_poll": sub.get("last_poll"),
                    "message_count": sub.get("message_count", 0),
                    "publish_count": sub.get("publish_count", 0),
                    "capabilities": registered_info.get("capabilities", []),
                    "port": registered_info.get("port"),
                    "assistant_id": registered_info.get("assistant_id")
                }

            # Add registered agents from memory
            for agent_name, agent_info in self.registered_agents.items():
                if agent_name not in agent_status:
                    agent_status[agent_name] = {
                        "registered": True,
                        "capabilities": agent_info.get("capabilities", []),
                        "port": agent_info.get("port"),
                        "assistant_id": agent_info.get("assistant_id")
                    }

            return create_success_response({
                "agents": agent_status,
                "total_agents": len(agent_status)
            })
        except Exception as e:
            return create_error_response(f"Failed to get agent status: {str(e)}", e)


