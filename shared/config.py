"""Configuration management for Seer agents"""

import json
import os
import socket
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator

class SeerConfig(BaseModel):
    """Centralized configuration for Seer system"""
    
    # Agent ports with defaults
    orchestrator_port: int = Field(default=8000, description="Orchestrator agent port")
    eval_agent_port: int = Field(default=8002, description="Eval agent port")
    coding_agent_port: int = Field(default=8003, description="Coding agent port")
    
    # Base URLs
    orchestrator_url: str = Field(default="http://127.0.0.1:8000", description="Orchestrator base URL")
    
    # Timeouts
    a2a_timeout: float = Field(default=120.0, description="A2A communication timeout in seconds")
    test_timeout: float = Field(default=60.0, description="Test execution timeout in seconds")
    
    # API configuration
    openai_api_key: str = Field(description="OpenAI API key")
    
    # Thread ID prefixes
    eval_suite_storage_thread: str = Field(default="eval_suite_storage", description="Eval suite storage thread ID")
    test_results_storage_thread: str = Field(default="test_results_storage", description="Test results storage thread ID")
    
    # UI configuration
    ui_port: int = Field(default=8501, description="Streamlit UI port")
    
    @validator('openai_api_key')
    def validate_openai_key(cls, v):
        if not v or v == "your_key_here":
            raise ValueError("OpenAI API key must be set")
        return v
    
    @validator('orchestrator_url')
    def validate_orchestrator_url(cls, v, values):
        if 'orchestrator_port' in values:
            port = values['orchestrator_port']
            if f":{port}" not in v:
                return f"http://127.0.0.1:{port}"
        return v
    
    def get_agent_url(self, agent_name: str) -> str:
        """Get URL for a specific agent"""
        port = getattr(self, f"{agent_name}_port")
        return f"http://127.0.0.1:{port}"
    
    def get_available_port(self, start_port: int = 8000, max_port: int = 8010) -> int:
        """Find an available port starting from start_port"""
        for port in range(start_port, max_port + 1):
            if self._is_port_available(port):
                return port
        raise RuntimeError(f"No available ports found in range {start_port}-{max_port}")
    
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return True
        except OSError:
            return False
    
    def generate_thread_id(self, prefix: str = "thread") -> str:
        """Generate a unique thread ID with optional prefix"""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

class AgentConfig:
    """Manages agent configuration from deployment-config.json"""
    
    def __init__(self, config_path: str = None):
        """Initialize with config file path"""
        if config_path is None:
            # Default to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "deployment-config.json"
        
        self.config_path = Path(config_path)
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                self._config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent"""
        if not self._config:
            raise RuntimeError("Configuration not loaded")
        
        agents = self._config.get("agents", {})
        if agent_name not in agents:
            raise KeyError(f"Agent '{agent_name}' not found in configuration")
        
        return agents[agent_name]
    
    def get_port(self, agent_name: str) -> int:
        """Get port for an agent"""
        config = self.get_agent_config(agent_name)
        return config["port"]
    
    def get_graph_name(self, agent_name: str) -> str:
        """Get graph name for an agent"""
        config = self.get_agent_config(agent_name)
        return config["graph_name"]
    
    def list_agents(self) -> list:
        """List all configured agents"""
        if not self._config:
            return []
        return list(self._config.get("agents", {}).keys())

# Global config instances
_config_instance: Optional[AgentConfig] = None
_seer_config_instance: Optional[SeerConfig] = None

def get_config() -> AgentConfig:
    """Get global agent configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = AgentConfig()
    return _config_instance

def get_seer_config() -> SeerConfig:
    """Get global Seer configuration instance"""
    global _seer_config_instance
    if _seer_config_instance is None:
        # Load from environment variables with defaults
        config_data = {
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "orchestrator_port": int(os.getenv("ORCHESTRATOR_PORT", "8000")),
            "eval_agent_port": int(os.getenv("EVAL_AGENT_PORT", "8002")),
            "coding_agent_port": int(os.getenv("CODING_AGENT_PORT", "8003")),
            "ui_port": int(os.getenv("UI_PORT", "8501")),
            "a2a_timeout": float(os.getenv("A2A_TIMEOUT", "120.0")),
            "test_timeout": float(os.getenv("TEST_TIMEOUT", "60.0")),
        }
        _seer_config_instance = SeerConfig(**config_data)
    return _seer_config_instance

def get_port(agent_name: str) -> int:
    """Convenience function to get port"""
    return get_config().get_port(agent_name)

def get_graph_name(agent_name: str) -> str:
    """Convenience function to get graph name"""
    return get_config().get_graph_name(agent_name)
