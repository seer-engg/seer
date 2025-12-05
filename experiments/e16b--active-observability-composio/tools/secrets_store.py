"""
Secrets store - loads specific secrets from .env file.
Simple dictionary-based store, similar to runtime_tool_store.
"""
from typing import Dict, Optional
from dotenv import dotenv_values
from pathlib import Path

class SecretsStore:
    """Simple store for secrets loaded from .env file."""
    
    def __init__(self):
        self._secrets: Dict[str, str] = {}
        self._load_secrets()
    
    def _load_secrets(self):
        """Load specific secrets from .env file."""
        # Find .env file (go up to seer root)
        project_root = Path(__file__).parents[2]  # Go up from experiments/e16b/.../tools/
        env_file = project_root / ".env"
        
        if not env_file.exists():
            # Fallback: try current directory
            env_file = Path(".env")
        
        if env_file.exists():
            # Read only from .env file
            env_vars = dotenv_values(env_file)
            # Load only the secrets we care about
            secrets_to_load = [
                "ASANA_WORKSPACE_ID",
                "ASANA_PROJECT_ID",
            ]
            
            for key in secrets_to_load:
                value = env_vars.get(key)
                if value:
                    self._secrets[key] = value
    
    def get(self, key: str) -> Optional[str]:
        """Get a secret by key."""
        return self._secrets.get(key)
    
    def get_all(self) -> Dict[str, str]:
        """Get all secrets."""
        return self._secrets.copy()
    
    def format_for_prompt(self) -> str:
        """Format secrets for inclusion in prompts."""
        if not self._secrets:
            return ""
        
        lines = []
        for key, value in sorted(self._secrets.items()):
            lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)

# Global instance
_secrets_store = SecretsStore()

