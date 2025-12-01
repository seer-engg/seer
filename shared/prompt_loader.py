import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

# Base directory for prompts
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

@dataclass
class PromptConfig:
    system: str
    user_template: str
    all_prompts: Dict[str, str]

def load_prompt(relative_path: str) -> PromptConfig:
    """
    Load a prompt configuration from a YAML file.
    
    Args:
        relative_path: Path relative to the 'prompts' directory (e.g. 'eval_agent/generator.yaml')
        
    Returns:
        PromptConfig object containing system prompt and user template.
    """
    full_path = PROMPTS_DIR / relative_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {full_path}")
        
    with open(full_path, "r") as f:
        data = yaml.safe_load(f)
        
    return PromptConfig(
        system=data.get("system", "").strip(),
        user_template=data.get("user_template", "").strip(),
        all_prompts=data
    )
