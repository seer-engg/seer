"""
State access utilities for eval agent Supervisor pattern.

Since we're storing complex state directly in the TypedDict (as dicts),
these utilities help convert between Pydantic models and dicts for state access.
"""
from typing import Any, Optional, Dict, List
from langchain.tools import ToolRuntime
from shared.schema import (
    AgentContext,
    DatasetExample,
    AlignmentState,
    AgentSpec,
    CodexOutput,
    DatasetContext,
    ExperimentContext,
    ExperimentResultContext,
    UserIntent,
)
from agents.eval_agent.models import Hypothesis
from shared.logger import get_logger

logger = get_logger("eval_agent.state_store")


def _model_to_dict(model: Any) -> Dict[str, Any]:
    """Convert Pydantic model to dict."""
    if model is None:
        return None
    if isinstance(model, dict):
        return model
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return model


def _dict_to_model(dict_value: Any, model_class: type) -> Any:
    """Convert dict to Pydantic model."""
    if dict_value is None:
        return None
    if isinstance(dict_value, model_class):
        return dict_value
    if isinstance(dict_value, dict):
        return model_class(**dict_value)
    return dict_value


# Convenience functions for accessing state via runtime.state
# Note: Tools receive state updates via Command pattern or return dict updates

def get_agent_context_from_state(state: Dict[str, Any]) -> Optional[AgentContext]:
    """Get AgentContext from state dict."""
    value = state.get("agent_context")
    if value is None:
        return None
    return _dict_to_model(value, AgentContext)


def set_agent_context_in_state(state: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
    """Set AgentContext in state dict. Returns update dict."""
    return {"agent_context": _model_to_dict(context)}


def get_dataset_examples_from_state(state: Dict[str, Any]) -> List[DatasetExample]:
    """Get dataset_examples from state dict."""
    value = state.get("dataset_examples", [])
    if not value:
        return []
    if isinstance(value, list):
        return [_dict_to_model(ex, DatasetExample) if isinstance(ex, dict) else ex for ex in value]
    return value if isinstance(value, list) else []


def set_dataset_examples_in_state(state: Dict[str, Any], examples: List[DatasetExample]) -> Dict[str, Any]:
    """Set dataset_examples in state dict. Returns update dict."""
    return {"dataset_examples": [_model_to_dict(ex) for ex in examples]}


def get_alignment_state_from_state(state: Dict[str, Any]) -> Optional[AlignmentState]:
    """Get alignment_state from state dict."""
    value = state.get("alignment_state")
    if value is None:
        return None
    return _dict_to_model(value, AlignmentState)


def set_alignment_state_in_state(state: Dict[str, Any], alignment_state: AlignmentState) -> Dict[str, Any]:
    """Set alignment_state in state dict. Returns update dict."""
    return {"alignment_state": _model_to_dict(alignment_state)}


def get_agent_spec_from_state(state: Dict[str, Any]) -> Optional[AgentSpec]:
    """Get agent_spec from state dict."""
    value = state.get("agent_spec")
    if value is None:
        return None
    return _dict_to_model(value, AgentSpec)


def set_agent_spec_in_state(state: Dict[str, Any], spec: AgentSpec) -> Dict[str, Any]:
    """Set agent_spec in state dict. Returns update dict."""
    return {"agent_spec": _model_to_dict(spec)}


def get_user_intent_from_state(state: Dict[str, Any]) -> Optional[UserIntent]:
    """Get user_intent from state dict."""
    value = state.get("user_intent")
    if value is None:
        return None
    return _dict_to_model(value, UserIntent)


def set_user_intent_in_state(state: Dict[str, Any], intent: UserIntent) -> Dict[str, Any]:
    """Set user_intent in state dict. Returns update dict."""
    return {"user_intent": _model_to_dict(intent)}

