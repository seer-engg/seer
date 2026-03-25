"""
Oura Ring tools — health data access for sleep, heart rate, readiness, activity, SpO2, stress, workouts, temperature.
"""

from seer.tools.base import register_tool
from seer.tools.oura.health import (
    OuraGetActivityTool,
    OuraGetHeartRateTool,
    OuraGetReadinessTool,
    OuraGetSleepTool,
    OuraGetSpO2Tool,
    OuraGetStressTool,
    OuraGetTemperatureTool,
    OuraGetWorkoutsTool,
)


def register_oura_tools():
    """Register all Oura Ring tools with the tool registry."""
    register_tool(OuraGetHeartRateTool())
    register_tool(OuraGetSleepTool())
    register_tool(OuraGetReadinessTool())
    register_tool(OuraGetActivityTool())
    register_tool(OuraGetSpO2Tool())
    register_tool(OuraGetStressTool())
    register_tool(OuraGetWorkoutsTool())
    register_tool(OuraGetTemperatureTool())


__all__ = [
    "register_oura_tools",
    "OuraGetHeartRateTool",
    "OuraGetSleepTool",
    "OuraGetReadinessTool",
    "OuraGetActivityTool",
    "OuraGetSpO2Tool",
    "OuraGetStressTool",
    "OuraGetWorkoutsTool",
    "OuraGetTemperatureTool",
]
