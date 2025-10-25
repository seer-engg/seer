

from langchain.tools import tool


@tool
def think(thought: str) -> str:
    """Use the tool to think about something.
        This is perfect to start your workflow.
        It will not obtain new information or take any actions, but just append the thought to the log and return the result.
        Use it when complex reasoning or some cache memory or a scratchpad is needed.
    
    Args:
        thought: Intermediate reasoning to log
        
    Returns:
        Acknowledgment that the thought was recorded
    """
    return thought