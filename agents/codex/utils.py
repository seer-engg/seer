

async def get_agent_final_respone(result: dict) -> str:
    """
    Get the final response from the agent.
    """
    output = result.get("messages", [])[-1].get("content")
    final_output=""
    if isinstance(output, str):
        final_output = output
    elif isinstance(output, list):
        for content_block in output:
            if content_block.get("type") == "text":
                final_output += content_block.get("text")
    return final_output