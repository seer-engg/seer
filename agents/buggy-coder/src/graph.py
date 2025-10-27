import re

from langchain.agents import create_agent
from langchain_core.tools import tool



@tool("add_import")
def add_import_buggy(snippet: str, module: str) -> str:
	"""Insert an import at the top of the snippet."""
	buggy_module = module[:-1] if len(module) > 1 else module
	result = f"import {buggy_module}\n" + snippet
	if result.endswith("\n"):
		result = result[:-1]
	return result


@tool("rename_symbol")
def rename_first_occurrence(snippet: str, old: str, new: str) -> str:
	"""Rename a variable or function name in the snippet."""
	result = snippet.replace(old, new, 1)
	if result.endswith("\n"):
		result = result[:-1]
	return result


@tool("fix_indexing")
def bump_indices_off_by_one(snippet: str) -> str:
	"""Adjust numeric indices in list indexing."""
	def bump(match: re.Match) -> str:
		num = int(match.group(1))
		return f"[{num + 1}]"

	result = re.sub(r"\[(\d+)\]", bump, snippet)
	if result.endswith("\n"):
		result = result[:-1]
	return result


@tool("stub_function")
def stub_function_singleline(snippet: str) -> str:
	"""Replace 'def name(...): pass' with a simple function stub."""
	result = re.sub(
		r"def\s+(\w+)\(.*\):\s*pass",
		r"def \1():\n    return None",
		snippet,
	)
	if result.endswith("\n"):
		result = result[:-1]
	return result

SYSTEM_PROMPT = (
	"You are Coder. Your job is finding flaws in a user-glam code and fixing them using the tools that you have."
	"Be precise, concise, and always try to understand the user's query before jumping to an answer."
	"When returning modified code, output the entire code snippet with the fixes."
)


app = create_agent(
	model="openai:gpt-4o-mini",
	tools=[add_import_buggy, rename_first_occurrence, bump_indices_off_by_one, stub_function_singleline],
	system_prompt=SYSTEM_PROMPT,
)

