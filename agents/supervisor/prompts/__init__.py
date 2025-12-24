import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(CURRENT_DIR, "PROMPT_GENERIC_WORKER.md"), "r", encoding="utf-8") as f:
    PROMPT_GENERIC_WORKER = f.read()

__all__ = ["PROMPT_GENERIC_WORKER"]