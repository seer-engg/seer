"""
Scratch data storage for large tool outputs.

The scratch system allows tools to store large datasets (e.g., 1000+ rows from Oura)
outside the LLM context window, while returning a lightweight reference that the
agent can use to load data into sandboxes.

Flow:
1. Tool stores data: `ref = await scratch_store.store(...)` -> ScratchDataRef
2. Tool returns ref (serialized as dict) - only ~200 bytes in context
3. Agent calls codebox_load_data with the handle
4. Data is uploaded directly to sandbox filesystem, bypassing context
"""
from seer.core.scratch.models import SCRATCH_DATA_REF_TYPE, ScratchDataRef, is_scratch_ref, parse_scratch_ref
from seer.core.scratch.store import ScratchStore

__all__ = [
    "SCRATCH_DATA_REF_TYPE",
    "ScratchDataRef",
    "ScratchStore",
    "is_scratch_ref",
    "parse_scratch_ref",
]
