#!/usr/bin/env python3
"""
Diagnostic script to inspect checkpoint contents in LangGraph checkpoints.
Checks channel_values and full checkpoint structure for trace keys.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from seer.api.agents.checkpointer import get_checkpointer  # noqa: E402


async def inspect_checkpoint_blob(thread_id: str):
    """Inspect checkpoint data for a given thread_id."""
    checkpointer = await get_checkpointer()
    if not checkpointer:
        print("ERROR: Checkpointer not available")
        return

    config_dict = {"configurable": {"thread_id": thread_id}}

    print(f"\n=== Inspecting checkpoints for thread_id: {thread_id} ===\n")

    checkpoint_count = 0
    trace_keys_found = []

    async for checkpoint_tuple in checkpointer.alist(config_dict):
        checkpoint_count += 1
        checkpoint_id = checkpoint_tuple.config.get("configurable", {}).get("checkpoint_id", "unknown")
        checkpoint = checkpoint_tuple.checkpoint
        channel_values = checkpoint.get("channel_values", {})

        print(f"\n--- Checkpoint {checkpoint_count}: {checkpoint_id} ---")
        print(f"Channel values keys: {list(channel_values.keys())}")

        # Check for trace keys in channel_values
        trace_keys_in_channel = [k for k in channel_values.keys() if k.startswith("_trace_") or k.startswith("__trace_")]
        if trace_keys_in_channel:
            print(f"✓ Found trace keys in channel_values: {trace_keys_in_channel}")
            trace_keys_found.extend(trace_keys_in_channel)
            for key in trace_keys_in_channel:
                print(f"  {key}: {type(channel_values[key]).__name__}")
        else:
            print("✗ No trace keys in channel_values")

        # Show all keys starting with underscore
        underscore_keys = [k for k in channel_values.keys() if k.startswith("_")]
        if underscore_keys:
            print(f"Keys starting with '_': {underscore_keys}")

        # Inspect full checkpoint structure
        print("\nFull checkpoint structure:")
        print(f"  - channel_values: {len(channel_values)} keys")
        print(f"  - channel_versions: {len(checkpoint.get('channel_versions', {}))} keys")
        print(f"  - versions_seen: {len(checkpoint.get('versions_seen', {}))} keys")

        # Check metadata
        metadata = checkpoint_tuple.metadata if hasattr(checkpoint_tuple, 'metadata') else {}
        if metadata:
            print(f"  - metadata keys: {list(metadata.keys())}")
            if 'writes' in metadata and metadata['writes']:
                print(f"  - writes in metadata: {list(metadata['writes'].keys())}")

        # Check pending_writes
        pending_writes = checkpoint_tuple.pending_writes if hasattr(checkpoint_tuple, 'pending_writes') else []
        if pending_writes:
            print(f"  - pending_writes: {len(pending_writes)} items")
            for pw in pending_writes[:3]:  # Show first 3
                print(f"    {pw}")

    print("\n=== Summary ===")
    print(f"Total checkpoints inspected: {checkpoint_count}")
    if trace_keys_found:
        print(f"✓ Trace keys found: {set(trace_keys_found)}")
    else:
        print("✗ No trace keys found in any checkpoint")
        print("\nThis suggests trace keys are either:")
        print("  1. Not being written to node outputs")
        print("  2. Being filtered out by LangGraph before checkpoint save")
        print("  3. Stored in __root__ blob but not exposed in channel_values")


async def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint_blob.py <thread_id>")
        print("Example: python inspect_checkpoint_blob.py run_7")
        sys.exit(1)

    thread_id = sys.argv[1]
    await inspect_checkpoint_blob(thread_id)


if __name__ == "__main__":
    asyncio.run(main())
