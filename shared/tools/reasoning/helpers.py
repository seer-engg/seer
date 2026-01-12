"""
Helper functions for the RLM sandbox environment.

Provides safe utility functions for examining, searching, chunking, and
recursively processing data with LLM calls.
"""

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from shared.logger import get_logger

logger = get_logger("shared.tools.reasoning.helpers")


class RLMHelpers:
    """Helper functions available in the RLM sandbox."""

    def __init__(self, context: Dict[str, Any], model: str, max_depth: int):
        self.context = context
        self.model = model
        self.max_depth = max_depth
        self.current_depth = 0
        self.execution_log: List[Dict[str, Any]] = []
        self.total_llm_calls = 0

    def examine(self, data: Any, max_items: int = 5) -> Dict[str, Any]:
        """
        Inspect data structure without printing full content.

        Args:
            data: Data to examine
            max_items: Maximum number of sample items to include

        Returns:
            Dictionary with type, size, keys/indices, and sample data
        """
        result = {
            'type': type(data).__name__,
            'size': None,
            'sample': None,
        }

        if isinstance(data, dict):
            result['size'] = len(data)
            result['keys'] = list(data.keys())[:max_items]
            if len(data) > max_items:
                result['keys_truncated'] = True
            sample_keys = list(data.keys())[:max_items]
            result['sample'] = {k: self._truncate_value(data[k]) for k in sample_keys}

        elif isinstance(data, (list, tuple)):
            result['size'] = len(data)
            result['sample'] = [self._truncate_value(item) for item in data[:max_items]]
            if len(data) > max_items:
                result['sample_truncated'] = True

        elif isinstance(data, str):
            result['size'] = len(data)
            result['sample'] = data[:200] + ('...' if len(data) > 200 else '')

        elif isinstance(data, (int, float, bool)) or data is None:
            result['value'] = data

        return result

    def _truncate_value(self, value: Any, max_len: int = 100) -> Any:
        """Truncate value for display purposes."""
        if isinstance(value, str) and len(value) > max_len:
            return value[:max_len] + '...'
        if isinstance(value, (list, tuple)) and len(value) > 3:
            return [self._truncate_value(v, 50) for v in value[:3]] + ['...']
        if isinstance(value, dict) and len(value) > 3:
            sample_keys = list(value.keys())[:3]
            return {k: self._truncate_value(value[k], 50) for k in sample_keys}
        return value

    def _search_dict(
        self,
        data: Dict[str, Any],
        pattern: Optional[str],
        key: Optional[str],
        value: Optional[Any]
    ) -> List[Any]:
        """Search in dictionary by key, value, or pattern."""
        results = []
        if key is not None:
            for k, v in data.items():
                if key.lower() in str(k).lower():
                    results.append({k: v})
        elif value is not None:
            for k, v in data.items():
                if v == value:
                    results.append({k: v})
        elif pattern is not None:
            regex = re.compile(pattern, re.IGNORECASE)
            for k, v in data.items():
                if isinstance(v, str) and regex.search(v):
                    results.append({k: v})
        return results

    def _search_list(
        self,
        data: List[Any],
        pattern: Optional[str],
        value: Optional[Any]
    ) -> List[Any]:
        """Search in list by pattern or value."""
        if pattern is not None:
            return self._search_list_by_pattern(data, pattern)
        if value is not None:
            return [item for item in data if item == value]
        return []

    def _search_list_by_pattern(self, data: List[Any], pattern: str) -> List[Any]:
        """Search list items by regex pattern."""
        regex = re.compile(pattern, re.IGNORECASE)
        results = []
        for item in data:
            if isinstance(item, str) and regex.search(item):
                results.append(item)
            elif isinstance(item, dict) and self._dict_matches_pattern(item, regex):
                results.append(item)
        return results

    def _dict_matches_pattern(self, item: Dict[str, Any], regex: re.Pattern) -> bool:
        """Check if any string value in dict matches pattern."""
        for v in item.values():
            if isinstance(v, str) and regex.search(v):
                return True
        return False

    def search(
        self,
        data: Any,
        pattern: Optional[str] = None,
        key: Optional[str] = None,
        value: Optional[Any] = None
    ) -> List[Any]:
        """
        Search/filter data by pattern, key, or value.

        Args:
            data: Data to search
            pattern: Regex pattern for string matching
            key: Key substring to match in dict keys
            value: Value to match exactly

        Returns:
            List of matching items
        """
        if isinstance(data, dict):
            return self._search_dict(data, pattern, key, value)
        if isinstance(data, list):
            return self._search_list(data, pattern, value)
        return []

    def chunk(
        self,
        data: Any,
        size: int = 1000,
        overlap: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Split data into manageable chunks with optional overlap.

        Args:
            data: Data to chunk (string, list, or dict)
            size: Chunk size
            overlap: Overlap between chunks (default: 0)

        Returns:
            List of chunk dictionaries with 'text', 'start', 'end', 'index' keys
        """
        chunks = []

        if isinstance(data, str):
            step = max(1, size - overlap)
            for i in range(0, len(data), step):
                chunk_text = data[i:i + size]
                chunks.append({
                    'text': chunk_text,
                    'start': i,
                    'end': min(i + size, len(data)),
                    'index': len(chunks),
                    'size': len(chunk_text)
                })
                if i + size >= len(data):
                    break

        elif isinstance(data, list):
            step = max(1, size - overlap)
            for i in range(0, len(data), step):
                chunk_items = data[i:i + size]
                chunks.append({
                    'text': chunk_items,
                    'start': i,
                    'end': min(i + size, len(data)),
                    'index': len(chunks),
                    'size': len(chunk_items)
                })
                if i + size >= len(data):
                    break

        elif isinstance(data, dict):
            items = list(data.items())
            step = max(1, size - overlap)
            for i in range(0, len(items), step):
                chunk_items = dict(items[i:i + size])
                chunks.append({
                    'text': chunk_items,
                    'start': i,
                    'end': min(i + size, len(items)),
                    'index': len(chunks),
                    'size': len(chunk_items)
                })
                if i + size >= len(items):
                    break

        else:
            chunks.append({
                'text': data,
                'start': 0,
                'end': 1,
                'index': 0,
                'size': 1
            })

        return chunks

    async def sub_llm(
        self,
        prompt: str,
        context_chunk: Optional[Any] = None,
        system: Optional[str] = None
    ) -> str:
        """
        Make recursive LLM call with depth tracking.

        Args:
            prompt: Prompt for the LLM
            context_chunk: Optional context to include
            system: Optional system message

        Returns:
            LLM response as string

        Raises:
            RecursionError: If max depth exceeded
        """
        if self.current_depth >= self.max_depth:
            raise RecursionError(
                f"Maximum recursion depth {self.max_depth} exceeded. "
                f"Consider increasing max_depth or simplifying the task."
            )

        # Import here to avoid circular dependency
        from shared.llm import get_llm_without_responses_api  # pylint: disable=import-outside-toplevel # Reason: Avoid circular import at module level
        from langchain_core.messages import HumanMessage, SystemMessage  # pylint: disable=import-outside-toplevel # Reason: Lazy import for performance

        messages = []
        if system:
            messages.append(SystemMessage(content=system))

        if context_chunk is not None:
            if isinstance(context_chunk, str):
                context_str = context_chunk
            else:
                try:
                    context_str = json.dumps(context_chunk, indent=2, default=str)
                except (TypeError, ValueError):
                    context_str = str(context_chunk)

            full_prompt = f"Context:\n{context_str}\n\nTask:\n{prompt}"
        else:
            full_prompt = prompt

        messages.append(HumanMessage(content=full_prompt))

        llm = get_llm_without_responses_api(model=self.model, temperature=0)

        self.current_depth += 1
        self.total_llm_calls += 1
        start_time = datetime.now(timezone.utc)

        log_entry = {
            'depth': self.current_depth,
            'operation': 'sub_llm',
            'prompt_preview': prompt[:100] if len(prompt) > 100 else prompt,
            'context_size': len(str(context_chunk)) if context_chunk else 0,
            'timestamp': start_time.isoformat(),
        }

        try:
            logger.debug("RLM sub_llm at depth %d: %s", self.current_depth, prompt[:50])
            response = await llm.ainvoke(messages)
            result = response.content if hasattr(response, 'content') else str(response)

            end_time = datetime.now(timezone.utc)
            duration_ms = (end_time - start_time).total_seconds() * 1000
            log_entry['duration_ms'] = round(duration_ms, 2)
            log_entry['response_preview'] = result[:100] if len(result) > 100 else result
            self.execution_log.append(log_entry)

            return result

        except Exception as e:
            logger.error("RLM sub_llm failed at depth %d: %s", self.current_depth, str(e))
            log_entry['error'] = str(e)
            self.execution_log.append(log_entry)
            raise

        finally:
            self.current_depth -= 1

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'total_llm_calls': self.total_llm_calls,
            'max_depth_reached': max((log['depth'] for log in self.execution_log), default=0),
            'execution_log_size': len(self.execution_log),
        }
