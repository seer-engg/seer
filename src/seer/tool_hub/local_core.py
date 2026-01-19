"""
Local ToolHub implementation using Chroma for vector storage.

Replaces Pinecone with local Chroma vector store for open-source deployment.
"""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI
from tqdm import tqdm

from seer.logger import get_logger

from .models import EnrichedTool, Tool, ToolFunction

logger = get_logger("shared.tool_hub.local_core")


class LocalToolHub:
    """
    Local ToolHub using Chroma for vector storage.

    Stores tool embeddings locally in Docker, eliminating need for Pinecone credentials.
    """

    def __init__(
        self,
        openai_api_key: str,
        persist_directory: str = "./data/tool_index",
        llm_model: str = "gpt-5-mini",
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: Optional[int] = None,
    ):
        """
        Initialize the LocalToolHub with Chroma.

        Args:
            openai_api_key: The OpenAI API key (required).
            persist_directory: Directory to store Chroma database (default: ./data/tool_index).
            llm_model: The model used for enrichment (default: gpt-5-mini).
            embedding_model: The model used for vector embedding (default: text-embedding-3-small).
            embedding_dimensions: Optional embedding dimensions (default: None, uses model default).
        """
        if not openai_api_key:
            raise ValueError("openai_api_key is required")

        self.async_client = AsyncOpenAI(api_key=openai_api_key)
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions

        # Ensure persist directory exists
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize embeddings
        embedding_kwargs = {}
        if embedding_dimensions:
            embedding_kwargs["dimensions"] = embedding_dimensions

        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=openai_api_key,
            **embedding_kwargs
        )

        # Initialize Chroma - will be loaded when needed
        self._vector_store: Optional[Chroma] = None
        self._collection_name = "tools"

        logger.info("LocalToolHub initialized with persist_directory=%s", persist_directory)

    def _get_vector_store(self) -> Chroma:
        """Get or create Chroma vector store instance."""
        if self._vector_store is None:
            self._vector_store = Chroma(
                persist_directory=str(self.persist_directory),
                collection_name=self._collection_name,
                embedding_function=self.embeddings,
            )
        return self._vector_store

    def _is_deprecated(self, t: Union[Tool, Dict[str, Any]]) -> bool:
        """Check if tool is deprecated."""
        if isinstance(t, dict):
            description = t.get("description", "") or t.get("function", {}).get("description", "")
            return "deprecated" in description.lower()
        if isinstance(t, Tool):
            return "deprecated" in (t.function.description or "").lower()
        return False

    def _normalize_dict_to_tool(self, t: dict) -> Optional[Tool]:
        """Normalize dict to Tool object."""
        if "function" in t:
            return Tool.from_dict(t)
        if "parameters" in t:
            return Tool(function=ToolFunction(**t))
        try:
            return Tool.from_dict(t)
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Skipping invalid tool structure %s: %s", t.keys(), exc)
            return None

    def _normalize_tools(self, tools: List[Union[Tool, Dict[str, Any]]]) -> List[Tool]:
        """
        Normalize tool inputs to Tool objects.

        Args:
            tools: List of Tool objects or dictionaries matching OpenAI tool schema.

        Returns:
            List of normalized Tool objects.
        """
        normalized_tools = []
        for t in tools:
            if self._is_deprecated(t):
                continue

            if isinstance(t, dict):
                tool = self._normalize_dict_to_tool(t)
                if tool:
                    normalized_tools.append(tool)
            elif isinstance(t, Tool):
                normalized_tools.append(t)
            else:
                raise ValueError(f"Unsupported tool type: {type(t)}")

        return normalized_tools

    async def ingest(
        self,
        tools: List[Union[Tool, Dict[str, Any]]],
        integration_name: str,
        max_workers: int = 10
    ):
        """
        Ingests tools, enriches them with metadata, and stores them in Chroma vector store.

        Args:
            tools: List of Tool objects or dictionaries matching OpenAI tool schema.
            integration_name: Integration name (e.g., "github", "asana") for namespace isolation.
            max_workers: Number of concurrent threads for enrichment (default: 10).
        """
        integration_name = integration_name.lower()

        logger.info("Ingesting %s tools for {integration_name} into Chroma...", len(tools))

        # Normalize inputs
        normalized_tools = self._normalize_tools(tools)

        if not normalized_tools:
            logger.warning("No valid tools to ingest after normalization.")
            return

        # Enrich tools
        enriched_tools = []
        logger.info("Enriching tools with concurrency (max_workers=%s)...", max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tool = {
                executor.submit(self._enrich_tool_metadata, tool): tool
                for tool in normalized_tools
            }

            for future in tqdm(
                as_completed(future_to_tool),
                total=len(normalized_tools),
                desc="Enriching Tools"
            ):
                tool = future_to_tool[future]
                try:
                    enriched = future.result()
                    enriched_tools.append(enriched)
                except (asyncio.TimeoutError, json.JSONDecodeError) as exc:
                    logger.warning("Failed to enrich %s: %s", tool.function.name, exc)
                except Exception:  # pylint: disable=broad-exception-caught # Reason: Tool enrichment is best-effort; log and continue
                    logger.exception("Unexpected enrichment error for %s", tool.function.name)

        # Generate embeddings and store in Chroma
        logger.info(
            "Generating embeddings and storing %s enriched tools in Chroma...",
            len(enriched_tools)
        )

        vector_store = self._get_vector_store()

        # Prepare documents for Chroma
        documents = []

        for enriched_tool in tqdm(enriched_tools, desc="Storing Tools"):
            try:
                # Use embedding_text as the document content
                doc_content = enriched_tool.embedding_text.replace("\n", " ")

                # Prepare metadata (Chroma supports nested dicts,
                # but we'll use strings for lists to be safe)
                metadata = {
                    "integration": integration_name,
                    "description": enriched_tool.description,
                    # Chroma prefers strings for lists
                    "use_cases": json.dumps(enriched_tool.use_cases),
                    "likely_neighbors": json.dumps(enriched_tool.likely_neighbors),
                    "required_params": json.dumps(enriched_tool.required_params),
                    "parameters": json.dumps(enriched_tool.parameters),
                    "embedding_text": enriched_tool.embedding_text,
                    "name": enriched_tool.name,  # Store name in metadata for easy retrieval
                }

                # Use tool name as ID (with integration prefix for uniqueness)
                vector_id = f"{integration_name}_{enriched_tool.name}"

                # Create Document object
                doc = Document(
                    page_content=doc_content,
                    metadata=metadata,
                )
                documents.append((doc, vector_id))

            except (ValueError, TypeError, KeyError) as exc:
                logger.exception("Failed to prepare %s: %s", enriched_tool.name, exc)

        # Batch add to Chroma
        if documents:
            try:
                # Chroma's add_documents doesn't support ids directly, so we use add_texts
                # But we need to extract texts, metadatas, and ids
                texts = [doc.page_content for doc, _ in documents]
                metadatas = [doc.metadata for doc, _ in documents]
                ids = [doc_id for _, doc_id in documents]

                vector_store.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                    ids=ids,
                )
                logger.info(
                    "✅ Stored %s/{len(enriched_tools)} tools for {integration_name} in Chroma",
                    len(documents)
                )
            except Exception as e:
                logger.error("Failed to store tools in Chroma: %s", e)
                raise

    def _filter_by_integration(
        self,
        results: List[tuple],
        integration_names: List[str],
        top_k: int
    ) -> List[tuple]:
        """Filter results by integration names."""
        if not integration_names:
            return results[:top_k]

        filtered_results = []
        for doc, score in results:
            doc_integration = doc.metadata.get("integration", "").lower()
            if doc_integration in integration_names:
                filtered_results.append((doc, score))
        return filtered_results[:top_k]

    def _parse_metadata_json(self, metadata: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """Parse JSON fields from metadata."""
        parsed = {
            "use_cases": [],
            "likely_neighbors": [],
            "required_params": [],
            "parameters": {}
        }

        try:
            if metadata.get("use_cases"):
                parsed["use_cases"] = json.loads(metadata["use_cases"])
            if metadata.get("likely_neighbors"):
                parsed["likely_neighbors"] = json.loads(metadata["likely_neighbors"])
            if metadata.get("required_params"):
                parsed["required_params"] = json.loads(metadata["required_params"])
            if metadata.get("parameters"):
                parsed["parameters"] = json.loads(metadata["parameters"])
        except json.JSONDecodeError:
            logger.warning("Failed to parse metadata for %s", tool_name)

        return parsed

    def _extract_tool_name(self, metadata: Dict[str, Any]) -> str:
        """Extract tool name from metadata with fallback."""
        tool_name = metadata.get("name", "")
        if not tool_name:
            doc_id = metadata.get("id", "")
            if "_" in doc_id:
                tool_name = doc_id.split("_", 1)[1]
            else:
                tool_name = doc_id
        return tool_name

    def _find_matching_neighbor(
        self,
        vector_store: Chroma,
        neighbor_name: str,
        integration: Optional[str]
    ) -> Optional[tuple]:
        """Find a matching neighbor by name and integration."""
        neighbor_results = vector_store.similarity_search_with_score(
            query=neighbor_name,
            k=10,
        )

        for neighbor_doc, _ in neighbor_results:
            neighbor_metadata = neighbor_doc.metadata
            if neighbor_metadata.get("name") == neighbor_name:
                if not integration or neighbor_metadata.get("integration") == integration:
                    return (neighbor_doc, neighbor_metadata)
        return None

    def _expand_neighbors(
        self,
        vector_store: Chroma,
        tool_results: List[Dict[str, Any]],
        tool_metadata_map: Dict[str, Any],
        selected_tool_names: set
    ) -> List[Dict[str, Any]]:
        """Expand results with graph neighbors."""
        expanded_results = []
        logger.debug("\n--- Expanded Tools (Graph Neighbors) ---")

        for tool_dict in tool_results:
            tool_name = tool_dict.get("name")
            if not tool_name:
                continue

            metadata = tool_metadata_map.get(tool_name, {})
            likely_neighbors = metadata.get("likely_neighbors", [])

            for neighbor_name in likely_neighbors:
                if neighbor_name in selected_tool_names:
                    continue

                try:
                    neighbor_integration = metadata.get("integration")
                    matching_neighbor = self._find_matching_neighbor(
                        vector_store, neighbor_name, neighbor_integration
                    )

                    if matching_neighbor:
                        _, neighbor_metadata = matching_neighbor
                        parsed = self._parse_metadata_json(neighbor_metadata, neighbor_name)

                        logger.debug("Adding Neighbor: %s (related to {tool_name})", neighbor_name)
                        selected_tool_names.add(neighbor_name)

                        tool_metadata_map[neighbor_name] = {
                            **neighbor_metadata,
                            **parsed
                        }

                        expanded_results.append({
                            "name": neighbor_name,
                            "description": neighbor_metadata.get("description", ""),
                            "parameters": parsed["parameters"]
                        })
                except (ValueError, KeyError, TypeError) as exc:
                    logger.warning("Failed to load neighbor %s: %s", neighbor_name, exc)

        return expanded_results

    async def query(
        self,
        query: str,
        integration_name: Optional[List[str]] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Query tools from Chroma using semantic search.
        Implements Hub & Spoke method: semantic search + dependency/neighbor expansion.

        Args:
            query: Search query string.
            integration_name: Optional list of integration names
                (e.g., ["github", "asana"]) for filtering.
            top_k: Number of top results to return from semantic search.

        Returns:
            List of tool dictionaries compatible with OpenAI tool schema.
        """
        vector_store = self._get_vector_store()

        integration_names = []
        if integration_name:
            integration_names = [
                ns.lower() if isinstance(ns, str) else str(ns).lower()
                for ns in integration_name
            ]

        try:
            query_top_k = top_k * 5 if integration_names else top_k
            results = vector_store.similarity_search_with_score(
                query=query.replace("\n", " "),
                k=query_top_k,
            )

            results = self._filter_by_integration(results, integration_names, top_k)
            if not results:
                return []

            selected_tool_names = set()
            tool_results: List[Dict[str, Any]] = []
            tool_metadata_map = {}

            logger.debug("\n--- Anchor Tools (Vector Match) ---")
            for doc, _ in results:
                metadata = doc.metadata
                tool_name = self._extract_tool_name(metadata)

                if tool_name and tool_name not in selected_tool_names:
                    logger.debug("Found: %s (score: {score:.3f})", tool_name)
                    selected_tool_names.add(tool_name)

                    parsed = self._parse_metadata_json(metadata, tool_name)
                    tool_metadata_map[tool_name] = {**metadata, **parsed}

                    tool_results.append({
                        "name": tool_name,
                        "description": metadata.get("description", ""),
                        "parameters": parsed["parameters"]
                    })

            tool_results = tool_results[:top_k]

            expanded_results = self._expand_neighbors(
                vector_store, tool_results, tool_metadata_map, selected_tool_names
            )

            return tool_results + expanded_results
        except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Query boundary; catch all errors and return empty results
            logger.exception("Chroma query failed: %s", e)
            return []

    def _enrich_tool_metadata(self, tool: Tool) -> EnrichedTool:
        """
        Uses LLM to generate rich metadata for tool retrieval.

        Args:
            tool: Tool object to enrich.

        Returns:
            EnrichedTool with metadata.
        """
        # Check if parameters schema is empty - if so, ask LLM to infer it
        has_empty_schema = not tool.function.parameters or tool.function.parameters == {}

        if has_empty_schema:
            prompt = f"""
            Analyze this tool definition:
            Name: {tool.function.name}
            Description: {tool.function.description}
            Parameters Schema: EMPTY - schema not provided

            I need to build a smart retrieval index. Provide the following in JSON format:
            1. "use_cases": List of 3-5 specific user intent questions this tool solves (e.g. "How do I delete a file?").
            2. "likely_neighbors": List of actual tool names likely used immediately BEFORE or AFTER this tool in a workflow (must be actual tool names, e.g. "GITHUB_LIST_REPOSITORY_INVITATIONS").
            3. "required_params": List of parameter names required to use this tool (e.g. "emails", "invitation_id"). Extract from description.
            4. "parameters_schema": Infer the parameter schema from the description. Return a JSON object with parameter names as keys
               and their schema as values. Follow JSON Schema format:
               {{"param_name": {{"type": "string|array|object|integer|boolean",
               "description": "...", "items": {{...}} if array, "properties": {{...}} if object}}}}
            5. "embedding_text": A consolidated paragraph combining name, description, and use cases for vector embedding.

            Return ONLY valid JSON matching this structure.
            """
        else:
            prompt = f"""
            Analyze this tool definition:
            Name: {tool.function.name}
            Description: {tool.function.description}
            Parameters: {json.dumps(tool.function.parameters)}

            I need to build a smart retrieval index. Provide the following in JSON format:
            1. "use_cases": List of 3-5 specific user intent questions this tool solves (e.g. "How do I delete a file?").
            2. "likely_neighbors": List of actual tool names likely used immediately BEFORE or AFTER this tool in a workflow (must be actual tool names, e.g. "GITHUB_LIST_REPOSITORY_INVITATIONS").
            3. "required_params": List of parameter names required to use this tool (e.g. "invitation_id", "user_id").
            4. "embedding_text": A consolidated paragraph combining name, description, and use cases for vector embedding.

            Return ONLY valid JSON matching this structure.
            """

        # Use async client in sync context (we're in a thread pool)
        # Create a new event loop for this thread
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        response = loop.run_until_complete(
            self.async_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a backend architect optimizing tool retrieval."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
        )

        content = json.loads(response.choices[0].message.content)

        # If schema was empty and LLM inferred parameters, use them
        if has_empty_schema and content.get('parameters_schema'):
            inferred_params = content.get('parameters_schema', {})
            if inferred_params:
                # Update tool.function.parameters with inferred schema
                tool.function.parameters = inferred_params
                logger.debug(
                    "📝 Inferred parameters for %s: {list(inferred_params.keys())}",
                    tool.function.name
                )
        elif has_empty_schema:
            # If LLM didn't provide parameters_schema, log warning
            logger.warning(
                "⚠️ Warning: Empty schema for %s but LLM didn't infer parameters_schema",
                tool.function.name
            )

        return EnrichedTool(
            name=tool.function.name,
            description=tool.function.description or "",
            parameters=tool.function.parameters or {},
            use_cases=content.get('use_cases', []),
            likely_neighbors=content.get('likely_neighbors', []),
            required_params=content.get('required_params', []),
            embedding_text=content.get('embedding_text', ""),
            original_tool=tool
        )

    def index_exists(self) -> bool:
        """
        Check if tool index already exists.

        Returns:
            True if index exists, False otherwise.
        """
        try:
            vector_store = self._get_vector_store()
            # Try a safe operation to check if collection is available
            # Using empty query with k=1 to minimize impact
            _ = vector_store.similarity_search("", k=1)
            return True
        except (ValueError, RuntimeError, AttributeError):
            # Collection doesn't exist or can't be accessed
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the tool index.

        Returns:
            Dictionary with index statistics.
        """
        try:
            vector_store = self._get_vector_store()
            # Try a safe operation to infer collection stats
            # Note: Without protected access, exact count is not available via public API
            # This is a workaround - consider caching count after operations
            results = vector_store.similarity_search("", k=1)
            # If successful, collection exists; exact count not available without protected access
            return {
                "exists": True,
                "tool_count": len(results) if results else 0,
                "persist_directory": str(self.persist_directory),
                "note": "Exact count unavailable via public API",
            }
        except (ValueError, RuntimeError, AttributeError):
            # Collection doesn't exist or can't be accessed
            return {
                "exists": False,
                "tool_count": 0,
                "persist_directory": str(self.persist_directory),
            }
