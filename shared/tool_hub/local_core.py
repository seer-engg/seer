"""
Local ToolHub implementation using Chroma for vector storage.

Replaces Pinecone with local Chroma vector store for open-source deployment.
"""
import json
import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import AsyncOpenAI
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from .models import Tool, EnrichedTool, ToolFunction
from shared.logger import get_logger

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
        
        logger.info(f"LocalToolHub initialized with persist_directory={persist_directory}")

    def _get_vector_store(self) -> Chroma:
        """Get or create Chroma vector store instance."""
        if self._vector_store is None:
            self._vector_store = Chroma(
                persist_directory=str(self.persist_directory),
                collection_name=self._collection_name,
                embedding_function=self.embeddings,
            )
        return self._vector_store

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
            # Filter out deprecated tools
            if isinstance(t, dict):
                description = t.get("description", "") or t.get("function", {}).get("description", "")
                if "deprecated" in description.lower():
                    continue
            elif isinstance(t, Tool):
                if "deprecated" in (t.function.description or "").lower():
                    continue
            
            # Normalize to Tool object
            if isinstance(t, dict):
                if "function" in t:
                    normalized_tools.append(Tool.from_dict(t))
                elif "parameters" in t:
                    # Direct function/action definition
                    normalized_tools.append(Tool(function=ToolFunction(**t)))
                else:
                    try:
                        normalized_tools.append(Tool.from_dict(t))
                    except Exception as e:
                        logger.warning(f"Skipping invalid tool structure: {t.keys()} - {e}")
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
        
        logger.info(f"Ingesting {len(tools)} tools for {integration_name} into Chroma...")
        
        # Normalize inputs
        normalized_tools = self._normalize_tools(tools)
        
        if not normalized_tools:
            logger.warning("No valid tools to ingest after normalization.")
            return
        
        # Enrich tools
        enriched_tools = []
        logger.info(f"Enriching tools with concurrency (max_workers={max_workers})...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tool = {
                executor.submit(self._enrich_tool_metadata, tool): tool
                for tool in normalized_tools
            }
            
            for future in tqdm(as_completed(future_to_tool), total=len(normalized_tools), desc="Enriching Tools"):
                tool = future_to_tool[future]
                try:
                    enriched = future.result()
                    enriched_tools.append(enriched)
                except Exception as e:
                    logger.error(f"Failed to enrich {tool.function.name}: {e}")
        
        # Generate embeddings and store in Chroma
        logger.info(f"Generating embeddings and storing {len(enriched_tools)} enriched tools in Chroma...")
        
        vector_store = self._get_vector_store()
        
        # Prepare documents for Chroma
        documents = []
        
        for enriched_tool in tqdm(enriched_tools, desc="Storing Tools"):
            try:
                # Use embedding_text as the document content
                doc_content = enriched_tool.embedding_text.replace("\n", " ")
                
                # Prepare metadata (Chroma supports nested dicts, but we'll use strings for lists to be safe)
                metadata = {
                    "integration": integration_name,
                    "description": enriched_tool.description,
                    "use_cases": json.dumps(enriched_tool.use_cases),  # Chroma prefers strings for lists
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
                
            except Exception as e:
                logger.error(f"Failed to prepare {enriched_tool.name}: {e}")
        
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
                logger.info(f"✅ Stored {len(documents)}/{len(enriched_tools)} tools for {integration_name} in Chroma")
            except Exception as e:
                logger.error(f"Failed to store tools in Chroma: {e}")
                raise

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
            integration_name: Optional list of integration names (e.g., ["github", "asana"]) for filtering.
            top_k: Number of top results to return from semantic search.
        
        Returns:
            List of tool dictionaries compatible with OpenAI tool schema.
        """
        vector_store = self._get_vector_store()
        
        # Normalize integration_name to list of lowercase strings
        if integration_name is None:
            integration_names = []
        else:
            integration_names = [ns.lower() if isinstance(ns, str) else str(ns).lower() for ns in integration_name]
        
        try:
            # Query Chroma
            # Get more results if querying multiple integrations for better coverage
            # Note: LangChain Chroma's similarity_search_with_score doesn't support 'where' parameter
            # We'll query more results and filter in Python
            query_top_k = top_k * 5 if integration_names else top_k  # Get more results to filter from
            
            # Query without filter (LangChain Chroma doesn't support where parameter)
            results = vector_store.similarity_search_with_score(
                query=query.replace("\n", " "),
                k=query_top_k,
            )
            
            # Filter results by integration if specified
            if integration_names:
                filtered_results = []
                for doc, score in results:
                    metadata = doc.metadata
                    doc_integration = metadata.get("integration", "").lower()
                    if doc_integration in integration_names:
                        filtered_results.append((doc, score))
                results = filtered_results[:top_k]  # Limit to top_k after filtering
            else:
                results = results[:top_k]  # Limit to top_k if no filter
            
            if not results:
                return []
            
            # Extract tool names from search results
            selected_tool_names = set()
            tool_results: List[Dict[str, Any]] = []
            tool_metadata_map = {}  # Store metadata for expansion
            
            logger.debug(f"\n--- Anchor Tools (Vector Match) ---")
            for doc, score in results:
                metadata = doc.metadata
                tool_name = metadata.get("name", "")
                
                if not tool_name:
                    # Fallback: extract from ID
                    doc_id = doc.metadata.get("id", "")
                    if "_" in doc_id:
                        tool_name = doc_id.split("_", 1)[1]
                    else:
                        tool_name = doc_id
                
                if tool_name and tool_name not in selected_tool_names:
                    logger.debug(f"Found: {tool_name} (score: {score:.3f})")
                    selected_tool_names.add(tool_name)
                    
                    # Parse JSON fields from metadata
                    use_cases = []
                    likely_neighbors = []
                    required_params = []
                    parameters = {}
                    
                    try:
                        if metadata.get("use_cases"):
                            use_cases = json.loads(metadata["use_cases"])
                        if metadata.get("likely_neighbors"):
                            likely_neighbors = json.loads(metadata["likely_neighbors"])
                        if metadata.get("required_params"):
                            required_params = json.loads(metadata["required_params"])
                        if metadata.get("parameters"):
                            parameters = json.loads(metadata["parameters"])
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse metadata for {tool_name}: {e}")
                    
                    tool_metadata_map[tool_name] = {
                        **metadata,
                        "use_cases": use_cases,
                        "likely_neighbors": likely_neighbors,
                        "required_params": required_params,
                        "parameters": parameters,
                    }
                    
                    tool_dict = {
                        "name": tool_name,
                        "description": metadata.get("description", ""),
                        "parameters": parameters
                    }
                    tool_results.append(tool_dict)
            
            # Limit to top_k after deduplication
            tool_results = tool_results[:top_k]
            
            # Graph Expansion (Spoke) - expand with neighbors
            expanded_results: List[Dict[str, Any]] = []
            logger.debug(f"\n--- Expanded Tools (Graph Neighbors) ---")
            
            for tool_dict in tool_results:
                tool_name = tool_dict.get("name")
                if not tool_name:
                    continue
                
                metadata = tool_metadata_map.get(tool_name, {})
                likely_neighbors = metadata.get("likely_neighbors", [])
                
                # Check neighbors - fetch from Chroma
                for neighbor_name in likely_neighbors:
                    if neighbor_name not in selected_tool_names:
                        try:
                            # Try to fetch neighbor by name
                            # Search for neighbor in same integration namespace
                            neighbor_integration = metadata.get("integration")
                            
                            # Query without filter (LangChain Chroma doesn't support where parameter)
                            # Get multiple results and filter by name and integration
                            neighbor_results = vector_store.similarity_search_with_score(
                                query=neighbor_name,  # Use name as query
                                k=10,  # Get more results to filter from
                            )
                            
                            # Filter results by name and integration
                            matching_neighbor = None
                            for neighbor_doc, neighbor_score in neighbor_results:
                                neighbor_metadata = neighbor_doc.metadata
                                # Check if name matches and integration matches (if specified)
                                if neighbor_metadata.get("name") == neighbor_name:
                                    if not neighbor_integration or neighbor_metadata.get("integration") == neighbor_integration:
                                        matching_neighbor = (neighbor_doc, neighbor_metadata)
                                        break
                            
                            if matching_neighbor:
                                neighbor_doc, neighbor_metadata = matching_neighbor
                                
                                # Parse parameters
                                neighbor_parameters = {}
                                try:
                                    if neighbor_metadata.get("parameters"):
                                        neighbor_parameters = json.loads(neighbor_metadata["parameters"])
                                except json.JSONDecodeError:
                                    pass
                                
                                logger.debug(f"Adding Neighbor: {neighbor_name} (related to {tool_name})")
                                selected_tool_names.add(neighbor_name)
                                
                                # Store neighbor metadata
                                neighbor_use_cases = []
                                neighbor_likely_neighbors = []
                                neighbor_required_params = []
                                try:
                                    if neighbor_metadata.get("use_cases"):
                                        neighbor_use_cases = json.loads(neighbor_metadata["use_cases"])
                                    if neighbor_metadata.get("likely_neighbors"):
                                        neighbor_likely_neighbors = json.loads(neighbor_metadata["likely_neighbors"])
                                    if neighbor_metadata.get("required_params"):
                                        neighbor_required_params = json.loads(neighbor_metadata["required_params"])
                                except json.JSONDecodeError:
                                    pass
                                
                                tool_metadata_map[neighbor_name] = {
                                    **neighbor_metadata,
                                    "use_cases": neighbor_use_cases,
                                    "likely_neighbors": neighbor_likely_neighbors,
                                    "required_params": neighbor_required_params,
                                    "parameters": neighbor_parameters,
                                }
                                
                                expanded_results.append({
                                    "name": neighbor_name,
                                    "description": neighbor_metadata.get("description", ""),
                                    "parameters": neighbor_parameters
                                })
                        except Exception as e:
                            logger.warning(f"Failed to load neighbor {neighbor_name}: {e}")
            
            final_selection = tool_results + expanded_results
            return final_selection
            
        except Exception as e:
            logger.error(f"Chroma query failed: {e}")
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
            4. "parameters_schema": Infer the parameter schema from the description. Return a JSON object with parameter names as keys and their schema as values. Follow JSON Schema format: {{"param_name": {{"type": "string|array|object|integer|boolean", "description": "...", "items": {{...}} if array, "properties": {{...}} if object}}}}
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
                    {"role": "system", "content": "You are a backend architect optimizing tool retrieval."},
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
                logger.debug(f"📝 Inferred parameters for {tool.function.name}: {list(inferred_params.keys())}")
        elif has_empty_schema:
            # If LLM didn't provide parameters_schema, log warning
            logger.warning(f"⚠️ Warning: Empty schema for {tool.function.name} but LLM didn't infer parameters_schema")
        
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
            # Try to get collection count - if it exists and has data, return True
            collection = vector_store._collection
            if collection and collection.count() > 0:
                return True
        except Exception:
            pass
        return False

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the tool index.
        
        Returns:
            Dictionary with index statistics.
        """
        try:
            vector_store = self._get_vector_store()
            collection = vector_store._collection
            if collection:
                count = collection.count()
                return {
                    "exists": True,
                    "tool_count": count,
                    "persist_directory": str(self.persist_directory),
                }
        except Exception:
            pass
        return {
            "exists": False,
            "tool_count": 0,
            "persist_directory": str(self.persist_directory),
        }

