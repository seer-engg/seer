import json
import asyncio
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import AsyncOpenAI
from tqdm import tqdm
from pinecone import Pinecone
from .models import Tool, EnrichedTool, ToolFunction

class ToolHub:
    def __init__(
        self, 
        openai_api_key: str, 
        pinecone_index_name: str,
        pinecone_api_key: str,
        llm_model: str = "gpt-5-mini", 
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: Optional[int] = None
    ):
        """
        Initialize the ToolHub (Pinecone-only).

        Args:
            openai_api_key: The OpenAI API key (required).
            llm_model: The model used for enrichment (default: gpt-5-mini).
            embedding_model: The model used for vector embedding (default: text-embedding-3-small).
            pinecone_api_key: Pinecone API key for vector store (required).
            pinecone_index_name: Pinecone index name (required).
            embedding_dimensions: Optional embedding dimensions (must match index dimension).
        """
        if not openai_api_key:
            raise ValueError("openai_api_key is required")
        if not pinecone_api_key:
            raise ValueError("pinecone_api_key is required")
        if not pinecone_index_name:
            raise ValueError("pinecone_index_name is required")
            
        self.async_client = AsyncOpenAI(api_key=openai_api_key)
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        
        # Pinecone setup
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_index_name = pinecone_index_name
        
        try:
            self.pc = Pinecone(api_key=pinecone_api_key)
        except Exception as e:
            raise ValueError(f"Could not initialize Pinecone: {e}") from e

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
                    # Direct function/action definition (e.g., from Composio's ActionModel)
                    normalized_tools.append(Tool(function=ToolFunction(**t)))
                else:
                    try:
                        normalized_tools.append(Tool.from_dict(t))
                    except Exception as e:
                        print(f"Skipping invalid tool structure: {t.keys()} - {e}")
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
        Ingests tools, enriches them with metadata, and stores them in Pinecone vector store.
        
        Args:
            tools: List of Tool objects or dictionaries matching OpenAI tool schema.
            integration_name: Integration name (e.g., "github", "asana") for namespace isolation.
            max_workers: Number of concurrent threads for enrichment (default: 10).
        """
        if not self.pc:
            raise ValueError("Pinecone not initialized. Provide pinecone_api_key in __init__.")
        
        integration_name = integration_name.lower()
        
        print(f"Ingesting {len(tools)} tools for {integration_name} into Pinecone...")
        
        # Normalize inputs
        normalized_tools = self._normalize_tools(tools)
        
        if not normalized_tools:
            print("No valid tools to ingest after normalization.")
            return
        
        # Enrich tools
        enriched_tools = []
        print(f"Enriching tools with concurrency (max_workers={max_workers})...")
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
                    print(f"Failed to enrich {tool.function.name}: {e}")
        
        # Generate embeddings and store in Pinecone
        print(f"Generating embeddings and storing {len(enriched_tools)} enriched tools in Pinecone...")
        
        # Get Pinecone index
        index = self.pc.Index(self.pinecone_index_name)
        
        # Prepare vectors for batch upload
        vectors_to_upsert = []
        stored_count = 0
        
        for enriched_tool in tqdm(enriched_tools, desc="Storing Tools"):
            try:
                # Generate embedding for embedding_text
                params = {
                    "input": enriched_tool.embedding_text.replace("\n", " "),
                    "model": self.embedding_model
                }
                if self.embedding_dimensions:
                    params["dimensions"] = self.embedding_dimensions
                embedding_response = await self.async_client.embeddings.create(**params)
                embedding = embedding_response.data[0].embedding
                
                # Prepare flattened metadata structure
                # Pinecone supports: strings, numbers, booleans, lists of strings
                # Only nested structures (like parameters dict) need to be JSON strings
                metadata = {
                    # Filtering/Display fields
                    "integration": integration_name,  # string
                    "description": enriched_tool.description,  # string (full, not truncated)
                    
                    # Flattened lists (directly accessible, no JSON parsing needed)
                    "use_cases": enriched_tool.use_cases,  # list of strings
                    "likely_neighbors": enriched_tool.likely_neighbors,  # list of strings
                    "required_params": enriched_tool.required_params,  # list of strings
                    
                    # Only nested structure (must be JSON string)
                    "parameters": json.dumps(enriched_tool.parameters),  # JSON string (nested dict)
                    
                    # Backward compatibility: embedding_text for re-embedding with different models
                    "embedding_text": enriched_tool.embedding_text,  # string (full, for re-embedding)
                }
                
                # Vector ID: use tool_name directly (tool names already include integration prefix)
                vector_id = enriched_tool.name
                
                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
                
                stored_count += 1
                
                # Batch upload every 100 vectors
                if len(vectors_to_upsert) >= 100:
                    index.upsert(vectors=vectors_to_upsert, namespace=integration_name)
                    vectors_to_upsert = []
                    
            except Exception as e:
                print(f"Failed to store {enriched_tool.name}: {e}")
        
        # Upload remaining vectors
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert, namespace=integration_name)
        
        print(f"✅ Stored {stored_count}/{len(enriched_tools)} tools for {integration_name} in Pinecone")

    async def query(
        self,
        query: str,
        integration_name: Optional[List[str]] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Query tools from Pinecone using semantic search.
        Implements Hub & Spoke method: semantic search + dependency/neighbor expansion.
        
        Args:
            query: Search query string.
            integration_name: Optional list of integration names (e.g., ["github", "asana"]) for multi-namespace search.
            top_k: Number of top results to return from semantic search.
        
        Returns:
            List of tool dictionaries compatible with OpenAI tool schema.
        """
        if not self.pc:
            raise ValueError("Pinecone not initialized. Provide pinecone_api_key in __init__.")
        
        # Normalize integration_name to list of lowercase strings
        if integration_name is None:
            integration_names = []
        else:
            integration_names = [ns.lower() if isinstance(ns, str) else str(ns).lower() for ns in integration_name]
        
        # Generate query embedding (async)
        query_embedding = await self._get_embedding_async(query.replace("\n", " "))
        query_vector = query_embedding[0]  # Already a list
        
        # Get Pinecone index host for async operations
        # Wrap describe_index in asyncio.to_thread to avoid blocking the event loop
        index_info = await asyncio.to_thread(self.pc.describe_index, name=self.pinecone_index_name)
        index_host = index_info.host
        
        # Base query kwargs
        base_query_kwargs = {
            "vector": query_vector,
            "top_k": top_k * 2 if len(integration_names) > 1 else top_k,  # Get more results if querying multiple namespaces
            "include_metadata": True
        }
        
        try:
            # Use IndexAsyncio for async operations - reuse same context for query and fetches
            async with self.pc.IndexAsyncio(host=index_host) as index:
                # Query multiple namespaces in parallel if specified
                if integration_names:
                    # Query each namespace in parallel
                    query_tasks = []
                    for ns in integration_names:
                        query_kwargs = base_query_kwargs.copy()
                        query_kwargs["namespace"] = ns
                        query_tasks.append(index.query(**query_kwargs))
                    
                    # Execute all queries in parallel
                    query_responses = await asyncio.gather(*query_tasks, return_exceptions=True)
                    
                    # Collect all matches from all namespaces
                    all_matches = []
                    for i, response in enumerate(query_responses):
                        if isinstance(response, Exception):
                            print(f"Pinecone query failed for namespace '{integration_names[i]}': {response}")
                            continue
                        if response.matches:
                            all_matches.extend(response.matches)
                    
                    # Sort by score (descending) and deduplicate by ID
                    all_matches.sort(key=lambda m: m.score, reverse=True)
                    seen_ids = set()
                    unique_matches = []
                    for match in all_matches:
                        if match.id not in seen_ids:
                            seen_ids.add(match.id)
                            unique_matches.append(match)
                    
                    # Create a mock query_response with merged matches
                    class MockQueryResponse:
                        def __init__(self, matches):
                            self.matches = matches[:top_k]  # Take top_k after deduplication
                    
                    query_response = MockQueryResponse(unique_matches)
                else:
                    # Query default namespace (no namespace specified)
                    query_response = await index.query(**base_query_kwargs)
                
                if not query_response.matches:
                    return []
                
                # Extract tool names from search results
                selected_tool_names = set()
                results: List[Dict[str, Any]] = []
                tool_metadata_map = {}  # Store metadata for expansion
                
                print(f"\n--- Anchor Tools (Vector Match) ---")
                for match in query_response.matches:
                    metadata = match.metadata
                    tool_name = match.id  # Use vector ID as tool name (ID is the tool name)
                    
                    if tool_name and tool_name not in selected_tool_names:
                        print(f"Found: {tool_name} (score: {match.score:.3f})")
                        selected_tool_names.add(tool_name)
                        
                        # Direct access - no JSON parsing needed (except parameters)
                        tool_metadata_map[tool_name] = metadata
                        
                        # Parse only parameters (the only nested structure)
                        parameters = {}
                        try:
                            parameters_str = metadata.get("parameters", "{}")
                            if parameters_str:
                                parameters = json.loads(parameters_str)
                        except json.JSONDecodeError:
                            pass
                        
                        tool_dict = {
                            "name": tool_name,  # Use ID directly
                            "description": metadata.get("description", ""),
                            "parameters": parameters
                        }
                        results.append(tool_dict)
                
                # 2. Graph Expansion (Spoke) - expand with neighbors only
                expanded_results: List[Dict[str, Any]] = []
                print(f"\n--- Expanded Tools (Graph Neighbors) ---")
                
                for tool_dict in results:
                    tool_name = tool_dict.get("name")
                    if not tool_name:
                        continue
                    
                    # Direct access to flattened metadata - no JSON parsing needed
                    metadata = tool_metadata_map.get(tool_name, {})
                    likely_neighbors = metadata.get("likely_neighbors", [])  # Direct access!
                    
                    # Check neighbors - fetch from Pinecone (async, reuse same index context)
                    for neighbor_name in likely_neighbors:
                        if neighbor_name not in selected_tool_names:
                            try:
                                # Fetch neighbor tool by ID (tool_name directly, no integration prefix)
                                # Use namespace from the tool's metadata to fetch from correct namespace
                                neighbor_id = neighbor_name
                                # Get namespace from metadata, or try first integration_name, or None (default namespace)
                                neighbor_namespace = metadata.get("integration")
                                if not neighbor_namespace and integration_names:
                                    neighbor_namespace = integration_names[0]  # Try first namespace if metadata doesn't have it
                                
                                # Use async fetch with same index context
                                fetch_response = await index.fetch(ids=[neighbor_id], namespace=neighbor_namespace)
                                
                                if neighbor_id in fetch_response.vectors:
                                    neighbor_metadata = fetch_response.vectors[neighbor_id].metadata  # Already flat!
                                    
                                    # Parse only parameters (the only nested structure)
                                    neighbor_parameters = {}
                                    try:
                                        neighbor_parameters_str = neighbor_metadata.get("parameters", "{}")
                                        if neighbor_parameters_str:
                                            neighbor_parameters = json.loads(neighbor_parameters_str)
                                    except json.JSONDecodeError:
                                        pass
                                    
                                    print(f"Adding Neighbor: {neighbor_name} (related to {tool_name})")
                                    selected_tool_names.add(neighbor_name)
                                    tool_metadata_map[neighbor_name] = neighbor_metadata
                                    expanded_results.append({
                                        "name": neighbor_name,  # Use ID directly
                                        "description": neighbor_metadata.get("description", ""),
                                        "parameters": neighbor_parameters
                                    })
                            except Exception as e:
                                print(f"Failed to load neighbor {neighbor_name}: {e}")
                
                final_selection = results + expanded_results
                return final_selection
        except Exception as e:
            print(f"Pinecone query failed: {e}")
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
            Parameters Schema: EMPTY - schema not provided by Composio

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
                print(f"📝 Inferred parameters for {tool.function.name}: {list(inferred_params.keys())}")
        elif has_empty_schema:
            # If LLM didn't provide parameters_schema, log warning
            print(f"⚠️ Warning: Empty schema for {tool.function.name} but LLM didn't infer parameters_schema")
        
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
    
    async def _get_embedding_async(self, text: str) -> List[float]:
        """Generates embeddings using OpenAI (asynchronous)."""
        # Replace newlines with spaces to handle multi-line inputs properly, 
        # as per OpenAI recommendation for embeddings
        cleaned_text = text.replace("\n", " ")
        
        params = {
            "input": cleaned_text,
            "model": self.embedding_model
        }
        if self.embedding_dimensions:
            params["dimensions"] = self.embedding_dimensions
        
        response = await self.async_client.embeddings.create(**params)
        return [response.data[0].embedding]  # Return as list for consistency
