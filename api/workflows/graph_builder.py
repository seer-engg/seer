"""
Workflow Graph Builder

Converts workflow JSON spec to LangGraph StateGraph with hybrid input resolution.
"""
from typing import Any, Dict, List, Optional, Callable, Literal, Set, Tuple
from collections import defaultdict
from typing_extensions import TypedDict
import re

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from shared.logger import get_logger
from api.agents.checkpointer import get_checkpointer

from .state import WorkflowState
from .schema import (
    BlockDefinition,
    EdgeDefinition,
    WorkflowSchema,
    BlockType,
    validate_workflow_graph,
)
from .nodes import get_node_function, resolve_inputs
from .models import Workflow
from .alias_utils import derive_block_aliases, collect_input_variables

logger = get_logger("api.workflows.graph_builder")

VAR_PATTERN = re.compile(r"\{\{([^}]+)\}\}")


class WorkflowGraphBuilder:
    """Builds LangGraph StateGraph from workflow JSON spec."""
    
    def __init__(self):
        """Initialize graph builder."""
        self._compiled_graphs: Dict[int, Any] = {}  # Cache compiled graphs
    
    def _iter_config_strings(self, value: Any, path: str = "config") -> List[Tuple[str, str]]:
        """Yield (path, string) pairs from nested block config structures."""
        if isinstance(value, str):
            return [(path, value)]
        strings: List[Tuple[str, str]] = []
        if isinstance(value, dict):
            for key, nested in value.items():
                strings.extend(self._iter_config_strings(nested, f"{path}.{key}"))
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                strings.extend(self._iter_config_strings(item, f"{path}[{idx}]"))
        return strings
    
    def _validate_template_references(
        self,
        blocks: List[BlockDefinition],
        alias_map: Dict[str, List[str]],
        input_variables: Set[str],
    ) -> None:
        """Ensure all {{ }} references target known block aliases."""
        alias_prefixes: Set[str] = set()
        for block in blocks:
            alias_prefixes.add(block.id)
            alias_prefixes.update(alias_map.get(block.id, []))
        known_simple_vars: Set[str] = set(input_variables or set())
        known_simple_vars.update({"output", "structured_output", "condition_result", "route", "items", "count"})
        
        for block in blocks:
            for path, text in self._iter_config_strings(block.config):
                if "{{" not in text:
                    continue
                
                matches = list(VAR_PATTERN.finditer(text))
                if not matches:
                    raise ValueError(
                        f"Block '{block.id}' contains '{{{{' without closing '}}}}' in {path}."
                    )
                
                for match in matches:
                    var_name = match.group(1).strip()
                    if not var_name:
                        raise ValueError(
                            f"Block '{block.id}' has an empty template variable in {path}."
                        )
                    if any(ch.isspace() for ch in var_name):
                        raise ValueError(
                            f"Block '{block.id}' references '{{{{{var_name}}}}}' with whitespace in {path}. "
                            "Template identifiers cannot contain spaces."
                        )
                    
                    if "." in var_name:
                        prefix = var_name.split(".", 1)[0]
                        if prefix not in alias_prefixes:
                            raise ValueError(
                                f"Block '{block.id}' references '{{{{{var_name}}}}}' in {path}, "
                                "but no block or alias named "
                                f"'{prefix}' exists. Use the sanitized alias from the builder."
                            )
                    else:
                        if var_name not in known_simple_vars:
                            logger.debug(
                                "Template reference '{{%s}}' in block '%s' at %s is assumed to be a simple variable.",
                                var_name,
                                block.id,
                                path,
                            )
    
    def _resolve_block_inputs(
        self,
        block: BlockDefinition,
        edges: List[EdgeDefinition],
        all_blocks: Dict[str, BlockDefinition],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Legacy helper kept for backward compatibility.
        
        Data is now passed exclusively via template references, so explicit
        edge-based input resolution is no longer required.
        """
        return {}
    
    def _create_node_function(
        self,
        block: BlockDefinition,
        all_blocks: Dict[str, BlockDefinition],
        all_edges: List[EdgeDefinition],
        user_id: Optional[str] = None,
        execution: Optional[Any] = None,
    ) -> Callable:
        """
        Create a node function for a block with input resolution.
        
        Args:
            block: Block definition
            all_blocks: All blocks in workflow
            all_edges: All edges in workflow
            user_id: User ID for tool execution
            execution: Execution record for logging
        
        Returns:
            Node function that takes state and returns updated state
        """
        # Pre-compute input resolution
        input_resolution = self._resolve_block_inputs(block, all_edges, all_blocks)
        
        # Get base node function
        base_func = get_node_function(block.type)
        if not base_func:
            raise ValueError(f"No node function for block type: {block.type}")
        
        # Create wrapped function with input resolution
        async def node_func(state: WorkflowState) -> WorkflowState:
            try:
                # Special handling for tool nodes (need user_id)
                if block.type == BlockType.TOOL:
                    return await base_func(
                        state,
                        block,
                        input_resolution,
                        user_id=user_id,
                        execution=execution,
                    )
                else:
                    return await base_func(
                        state,
                        block,
                        input_resolution,
                        execution=execution,
                    )
            except Exception as e:
                logger.error(f"Error executing block {block.id}: {e}", exc_info=True)
                raise
        
        return node_func
    
    def _create_conditional_edge_function(
        self,
        block: BlockDefinition,
    ) -> Callable:
        """
        Create conditional edge function for if/else blocks.
        
        Args:
            block: If/else block definition
        
        Returns:
            Function that routes based on condition result
        """
        def conditional_route(state: WorkflowState) -> Literal["true", "false"]:
            # Get condition result from block output
            block_output = state["block_outputs"].get(block.id, {})
            route = block_output.get("route", "false")
            return route  # type: ignore
        
        return conditional_route
    
    def _find_input_blocks(self, blocks: List[BlockDefinition], edges: List[EdgeDefinition]) -> List[str]:
        """Find blocks with no incoming edges (input blocks)."""
        incoming = {edge.target for edge in edges}
        return [block.id for block in blocks if block.id not in incoming]
    
    def _find_output_blocks(self, blocks: List[BlockDefinition], edges: List[EdgeDefinition]) -> List[str]:
        """Find blocks with no outgoing edges (these become workflow outputs)."""
        outgoing = {edge.source for edge in edges}
        return [block.id for block in blocks if block.id not in outgoing]
    
    async def build_graph(
        self,
        workflow: Workflow,
        checkpointer: Optional[AsyncPostgresSaver] = None,
        user_id: Optional[str] = None,
    ) -> Any:
        """
        Build and compile LangGraph from workflow JSON spec.
        
        Args:
            workflow: Workflow model instance
            checkpointer: Optional checkpointer (will fetch if not provided)
            user_id: User ID for tool execution
        
        Returns:
            Compiled LangGraph graph
        """
        # Validate and parse workflow schema
        schema = validate_workflow_graph(workflow.graph_data)
        graph_data = workflow.graph_data or {}
        alias_map = derive_block_aliases(graph_data)
        input_variables = collect_input_variables(graph_data)
        self._validate_template_references(schema.blocks, alias_map, input_variables)
        
        # Get checkpointer if not provided
        if checkpointer is None:
            checkpointer = await get_checkpointer()
        
        # Validate checkpointer has required methods (get_checkpointer may return None if initialization failed)
        if checkpointer is not None and not hasattr(checkpointer, 'get_next_version'):
            logger.warning("Checkpointer missing get_next_version method, compiling without checkpointer")
            checkpointer = None
        
        # Create StateGraph
        workflow_graph = StateGraph(WorkflowState)
        
        # Build blocks dict for quick lookup
        blocks_dict = {block.id: block for block in schema.blocks}
        
        # Create node functions for each block
        node_functions = {}
        for block in schema.blocks:
            node_func = self._create_node_function(
                block,
                blocks_dict,
                schema.edges,
                user_id=user_id,
            )
            workflow_graph.add_node(block.id, node_func)
            node_functions[block.id] = node_func
        
        # Build edges
        # Find input blocks (no incoming edges)
        input_blocks = self._find_input_blocks(schema.blocks, schema.edges)
        
        # Connect START to input blocks
        if input_blocks:
            if len(input_blocks) == 1:
                workflow_graph.add_edge(START, input_blocks[0])
            else:
                # Multiple input blocks - connect START to first one
                # (could be improved with a merge node)
                workflow_graph.add_edge(START, input_blocks[0])
        
        # Group edges by source to handle conditional edges properly
        edges_by_source = defaultdict(list)
        for edge in schema.edges:
            edges_by_source[edge.source].append(edge)
        
        # Build edges
        for source_id, edges in edges_by_source.items():
            source_block = blocks_dict[source_id]
            
            # Check if source is an if_else block
            if source_block.type == BlockType.IF_ELSE:
                # Use conditional edge - need to map true/false routes
                conditional_func = self._create_conditional_edge_function(source_block)
                
                # Build route map from edges
                route_map = {}
                for edge in edges:
                    target_branch = edge.branch
                    if target_branch not in ("true", "false"):
                        target_branch = "true" if "true" not in route_map else "false"
                    if target_branch in route_map:
                        continue
                    route_map[target_branch] = edge.target
                
                # Ensure both routes are defined
                if "true" not in route_map:
                    route_map["true"] = END
                if "false" not in route_map:
                    route_map["false"] = END
                
                workflow_graph.add_conditional_edges(
                    source_id,
                    conditional_func,
                    route_map,
                )
            else:
                # Regular edges
                for edge in edges:
                    workflow_graph.add_edge(edge.source, edge.target)
        
        # Connect output blocks to END
        output_blocks = self._find_output_blocks(schema.blocks, schema.edges)
        if output_blocks:
            if len(output_blocks) == 1:
                workflow_graph.add_edge(output_blocks[0], END)
            else:
                # Multiple output blocks - connect last one to END
                # (could be improved with a merge node)
                workflow_graph.add_edge(output_blocks[-1], END)
        
        # Compile graph
        # Note: If checkpointer is None or invalid, compile without checkpointer
        # This allows workflows to run even if checkpointing isn't configured
        if checkpointer is not None and hasattr(checkpointer, 'get_next_version'):
            compiled_graph = workflow_graph.compile(checkpointer=checkpointer)
        else:
            logger.warning("Compiling workflow graph without checkpointer")
            compiled_graph = workflow_graph.compile()
        
        return compiled_graph
    
    async def get_compiled_graph(
        self,
        workflow_id: int,
        workflow: Optional[Workflow] = None,
        user_id: Optional[str] = None,
    ) -> Any:
        """
        Get compiled graph, always rebuilding to ensure latest code is used.
        
        Args:
            workflow_id: Workflow ID
            workflow: Optional workflow instance (will fetch if not provided)
            user_id: User ID for tool execution
        
        Returns:
            Compiled LangGraph graph
        """
        # Always rebuild the graph to ensure latest node function code is used
        # This is important because node functions capture code at compile time
        # and caching would prevent code updates from taking effect
        
        # Build and compile
        if workflow is None:
            from .services import get_workflow
            workflow = await get_workflow(workflow_id)
        
        compiled = await self.build_graph(workflow, user_id=user_id)
        
        return compiled
    
    def invalidate_cache(self, workflow_id: int):
        """Invalidate cached graph for a workflow."""
        self._compiled_graphs.pop(workflow_id, None)


# Global graph builder instance
_graph_builder = WorkflowGraphBuilder()


async def get_workflow_graph_builder() -> WorkflowGraphBuilder:
    """Get global graph builder instance."""
    return _graph_builder


__all__ = [
    "WorkflowGraphBuilder",
    "get_workflow_graph_builder",
]

