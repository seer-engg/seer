"""
Workflow executor that interprets node graph and executes blocks.
"""
import asyncio
import json
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

from shared.logger import get_logger
from shared.llm import get_llm
from shared.tools.composio import ComposioMCPClient
from composio import Composio
from composio_langchain import LangchainProvider
from langchain_core.messages import HumanMessage, SystemMessage

from .models import Workflow, WorkflowBlock, WorkflowEdge, WorkflowExecution, BlockExecution
from .schema import BlockType, BlockDefinition, EdgeDefinition, WorkflowSchema, validate_workflow_graph
from .code_executor import execute_code_block, CodeExecutionError

logger = get_logger("api.workflows.executor")


class WorkflowExecutionError(Exception):
    """Exception raised during workflow execution."""
    pass


class WorkflowExecutor:
    """
    Generic executor that interprets node graph and executes blocks.
    
    Features:
    - Topological sort for dependency resolution
    - Support for all block types (tool, code, llm, if_else, for_loop, variable)
    - Control flow handling (if/else, loops)
    - Data passing between blocks
    - Error handling and retries
    """
    
    def __init__(self, user_id: str):
        """
        Initialize workflow executor.
        
        Args:
            user_id: User ID for OAuth and tool execution
        """
        self.user_id = user_id
        self.composio_client = Composio(provider=LangchainProvider())
    
    async def execute(
        self,
        workflow: Workflow,
        input_data: Optional[Dict[str, Any]] = None,
        execution: Optional[WorkflowExecution] = None,
    ) -> Dict[str, Any]:
        """
        Execute a workflow.
        
        Args:
            workflow: Workflow model instance
            input_data: Input data for workflow
            execution: Optional execution record for logging
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Validate and parse workflow schema
            schema = validate_workflow_graph(workflow.graph_data)
            
            # Build execution graph
            blocks_dict = {block.id: block for block in schema.blocks}
            edges_dict = {edge.id: edge for edge in schema.edges}
            
            # Build adjacency lists
            incoming_edges = defaultdict(list)
            outgoing_edges = defaultdict(list)
            for edge in schema.edges:
                outgoing_edges[edge.source].append(edge)
                incoming_edges[edge.target].append(edge)
            
            # Find input blocks (blocks with no incoming edges)
            input_blocks = [block for block in schema.blocks 
                          if block.id not in incoming_edges]
            
            # Find output blocks (blocks with no outgoing edges)
            output_blocks = [block for block in schema.blocks 
                           if block.id not in outgoing_edges]
            
            # Topological sort
            execution_order = self._topological_sort(schema.blocks, schema.edges)
            
            # Execution context (shared state between blocks)
            context: Dict[str, Any] = {
                '_input': input_data or {},
                '_variables': {},
                '_outputs': {},  # block_id -> output
            }
            
            # Execute blocks in topological order
            for block_id in execution_order:
                block = blocks_dict[block_id]
                
                # Get inputs from connected blocks
                block_inputs = self._get_block_inputs(block, incoming_edges, context)
                
                # Execute block
                try:
                    output = await self._execute_block(
                        block,
                        block_inputs,
                        context,
                        execution,
                    )
                    
                    # Store output in context
                    context['_outputs'][block_id] = output
                    
                except Exception as e:
                    error_msg = f"Error executing block {block_id}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    
                    if execution:
                        await self._log_block_execution(
                            execution,
                            block,
                            block_inputs,
                            None,
                            error_msg,
                        )
                    
                    raise WorkflowExecutionError(error_msg) from e
            
            # Collect outputs from output blocks
            final_outputs = {}
            for output_block in output_blocks:
                if output_block.id in context['_outputs']:
                    final_outputs[output_block.id] = context['_outputs'][output_block.id]
            
            return {
                'output': final_outputs if final_outputs else context.get('_outputs', {}),
                'variables': context.get('_variables', {}),
                'success': True,
            }
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'output': None,
                'variables': {},
                'success': False,
                'error': error_msg,
            }
    
    def _topological_sort(
        self,
        blocks: List[BlockDefinition],
        edges: List[EdgeDefinition],
    ) -> List[str]:
        """
        Perform topological sort on workflow blocks.
        
        Args:
            blocks: List of block definitions
            edges: List of edge definitions
            
        Returns:
            List of block IDs in execution order
        """
        # Build graph
        graph = {block.id: [] for block in blocks}
        in_degree = {block.id: 0 for block in blocks}
        
        for edge in edges:
            graph[edge.source].append(edge.target)
            in_degree[edge.target] += 1
        
        # Kahn's algorithm
        queue = deque([block_id for block_id, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycles
        if len(result) != len(blocks):
            raise WorkflowExecutionError("Workflow graph contains cycles")
        
        return result
    
    def _get_block_inputs(
        self,
        block: BlockDefinition,
        incoming_edges: Dict[str, List[EdgeDefinition]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Get inputs for a block from connected blocks.
        
        Args:
            block: Block definition
            incoming_edges: Dictionary mapping block IDs to incoming edges
            context: Execution context
            
        Returns:
            Dictionary of input values
        """
        inputs = {}
        
        for edge in incoming_edges.get(block.id, []):
            source_output = context['_outputs'].get(edge.source)
            
            if source_output is not None:
                # If source has specific handle, use it; otherwise use entire output
                if edge.source_handle:
                    inputs[edge.target_handle or 'input'] = source_output.get(edge.source_handle, source_output)
                else:
                    inputs[edge.target_handle or 'input'] = source_output
        
        return inputs
    
    async def _execute_block(
        self,
        block: BlockDefinition,
        inputs: Dict[str, Any],
        context: Dict[str, Any],
        execution: Optional[WorkflowExecution] = None,
    ) -> Any:
        """
        Execute a single block.
        
        Args:
            block: Block definition
            inputs: Block inputs
            context: Execution context
            execution: Optional execution record
            
        Returns:
            Block output
        """
        start_time = datetime.utcnow()
        
        try:
            if block.type == BlockType.INPUT:
                output = context.get('_input', {})
                
            elif block.type == BlockType.OUTPUT:
                # Output block just passes through its input
                output = inputs.get('input', {})
                
            elif block.type == BlockType.VARIABLE:
                # Variable block: store or retrieve value
                var_name = block.config.get('name')
                if 'value' in inputs:
                    # Store value
                    context['_variables'][var_name] = inputs['value']
                    output = inputs['value']
                else:
                    # Retrieve value
                    output = context['_variables'].get(var_name)
                
            elif block.type == BlockType.CODE:
                # Execute Python code
                code = block.python_code or ""
                code_context = {
                    **context.get('_variables', {}),
                    **inputs,
                    '_inputs': inputs,
                }
                
                result = await execute_code_block(code, code_context)
                
                if result.get('error'):
                    raise CodeExecutionError(result['error'])
                
                # Update context variables
                context['_variables'].update(result.get('variables', {}))
                output = result.get('output')
                
            elif block.type == BlockType.TOOL:
                # Execute Composio tool
                tool_name = block.config.get('tool_name')
                tool_params = block.config.get('params', {})
                
                # Merge inputs into params
                tool_params = {**tool_params, **inputs}
                
                # Execute tool with OAuth scope if specified
                oauth_scope = block.oauth_scope
                tools = await asyncio.to_thread(
                    self.composio_client.tools.get,
                    user_id=self.user_id,
                    tools=[tool_name],
                )
                
                if not tools:
                    raise WorkflowExecutionError(f"Tool '{tool_name}' not found")
                
                tool = tools[0]
                output = await asyncio.to_thread(tool.invoke, tool_params)
                
            elif block.type == BlockType.LLM:
                # Execute LLM block
                system_prompt = block.config.get('system_prompt', '')
                user_message = inputs.get('input', '')
                model = block.config.get('model', 'gpt-5-mini')
                temperature = block.config.get('temperature', 0.2)
                
                llm = get_llm(model=model, temperature=temperature)
                
                messages = []
                if system_prompt:
                    messages.append(SystemMessage(content=system_prompt))
                messages.append(HumanMessage(content=str(user_message)))
                
                response = await llm.ainvoke(messages)
                output = response.content
                
            elif block.type == BlockType.IF_ELSE:
                # Conditional block: evaluate condition and route
                condition = block.config.get('condition', '')
                condition_code = f"_result = {condition}"
                
                # Evaluate condition in context
                eval_context = {
                    **context.get('_variables', {}),
                    **inputs,
                }
                
                result = await execute_code_block(condition_code, eval_context)
                
                if result.get('error'):
                    raise CodeExecutionError(f"Condition evaluation error: {result['error']}")
                
                condition_result = result.get('output')
                is_true = bool(condition_result)
                
                # Return routing information
                output = {
                    'condition_result': is_true,
                    'route': 'true' if is_true else 'false',
                }
                
            elif block.type == BlockType.FOR_LOOP:
                # For loop block: iterate over array
                array_var = block.config.get('array_var')
                item_var = block.config.get('item_var', 'item')
                
                # Get array from context or inputs
                array = inputs.get(array_var) or context['_variables'].get(array_var) or inputs.get('input', [])
                
                if not isinstance(array, (list, tuple)):
                    raise WorkflowExecutionError(f"For loop array '{array_var}' is not iterable")
                
                # Store loop results
                loop_results = []
                for item in array:
                    # Set item variable in context
                    context['_variables'][item_var] = item
                    loop_results.append(item)
                
                output = {
                    'items': loop_results,
                    'count': len(loop_results),
                }
                
            else:
                raise WorkflowExecutionError(f"Unsupported block type: {block.type}")
            
            # Log execution
            if execution:
                execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                await self._log_block_execution(
                    execution,
                    block,
                    inputs,
                    output,
                    None,
                    execution_time_ms,
                )
            
            return output
            
        except Exception as e:
            # Log error
            if execution:
                execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                await self._log_block_execution(
                    execution,
                    block,
                    inputs,
                    None,
                    str(e),
                    execution_time_ms,
                )
            raise
    
    async def _log_block_execution(
        self,
        execution: WorkflowExecution,
        block: BlockDefinition,
        inputs: Dict[str, Any],
        output: Any,
        error: Optional[str],
        execution_time_ms: Optional[int] = None,
    ) -> None:
        """
        Log block execution to database.
        
        Args:
            execution: Workflow execution record
            block: Block definition
            inputs: Block inputs
            output: Block output
            error: Error message if any
            execution_time_ms: Execution time in milliseconds
        """
        try:
            # Find or create block model
            workflow_block = await WorkflowBlock.get_or_none(
                workflow_id=execution.workflow_id,
                block_id=block.id,
            )
            
            if not workflow_block:
                # Create block if it doesn't exist
                workflow_block = await WorkflowBlock.create(
                    workflow_id=execution.workflow_id,
                    block_id=block.id,
                    block_type=block.type.value,
                    block_config=block.config,
                    python_code=block.python_code,
                    position_x=block.position.get('x', 0),
                    position_y=block.position.get('y', 0),
                    oauth_scope=block.oauth_scope,
                )
            
            # Create execution log
            status = 'failed' if error else 'completed'
            completed_at = datetime.utcnow() if not error else None
            
            await BlockExecution.create(
                execution_id=execution.id,
                block_id=workflow_block.id,
                status=status,
                input_data=inputs,
                output_data=output,
                error_message=error,
                execution_time_ms=execution_time_ms,
                completed_at=completed_at,
            )
            
        except Exception as e:
            logger.error(f"Failed to log block execution: {e}", exc_info=True)


__all__ = [
    "WorkflowExecutionError",
    "WorkflowExecutor",
]

