from langsmith.schemas import Example


import os
import asyncio
from langsmith import evaluate, Client
from agents.reflexion.graph import graph as reflexion_graph
from dotenv import load_dotenv
load_dotenv()
from evals.judge import correctness_evaluator
from langchain_core.messages import HumanMessage


client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
datasets = [
   ("open-critic-gpt-100", ["base"]),
]

graphs = {
    "reflexion": reflexion_graph,
}

for graph_name, graph in graphs.items():
    # Synchronous wrapper for async graph invocation
    def sync_graph_invoke(inputs, _graph=graph):
        """Invoke compiled langgraph graph synchronously for evaluation purposes."""
        # Transform LangSmith example format to LangChain message format
        # LangSmith format: {'Human': 'text'} or similar
        # LangChain format: [HumanMessage(content='text')]
        
        if isinstance(inputs, dict):
            # Extract the human message from the input dict
            if 'Human' in inputs:
                message_content = inputs['Human']
            elif 'input' in inputs:
                message_content = inputs['input']
            else:
                # Fallback: take the first value
                message_content = next(iter(inputs.values()))
            
            messages = [HumanMessage(content=message_content)]
        elif isinstance(inputs, str):
            messages = [HumanMessage(content=inputs)]
        else:
            messages = inputs
        
        return asyncio.run(_graph.ainvoke({'messages': messages}))

    for dataset_name, splits in datasets:
        print(f"Evaluating {dataset_name} with splits {splits}")
        results = evaluate(
            sync_graph_invoke,
            data=list[Example](client.list_examples(dataset_name=dataset_name, splits=splits))[:5],
            evaluators=[correctness_evaluator], ## Auto eval will be fired in server side , no need to pass evaluators here
            experiment_prefix=f"lokesh-",
            client=client,
            num_repetitions=1,
            max_concurrency=1,  ### only use upto 5 concurrent requests , greater than 5 will cause connection issues in execute action plan
        )