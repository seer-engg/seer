from agents.codex.graphs.planner.graph import graph as planner_graph
import asyncio
from agents.codex.common.state import PlannerState


async def test_planner():
    state = PlannerState(
        request="Add a new feature to the project",
        repo_path="https://github.com/user/repo.git",
        messages=[],
        taskPlan={},
    )
    result = await planner_graph.nodes['raise-pr'].ainvoke(state)
    print(result)


if __name__ == "__main__":
    asyncio.run(test_planner())