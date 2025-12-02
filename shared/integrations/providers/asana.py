from .base import BaseProvider
from typing import Dict, Any
from shared.config import config
from shared.logger import get_logger
from shared.tools import ComposioMCPClient
import asyncio


logger = get_logger("shared.integrations.providers.asana")

class AsanaProvider(BaseProvider):
    def __init__(self):
        self.mcp_client = ComposioMCPClient(service_names=["ASANA"], user_id=config.user_id).get_client()
        self.version = "20251111_00"

    @property
    def persistent_resource(self) -> Dict[str, Any]:
        return {
            "workspace_id": config.asana_workspace_id,
            "team_id":config.asana_team_gid
        }

    async def provision_resources(self, seed:str) -> Dict[str, Any]:
        
        logger.info(f"Provisioning ASANA resources with seed {seed}")
        payload = {
            "data": {
                "name": seed,
                "workspace_id": config.asana_workspace_id,
                "team":config.asana_team_gid
            },
            "opt_pretty": True,
        }
        result = await asyncio.to_thread(self.mcp_client.tools.execute,
            "ASANA_CREATE_A_PROJECT",
            user_id=config.composio_user_id,
            arguments=payload,
            version=self.version
        )
        project_gid = result.get('data').get('data').get('gid')
        logger.info(f"Project GID: {project_gid}")
        return {
            "project_gid": project_gid,
        }

    async def cleanup_resources(self, resources: Dict[str, Any]) -> None:

        project_gid = resources.get('project_gid')
        logger.info(f"Cleaning up ASANA resources with project id ")
        payload = {
            "project_gid": project_gid,
        }
        result = await asyncio.to_thread(self.mcp_client.tools.execute,
            "ASANA_DELETE_PROJECT",
            user_id=config.composio_user_id,
            arguments=payload,
            version=self.version
        )
        return result
        