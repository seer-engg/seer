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

    async def provision_resources(self, seed:str, user_id:str) -> Dict[str, Any]:
        
        logger.info(f"Provisioning ASANA resources with seed {seed}")
        
        # For free plans without Teams, reuse existing project if configured
        if config.asana_project_id:
            logger.info(f"Reusing existing Asana project: {config.asana_project_id} (free plan - no teams)")
            return {
                "project_gid": config.asana_project_id,
            }
        
        # Try to create a new project (requires Teams on paid plans)
        logger.info("Attempting to create new Asana project...")
        team_gid = config.asana_team_gid
        
        # Asana API requires 'workspace' or 'team' (not 'workspace_id' or 'team_id')
        # Note: Free plans don't have Teams, so project creation will likely fail
        data_payload = {"name": seed}
        if team_gid:
            data_payload["team"] = team_gid
        elif config.asana_workspace_id:
            data_payload["workspace"] = config.asana_workspace_id
        else:
            raise ValueError("Either asana_workspace_id, asana_team_gid, or asana_project_id must be configured")
        
        payload = {
            "data": data_payload,
            "opt_pretty": True,
        }
        result = await asyncio.to_thread(self.mcp_client.tools.execute,
            "ASANA_CREATE_A_PROJECT",
            user_id=user_id,
            arguments=payload,
            version=self.version
        )
        # Handle different response structures
        if result and result.get('data'):
            data = result.get('data')
            if isinstance(data, dict):
                project_gid = data.get('data', {}).get('gid') or data.get('gid')
            else:
                project_gid = None
        else:
            logger.error(f"Asana project creation failed: {result}")
            project_gid = None
        
        if not project_gid:
            raise ValueError(f"Failed to create Asana project. Response: {result}. For free plans, set ASANA_PROJECT_ID in .env to reuse an existing project.")
        
        logger.info(f"Project GID: {project_gid}")
        return {
            "project_gid": project_gid,
        }

    async def cleanup_resources(self, resources: Dict[str, Any], user_id:str) -> None:

        project_gid = resources.get('project_gid')
        logger.info(f"Cleaning up ASANA resources with project id ")
        payload = {
            "project_gid": project_gid,
        }
        result = await asyncio.to_thread(self.mcp_client.tools.execute,
            "ASANA_DELETE_PROJECT",
            user_id=user_id,
            arguments=payload,
            version=self.version
        )
        return result
        