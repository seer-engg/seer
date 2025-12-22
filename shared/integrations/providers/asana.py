from .base import BaseProvider
from typing import Dict, Any
from shared.config import config
from shared.logger import get_logger
from shared.tools.executor import get_oauth_token
from shared.database.models import User
import httpx
import asyncio


logger = get_logger("shared.integrations.providers.asana")

class AsanaProvider(BaseProvider):
    def __init__(self):
        self.api_base_url = "https://app.asana.com/api/1.0"

    @property
    def persistent_resource(self) -> Dict[str, Any]:
        return {
            "workspace_id": config.asana_workspace_id,
            "team_id":config.asana_team_gid
        }

    async def provision_resources(self, seed:str, user_id:str) -> Dict[str, Any]:
        """
        Provision Asana resources using direct API calls.
        
        Args:
            seed: Project name seed
            user_id: User ID for OAuth token lookup
        
        Returns:
            Dict with project_gid
        """
        logger.info(f"Provisioning ASANA resources with seed {seed}")
        
        # For free plans without Teams, reuse existing project if configured
        if config.asana_project_id:
            logger.info(f"Reusing existing Asana project: {config.asana_project_id} (free plan - no teams)")
            return {
                "project_gid": config.asana_project_id,
            }
        
        # Get OAuth token for Asana
        try:
            connection, access_token = await get_oauth_token(user_id=user_id, provider="asana")
        except Exception as e:
            raise ValueError(f"No active Asana OAuth connection found for user {user_id}: {e}")
        
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
        
        # Make direct API call to Asana
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base_url}/projects",
                    headers=headers,
                    json={"data": data_payload},
                    params={"opt_pretty": "true"}
                )
                response.raise_for_status()
                result = response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Asana API error: {e.response.status_code} - {e.response.text}")
            raise ValueError(f"Failed to create Asana project: {e.response.text[:200]}")
        except Exception as e:
            logger.exception(f"Unexpected error creating Asana project: {e}")
            raise ValueError(f"Failed to create Asana project: {str(e)}")
        
        # Handle response structure
        if result and result.get('data'):
            data = result.get('data')
            if isinstance(data, dict):
                project_gid = data.get('gid')
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
        """
        Cleanup Asana resources using direct API calls.
        
        Args:
            resources: Dict containing project_gid
            user_id: User ID for OAuth token lookup
        """
        project_gid = resources.get('project_gid')
        if not project_gid:
            logger.warning("No project_gid provided for cleanup")
            return
        
        logger.info(f"Cleaning up ASANA resources with project id {project_gid}")
        
        # Get OAuth token for Asana
        try:
            connection, access_token = await get_oauth_token(user_id=user_id, provider="asana")
        except Exception as e:
            logger.error(f"No active Asana OAuth connection found for user {user_id}: {e}")
            return
        
        # Make direct API call to delete project
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.api_base_url}/projects/{project_gid}",
                    headers=headers,
                    params={"opt_pretty": "true"}
                )
                response.raise_for_status()
                logger.info(f"Successfully deleted Asana project {project_gid}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Asana API error deleting project: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            logger.exception(f"Unexpected error deleting Asana project: {e}")
        