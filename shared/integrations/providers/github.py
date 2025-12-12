from .base import BaseProvider
from typing import Dict, Any
from shared.logger import get_logger

logger = get_logger("shared.integrations.providers.github")

class GithubProvider(BaseProvider):
    

    @property
    def persistent_resource(self) -> Dict[str, Any]:
        return {
            "github organization": "seer-engg",
            "testing repo": "label-edgecase-repo",
            # TODO: add default user
        }


    async def provision_resources(self, seed: str, user_id:str) -> Dict[str, Any]:
        logger.warning("Provisioning GitHub resources is not implemented")
        pass

    async def cleanup_resources(self, resources: Dict[str, Any], user_id:str) -> None:
        logger.warning("Cleaning up GitHub resources is not implemented")
        pass