from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseProvider(ABC):

    @property
    def persistent_resource(self) -> Dict[str, Any]:
        pass

    async def provision_resources(self, seed:str, user_id:str) -> Dict[str, Any]:
        raise NotImplementedError("Provisioning resources is not implemented for this provider")

    async def cleanup_resources(self, resources: Dict[str, Any], user_id:str) -> None:
        raise NotImplementedError("Cleaning up resources is not implemented for this provider")

