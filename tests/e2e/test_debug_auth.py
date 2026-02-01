"""Debug test to check auth and routing."""
import pytest
from httpx import AsyncClient


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_debug_routes(db_engine, authenticated_e2e_client: AsyncClient):
    """Debug: Check what routes are available."""
    # Try hitting different endpoints with correct paths
    response = await authenticated_e2e_client.get("/api/v1/workflows")
    print(f"\nGET /api/v1/workflows: {response.status_code}")
    print(f"Response: {response.text}")

    response2 = await authenticated_e2e_client.get("/api/v1/triggers")
    print(f"\nGET /api/v1/triggers: {response2.status_code}")
    print(f"Response: {response2.text}")

    # Try a known endpoint
    response3 = await authenticated_e2e_client.get("/api/integrations/")
    print(f"\nGET /api/integrations/: {response3.status_code}")
    print(f"Response: {response3.text[:200]}")
