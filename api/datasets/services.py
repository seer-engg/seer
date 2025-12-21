from typing import Any, Dict, List

import httpx
from fastapi import HTTPException

from src.core.config import LANGFUSE_BASE_URL, get_auth_headers


async def list_datasets(limit: int) -> List[Dict[str, Any]]:
    """List datasets from Langfuse."""
    headers = get_auth_headers()

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{LANGFUSE_BASE_URL}/api/public/datasets",
                headers=headers,
                params={"limit": limit, "page": 1},
                timeout=30.0,
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error fetching datasets: {response.text[:500]}",
            )

        data = response.json()
        datasets = data.get("datasets", [])

        # Format datasets
        formatted_datasets: List[Dict[str, Any]] = []
        for ds in datasets:
            formatted_datasets.append(
                {
                    "id": ds.get("id"),
                    "name": ds.get("name"),
                    "description": ds.get("description"),
                    "data_type": ds.get("dataType"),
                    "created_at": ds.get("createdAt"),
                    "modified_at": ds.get("updatedAt"),
                    "example_count": ds.get("exampleCount"),
                    "session_count": ds.get("sessionCount"),
                    "last_session_start_time": ds.get("lastSessionStartTime"),
                }
            )

        return formatted_datasets

    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching datasets: {str(exc)}",
        ) from exc


async def get_dataset_detail(
    dataset_id: str,
    example_limit: int,
    experiment_limit: int,
) -> Dict[str, Any]:
    """Get dataset detail with examples and experiments."""
    headers = get_auth_headers()

    try:
        async with httpx.AsyncClient() as client:
            # Fetch dataset
            response = await client.get(
                f"{LANGFUSE_BASE_URL}/api/public/datasets/{dataset_id}",
                headers=headers,
                timeout=30.0,
            )

            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Dataset not found")

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Error fetching dataset: {response.text[:500]}",
                )

            dataset_data = response.json()

            # Fetch examples
            examples_response = await client.get(
                f"{LANGFUSE_BASE_URL}/api/public/datasets/{dataset_id}/examples",
                headers=headers,
                params={"limit": example_limit, "page": 1},
                timeout=30.0,
            )

            examples: List[Dict[str, Any]] = []
            if examples_response.status_code == 200:
                examples_data = examples_response.json()
                examples = examples_data.get("examples", [])

            # Fetch experiments
            experiments_response = await client.get(
                f"{LANGFUSE_BASE_URL}/api/public/datasets/{dataset_id}/experiments",
                headers=headers,
                params={"limit": experiment_limit, "page": 1},
                timeout=30.0,
            )

            experiments: List[Dict[str, Any]] = []
            if experiments_response.status_code == 200:
                experiments_data = experiments_response.json()
                experiments = experiments_data.get("experiments", [])

        return {
            "id": dataset_data.get("id"),
            "name": dataset_data.get("name"),
            "description": dataset_data.get("description"),
            "data_type": dataset_data.get("dataType"),
            "created_at": dataset_data.get("createdAt"),
            "modified_at": dataset_data.get("updatedAt"),
            "example_count": dataset_data.get("exampleCount"),
            "session_count": dataset_data.get("sessionCount"),
            "last_session_start_time": dataset_data.get("lastSessionStartTime"),
            "inputs_schema": dataset_data.get("inputsSchema"),
            "outputs_schema": dataset_data.get("outputsSchema"),
            "transformations": dataset_data.get("transformations", []),
            "examples": examples,
            "experiments": experiments,
        }

    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching dataset: {str(exc)}",
        ) from exc


__all__ = ["list_datasets", "get_dataset_detail"]


