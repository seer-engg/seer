from typing import Any, Dict, List

from fastapi import APIRouter, Query

from src.datasets.services import get_dataset_detail, list_datasets


router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("", response_model=List[Dict[str, Any]])
async def list_datasets_endpoint(
    limit: int = Query(50, ge=1, le=200),
) -> List[Dict[str, Any]]:
    """List datasets from Langfuse."""
    return await list_datasets(limit=limit)


@router.get("/{dataset_id}", response_model=Dict[str, Any])
async def get_dataset_detail_endpoint(
    dataset_id: str,
    example_limit: int = Query(10, ge=1, le=100),
    experiment_limit: int = Query(10, ge=1, le=100),
) -> Dict[str, Any]:
    """Get dataset detail with examples and experiments."""
    return await get_dataset_detail(
        dataset_id=dataset_id,
        example_limit=example_limit,
        experiment_limit=experiment_limit,
    )


__all__ = ["router"]


