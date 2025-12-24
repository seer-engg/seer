from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Response, status, Request

from api.projects import services
from api.projects.models import (
    ProjectCreate,
    ProjectListResponse,
    ProjectPublic,
    ProjectUpdate,
)


router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("", response_model=ProjectListResponse)
async def list_projects(request: Request) -> ProjectListResponse:
    """List all available projects."""
    projects = await services.list_projects()
    return ProjectListResponse(
        projects=[
            ProjectPublic.model_validate(project, from_attributes=True)
            for project in projects
        ],
        user=request.state.db_user,
    )


@router.get("/{project_id}", response_model=ProjectPublic)
async def get_project(project_id: UUID) -> ProjectPublic:
    """Fetch a single project."""
    project = await services.get_project(project_id)
    return ProjectPublic.model_validate(project, from_attributes=True)


@router.post("", response_model=ProjectPublic, status_code=status.HTTP_201_CREATED)
async def create_project(payload: ProjectCreate) -> ProjectPublic:
    """Create a new project."""
    project = await services.create_project(payload)
    return ProjectPublic.model_validate(project, from_attributes=True)


@router.put("/{project_id}", response_model=ProjectPublic)
async def update_project(project_id: UUID, payload: ProjectUpdate) -> ProjectPublic:
    """Update an existing project."""
    project = await services.update_project(project_id, payload)
    return ProjectPublic.model_validate(project, from_attributes=True)


@router.delete(
    "/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def delete_project(project_id: UUID) -> Response:
    """Delete a project."""
    await services.delete_project(project_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = ["router"]


