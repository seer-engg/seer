from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import HTTPException, status
from tortoise.exceptions import DoesNotExist, IntegrityError

from api.projects.models import Project, ProjectCreate, ProjectUpdate


async def list_projects() -> List[Project]:
    """Fetch all projects ordered by name."""
    return await Project.all().order_by("project_name")


async def get_project(project_id: UUID) -> Project:
    """Fetch a single project or raise 404."""
    try:
        return await Project.get(id=project_id)
    except DoesNotExist as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        ) from exc


async def create_project(payload: ProjectCreate) -> Project:
    """Persist a new project."""
    try:
        project = await Project.create(**payload.model_dump())
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project name already exists",
        ) from exc
    return project


async def update_project(project_id: UUID, payload: ProjectUpdate) -> Project:
    """Update an existing project."""
    project = await get_project(project_id)
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        return project
    for field, value in updates.items():
        setattr(project, field, value)
    try:
        await project.save()
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project name already exists",
        ) from exc
    await project.refresh_from_db()
    return project


async def delete_project(project_id: UUID) -> None:
    """Delete a project."""
    project = await get_project(project_id)
    await project.delete()


__all__ = [
    "create_project",
    "delete_project",
    "get_project",
    "list_projects",
    "update_project",
]
