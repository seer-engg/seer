"""Dev-only API endpoints. Only active when config.enable_user_emulation=True."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from tortoise.expressions import Q

from seer.config import config
from seer.database import User

router = APIRouter(prefix="/dev", tags=["dev"])


class UserSummary(BaseModel):
    user_id: str
    email: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]


def _require_emulation_enabled() -> None:
    if not config.enable_user_emulation:
        raise HTTPException(status_code=404, detail="Not found")


@router.get("/users/search", response_model=List[UserSummary])
async def search_users(
    q: str = Query(default="", max_length=200),
) -> List[UserSummary]:
    """Search users by email, name, or user_id. Only available when enable_user_emulation=True."""
    _require_emulation_enabled()

    q = q.strip()
    if not q:
        users = await User.all().order_by("email").limit(20)
    else:
        users = await User.filter(
            Q(email__icontains=q)
            | Q(first_name__icontains=q)
            | Q(last_name__icontains=q)
            | Q(user_id__icontains=q)
        ).order_by("email").limit(20)

    return [
        UserSummary(
            user_id=u.user_id,
            email=u.email,
            first_name=u.first_name,
            last_name=u.last_name,
        )
        for u in users
    ]
