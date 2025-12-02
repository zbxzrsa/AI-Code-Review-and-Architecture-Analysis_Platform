"""
Users router - User management endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

router = APIRouter()


class UserResponse(BaseModel):
    """User response model."""
    id: str
    email: str
    name: str
    role: str
    avatar: Optional[str] = None
    created_at: datetime
    last_login_at: Optional[datetime] = None


class UserUpdateRequest(BaseModel):
    """User update request model."""
    name: Optional[str] = None
    avatar: Optional[str] = None


class UserListResponse(BaseModel):
    """User list response model."""
    items: List[UserResponse]
    total: int
    page: int
    limit: int


@router.get("/", response_model=UserListResponse)
async def list_users(page: int = 1, limit: int = 20):
    """
    List all users (admin only).
    """
    # TODO: Implement with actual database
    return UserListResponse(
        items=[],
        total=0,
        page=page,
        limit=limit,
    )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    """
    Get user by ID.
    """
    # TODO: Implement with actual database
    return UserResponse(
        id=user_id,
        email="user@example.com",
        name="Test User",
        role="user",
        created_at=datetime.utcnow(),
    )


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, request: UserUpdateRequest):
    """
    Update user profile.
    """
    # TODO: Implement with actual database
    return UserResponse(
        id=user_id,
        email="user@example.com",
        name=request.name or "Test User",
        role="user",
        avatar=request.avatar,
        created_at=datetime.utcnow(),
    )


@router.delete("/{user_id}")
async def delete_user(user_id: str):
    """
    Delete user (admin only).
    """
    # TODO: Implement with actual database
    return {"message": "User deleted successfully"}
