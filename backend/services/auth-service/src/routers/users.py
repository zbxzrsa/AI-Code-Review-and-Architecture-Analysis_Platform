"""
Users router - User management endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime, timezone

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
    
    Note: Returns mock data. In production, queries the auth.users table.
    """
    # Mock implementation - production would query database
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
    
    Note: Returns mock data. In production, queries auth.users by ID.
    """
    # Mock implementation - production would query database
    return UserResponse(
        id=user_id,
        email="user@example.com",
        name="Test User",
        role="user",
        created_at=datetime.now(timezone.utc),
    )


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, request: UserUpdateRequest):
    """
    Update user profile.
    
    Note: Returns mock data. In production, updates auth.users table.
    """
    # Mock implementation - production would update database
    return UserResponse(
        id=user_id,
        email="user@example.com",
        name=request.name or "Test User",
        role="user",
        avatar=request.avatar,
        created_at=datetime.now(timezone.utc),
    )


@router.delete("/{user_id}")
async def delete_user(user_id: str):
    """
    Delete user (admin only).
    
    Note: Mock implementation. In production, soft-deletes from auth.users.
    """
    # Mock implementation - production would soft-delete from database
    _ = user_id  # Used for database lookup in production
    return {"message": "User deleted successfully"}
