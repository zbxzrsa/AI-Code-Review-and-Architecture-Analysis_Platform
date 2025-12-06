"""
User Models API - Endpoints for Users to Manage Custom AI Models

Users can:
- Add their own AI models
- Configure model preferences
- Switch between available models
- View usage statistics

Note: Users can ONLY access Code AI, not Version Control AI
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

router = APIRouter(prefix="/api/v2/models", tags=["User Models"])


# Request/Response Models

class RegisterModelRequest(BaseModel):
    """Request to register a custom model"""
    provider: str = Field(..., description="Model provider (openai, anthropic, custom, etc.)")
    model_name: str = Field(..., description="Model identifier")
    api_endpoint: str = Field(..., description="API endpoint URL")
    api_key: str = Field(..., description="API key for authentication")
    display_name: str = Field(..., description="Display name for the model")
    description: str = Field("", description="Model description")
    max_tokens: int = Field(4096, ge=1, le=128000)
    temperature: float = Field(0.7, ge=0, le=2)
    supports_streaming: bool = Field(True)
    supported_languages: List[str] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "provider": "openai",
                "model_name": "gpt-4-turbo",
                "api_endpoint": "https://api.openai.com/v1",
                "api_key": "sk-...",
                "display_name": "My GPT-4 Turbo",
                "description": "Custom GPT-4 configuration",
                "max_tokens": 4096,
                "temperature": 0.7,
                "supports_streaming": True,
                "supported_languages": ["python", "javascript", "typescript"]
            }
        }


class UpdateModelRequest(BaseModel):
    """Request to update a model"""
    display_name: Optional[str] = None
    description: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    api_key: Optional[str] = None
    supports_streaming: Optional[bool] = None


class SetPreferenceRequest(BaseModel):
    """Request to set model preference"""
    default_model_id: str
    fallback_model_id: Optional[str] = None


class LanguagePreferenceRequest(BaseModel):
    """Request to set language-specific preference"""
    language: str
    model_id: str


class AnalysisPreferenceRequest(BaseModel):
    """Request to set analysis-type preference"""
    analysis_type: str
    model_id: str


class ModelResponse(BaseModel):
    """Model information response"""
    model_id: str
    provider: str
    model_name: str
    display_name: str
    description: str
    status: str
    is_builtin: bool
    supports_streaming: bool
    created_at: Optional[str] = None


class UsageStatsResponse(BaseModel):
    """Usage statistics response"""
    total_models: int
    active_models: int
    total_requests: int
    total_tokens: int
    error_rate: float


# Endpoints

@router.get("/available", response_model=List[ModelResponse])
async def get_available_models():
    """
    Get all available models for the current user
    
    Returns:
    - Built-in models (OpenAI, Anthropic, Google)
    - User's custom registered models
    """
    # Placeholder - in production, get from model_registry
    return [
        ModelResponse(
            model_id="openai-gpt4",
            provider="openai",
            model_name="gpt-4-turbo-preview",
            display_name="GPT-4 Turbo",
            description="OpenAI's most capable model",
            status="active",
            is_builtin=True,
            supports_streaming=True
        ),
        ModelResponse(
            model_id="anthropic-claude3",
            provider="anthropic",
            model_name="claude-3-5-sonnet-20241022",
            display_name="Claude 3.5 Sonnet",
            description="Anthropic's latest model",
            status="active",
            is_builtin=True,
            supports_streaming=True
        ),
        ModelResponse(
            model_id="google-gemini",
            provider="google",
            model_name="gemini-pro",
            display_name="Gemini Pro",
            description="Google's Gemini Pro",
            status="active",
            is_builtin=True,
            supports_streaming=True
        )
    ]


@router.post("/register", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def register_model(request: RegisterModelRequest):
    """
    Register a new custom AI model
    
    The model will be validated before activation.
    Users can only register Code AI models (not Version Control AI).
    """
    # Placeholder - in production, use model_registry.register_model()
    return ModelResponse(
        model_id=f"user_{request.model_name[:8]}",
        provider=request.provider,
        model_name=request.model_name,
        display_name=request.display_name,
        description=request.description,
        status="pending",
        is_builtin=False,
        supports_streaming=request.supports_streaming,
        created_at=datetime.now().isoformat()
    )


@router.get("/my-models", response_model=List[ModelResponse])
async def get_my_models():
    """
    Get all models registered by the current user
    """
    # Placeholder
    return []


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str):
    """
    Get details of a specific model
    """
    # Placeholder - check if model exists and user has access
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Model {model_id} not found"
    )


@router.put("/{model_id}", response_model=ModelResponse)
async def update_model(model_id: str, request: UpdateModelRequest):
    """
    Update a custom model configuration
    
    Only the owner can update their models.
    Updates will require re-validation.
    """
    # Placeholder
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Model {model_id} not found"
    )


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(model_id: str):
    """
    Delete a custom model
    
    Only the owner can delete their models.
    Built-in models cannot be deleted.
    """
    # Placeholder
    pass


@router.post("/{model_id}/validate")
async def validate_model(model_id: str):
    """
    Trigger validation for a model
    
    Tests the API connection and validates the model is working.
    """
    return {
        "model_id": model_id,
        "validation_status": "pending",
        "message": "Validation started"
    }


# Preferences

@router.get("/preferences/current")
async def get_preferences():
    """
    Get current user's model preferences
    """
    return {
        "default_model_id": "anthropic-claude3",
        "fallback_model_id": "openai-gpt4",
        "preferences_by_language": {},
        "preferences_by_analysis_type": {}
    }


@router.post("/preferences/default")
async def set_default_preference(request: SetPreferenceRequest):
    """
    Set default model preference
    """
    return {
        "status": "updated",
        "default_model_id": request.default_model_id,
        "fallback_model_id": request.fallback_model_id
    }


@router.post("/preferences/language")
async def set_language_preference(request: LanguagePreferenceRequest):
    """
    Set preferred model for a specific programming language
    """
    return {
        "status": "updated",
        "language": request.language,
        "model_id": request.model_id
    }


@router.post("/preferences/analysis")
async def set_analysis_preference(request: AnalysisPreferenceRequest):
    """
    Set preferred model for a specific analysis type
    """
    return {
        "status": "updated",
        "analysis_type": request.analysis_type,
        "model_id": request.model_id
    }


# Usage Statistics

@router.get("/usage/stats", response_model=UsageStatsResponse)
async def get_usage_stats():
    """
    Get usage statistics for user's models
    """
    return UsageStatsResponse(
        total_models=0,
        active_models=0,
        total_requests=0,
        total_tokens=0,
        error_rate=0.0
    )


@router.get("/usage/history")
async def get_usage_history(days: int = 30):
    """
    Get usage history for the past N days
    """
    return {
        "period_days": days,
        "daily_usage": [],
        "total_requests": 0,
        "total_tokens": 0
    }
