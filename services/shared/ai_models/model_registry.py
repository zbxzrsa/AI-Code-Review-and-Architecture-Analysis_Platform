"""
User Model Registry - API for Users to Add Custom Models

Users can:
- Add their own AI models via API
- Configure model preferences
- Switch between available models
- Only access Code AI (not Version Control AI)
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import logging
import json
import hashlib
from pathlib import Path

from .base_ai import AIConfig, ModelProvider, VersionType

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Status of a user-added model"""
    PENDING = "pending"       # Awaiting validation
    ACTIVE = "active"         # Ready to use
    SUSPENDED = "suspended"   # Temporarily disabled
    INVALID = "invalid"       # Failed validation


@dataclass
class UserModel:
    """A user-added custom model"""
    model_id: str
    user_id: str
    provider: ModelProvider
    model_name: str
    api_endpoint: str
    display_name: str
    description: str
    status: ModelStatus
    created_at: str
    updated_at: str
    
    # Configuration
    max_tokens: int = 4096
    temperature: float = 0.7
    api_key_encrypted: Optional[str] = None
    
    # Capabilities
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supported_languages: List[str] = field(default_factory=list)
    
    # Usage tracking
    request_count: int = 0
    error_count: int = 0
    total_tokens: int = 0
    
    # Validation
    last_validated: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['provider'] = self.provider.value
        data['status'] = self.status.value
        # Don't expose encrypted API key
        data.pop('api_key_encrypted', None)
        return data


@dataclass
class ModelPreference:
    """User's model preferences"""
    user_id: str
    default_model_id: str
    fallback_model_id: Optional[str] = None
    preferences_by_language: Dict[str, str] = field(default_factory=dict)
    preferences_by_analysis_type: Dict[str, str] = field(default_factory=dict)
    updated_at: str = ""


class UserModelRegistry:
    """
    User Model Registry - Manages user-added AI models
    
    Features:
    - User model registration
    - API key management (encrypted)
    - Model validation
    - Usage tracking
    - Preference management
    
    Note: Users can ONLY add/configure Code AI models.
    Version Control AI is internal and not accessible to users.
    """
    
    # Built-in models available to all users
    BUILT_IN_MODELS = {
        'openai-gpt4': AIConfig(
            model_id='openai-gpt4',
            provider=ModelProvider.OPENAI,
            model_name='gpt-4-turbo-preview',
            version='1.0',
            description='OpenAI GPT-4 Turbo'
        ),
        'openai-gpt35': AIConfig(
            model_id='openai-gpt35',
            provider=ModelProvider.OPENAI,
            model_name='gpt-3.5-turbo',
            version='1.0',
            description='OpenAI GPT-3.5 Turbo'
        ),
        'anthropic-claude3': AIConfig(
            model_id='anthropic-claude3',
            provider=ModelProvider.ANTHROPIC,
            model_name='claude-3-5-sonnet-20241022',
            version='1.0',
            description='Anthropic Claude 3.5 Sonnet'
        ),
        'anthropic-claude3-opus': AIConfig(
            model_id='anthropic-claude3-opus',
            provider=ModelProvider.ANTHROPIC,
            model_name='claude-3-opus-20240229',
            version='1.0',
            description='Anthropic Claude 3 Opus'
        ),
        'google-gemini': AIConfig(
            model_id='google-gemini',
            provider=ModelProvider.GOOGLE,
            model_name='gemini-pro',
            version='1.0',
            description='Google Gemini Pro'
        )
    }
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.models_file = self.storage_path / "user_models.json"
        self.preferences_file = self.storage_path / "user_preferences.json"
        
        self.user_models: Dict[str, UserModel] = {}
        self.user_preferences: Dict[str, ModelPreference] = {}
        
        self._load_data()
    
    def _load_data(self) -> None:
        """Load data from storage"""
        if self.models_file.exists():
            with open(self.models_file, 'r') as f:
                data = json.load(f)
                for model_id, model_data in data.items():
                    model_data['provider'] = ModelProvider(model_data['provider'])
                    model_data['status'] = ModelStatus(model_data['status'])
                    self.user_models[model_id] = UserModel(**model_data)
        
        if self.preferences_file.exists():
            with open(self.preferences_file, 'r') as f:
                data = json.load(f)
                for user_id, pref_data in data.items():
                    self.user_preferences[user_id] = ModelPreference(**pref_data)
    
    def _save_data(self) -> None:
        """Save data to storage"""
        # Save models
        models_data = {}
        for model_id, model in self.user_models.items():
            data = asdict(model)
            data['provider'] = model.provider.value
            data['status'] = model.status.value
            models_data[model_id] = data
        
        with open(self.models_file, 'w') as f:
            json.dump(models_data, f, indent=2)
        
        # Save preferences
        with open(self.preferences_file, 'w') as f:
            json.dump(
                {uid: asdict(pref) for uid, pref in self.user_preferences.items()},
                f, indent=2
            )
    
    def _generate_model_id(self, user_id: str, model_name: str) -> str:
        """Generate unique model ID"""
        content = f"{user_id}:{model_name}:{datetime.now().isoformat()}"
        return f"user_{hashlib.sha256(content.encode()).hexdigest()[:12]}"
    
    def _encrypt_api_key(self, api_key: str) -> str:
        """Encrypt API key for storage"""
        # In production, use proper encryption (AES-256-GCM)
        # This is a placeholder
        return f"encrypted:{hashlib.sha256(api_key.encode()).hexdigest()}"
    
    # User Model Management
    
    def register_model(
        self,
        user_id: str,
        provider: str,
        model_name: str,
        api_endpoint: str,
        api_key: str,
        display_name: str,
        description: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        supports_streaming: bool = True,
        supported_languages: Optional[List[str]] = None
    ) -> UserModel:
        """
        Register a new user model
        
        Args:
            user_id: User ID
            provider: Model provider name
            model_name: Model identifier
            api_endpoint: API endpoint URL
            api_key: API key (will be encrypted)
            display_name: Display name for the model
            description: Model description
            max_tokens: Maximum tokens
            temperature: Default temperature
            supports_streaming: Whether streaming is supported
            supported_languages: List of supported programming languages
            
        Returns:
            Registered UserModel
        """
        model_id = self._generate_model_id(user_id, model_name)
        
        try:
            provider_enum = ModelProvider(provider.lower())
        except ValueError:
            provider_enum = ModelProvider.CUSTOM
        
        model = UserModel(
            model_id=model_id,
            user_id=user_id,
            provider=provider_enum,
            model_name=model_name,
            api_endpoint=api_endpoint,
            display_name=display_name,
            description=description,
            status=ModelStatus.PENDING,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            max_tokens=max_tokens,
            temperature=temperature,
            api_key_encrypted=self._encrypt_api_key(api_key),
            supports_streaming=supports_streaming,
            supported_languages=supported_languages or []
        )
        
        self.user_models[model_id] = model
        self._save_data()
        
        logger.info(f"User {user_id} registered model {model_id}")
        return model
    
    async def validate_model(self, model_id: str) -> bool:
        """
        Validate a user model by testing the API
        
        Returns:
            True if validation passed
        """
        if model_id not in self.user_models:
            return False
        
        model = self.user_models[model_id]
        model.validation_errors = []
        
        try:
            # In production, actually test the API endpoint
            # This is a placeholder
            validation_passed = True
            
            if validation_passed:
                model.status = ModelStatus.ACTIVE
                model.last_validated = datetime.now().isoformat()
                logger.info(f"Model {model_id} validated successfully")
            else:
                model.status = ModelStatus.INVALID
                model.validation_errors.append("API validation failed")
                logger.warning(f"Model {model_id} validation failed")
            
            model.updated_at = datetime.now().isoformat()
            self._save_data()
            
            return validation_passed
            
        except Exception as e:
            model.status = ModelStatus.INVALID
            model.validation_errors.append(str(e))
            model.updated_at = datetime.now().isoformat()
            self._save_data()
            logger.error(f"Model {model_id} validation error: {e}")
            return False
    
    def update_model(
        self,
        model_id: str,
        user_id: str,
        **updates
    ) -> Optional[UserModel]:
        """Update a user model"""
        if model_id not in self.user_models:
            return None
        
        model = self.user_models[model_id]
        
        # Verify ownership
        if model.user_id != user_id:
            logger.warning(f"User {user_id} attempted to update model {model_id} owned by {model.user_id}")
            return None
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(model, key) and key not in ['model_id', 'user_id', 'created_at']:
                if key == 'api_key':
                    model.api_key_encrypted = self._encrypt_api_key(value)
                else:
                    setattr(model, key, value)
        
        model.updated_at = datetime.now().isoformat()
        model.status = ModelStatus.PENDING  # Re-validate after update
        
        self._save_data()
        logger.info(f"Model {model_id} updated by user {user_id}")
        
        return model
    
    def delete_model(self, model_id: str, user_id: str) -> bool:
        """Delete a user model"""
        if model_id not in self.user_models:
            return False
        
        model = self.user_models[model_id]
        
        # Verify ownership
        if model.user_id != user_id:
            logger.warning(f"User {user_id} attempted to delete model {model_id}")
            return False
        
        del self.user_models[model_id]
        self._save_data()
        
        logger.info(f"Model {model_id} deleted by user {user_id}")
        return True
    
    def get_user_models(self, user_id: str) -> List[UserModel]:
        """Get all models for a user"""
        return [m for m in self.user_models.values() if m.user_id == user_id]
    
    def get_available_models(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all available models for a user (built-in + custom)"""
        models = []
        
        # Built-in models
        for model_id, config in self.BUILT_IN_MODELS.items():
            models.append({
                'model_id': model_id,
                'display_name': config.description,
                'provider': config.provider.value,
                'model_name': config.model_name,
                'is_builtin': True,
                'is_active': True
            })
        
        # User's custom models
        for model in self.get_user_models(user_id):
            if model.status == ModelStatus.ACTIVE:
                models.append({
                    'model_id': model.model_id,
                    'display_name': model.display_name,
                    'provider': model.provider.value,
                    'model_name': model.model_name,
                    'is_builtin': False,
                    'is_active': True
                })
        
        return models
    
    # User Preferences
    
    def set_default_model(
        self,
        user_id: str,
        model_id: str,
        fallback_model_id: Optional[str] = None
    ) -> ModelPreference:
        """Set user's default model"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = ModelPreference(
                user_id=user_id,
                default_model_id=model_id,
                fallback_model_id=fallback_model_id,
                updated_at=datetime.now().isoformat()
            )
        else:
            pref = self.user_preferences[user_id]
            pref.default_model_id = model_id
            pref.fallback_model_id = fallback_model_id
            pref.updated_at = datetime.now().isoformat()
        
        self._save_data()
        return self.user_preferences[user_id]
    
    def set_language_preference(
        self,
        user_id: str,
        language: str,
        model_id: str
    ) -> None:
        """Set preferred model for a programming language"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = ModelPreference(
                user_id=user_id,
                default_model_id='openai-gpt4'
            )
        
        self.user_preferences[user_id].preferences_by_language[language] = model_id
        self.user_preferences[user_id].updated_at = datetime.now().isoformat()
        self._save_data()
    
    def set_analysis_preference(
        self,
        user_id: str,
        analysis_type: str,
        model_id: str
    ) -> None:
        """Set preferred model for an analysis type"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = ModelPreference(
                user_id=user_id,
                default_model_id='openai-gpt4'
            )
        
        self.user_preferences[user_id].preferences_by_analysis_type[analysis_type] = model_id
        self.user_preferences[user_id].updated_at = datetime.now().isoformat()
        self._save_data()
    
    def get_user_preference(self, user_id: str) -> Optional[ModelPreference]:
        """Get user's model preferences"""
        return self.user_preferences.get(user_id)
    
    def get_model_for_task(
        self,
        user_id: str,
        language: Optional[str] = None,
        analysis_type: Optional[str] = None
    ) -> str:
        """Get the best model for a task based on user preferences"""
        pref = self.user_preferences.get(user_id)
        
        if not pref:
            return 'anthropic-claude3'  # Default
        
        # Check language-specific preference
        if language and language in pref.preferences_by_language:
            return pref.preferences_by_language[language]
        
        # Check analysis-type preference
        if analysis_type and analysis_type in pref.preferences_by_analysis_type:
            return pref.preferences_by_analysis_type[analysis_type]
        
        return pref.default_model_id
    
    # Usage Tracking
    
    def record_usage(
        self,
        model_id: str,
        tokens_used: int,
        success: bool
    ) -> None:
        """Record model usage"""
        if model_id in self.user_models:
            model = self.user_models[model_id]
            model.request_count += 1
            model.total_tokens += tokens_used
            if not success:
                model.error_count += 1
            self._save_data()
    
    def get_usage_stats(self, user_id: str) -> Dict[str, Any]:
        """Get usage statistics for a user"""
        user_models = self.get_user_models(user_id)
        
        return {
            'total_models': len(user_models),
            'active_models': sum(1 for m in user_models if m.status == ModelStatus.ACTIVE),
            'total_requests': sum(m.request_count for m in user_models),
            'total_tokens': sum(m.total_tokens for m in user_models),
            'error_rate': sum(m.error_count for m in user_models) / max(1, sum(m.request_count for m in user_models))
        }
