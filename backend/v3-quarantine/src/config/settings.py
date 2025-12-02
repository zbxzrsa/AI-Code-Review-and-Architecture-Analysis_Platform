"""Configuration settings for V3 quarantine."""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from backend.shared.config.settings import settings

__all__ = ["settings"]
