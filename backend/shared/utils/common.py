"""
Common Utilities Module

Shared utilities extracted from multiple modules to reduce code duplication.
This module consolidates frequently used functions.
"""

import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypeVar, Callable
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# String Utilities
# =============================================================================

def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    uid = str(uuid.uuid4())
    return f"{prefix}_{uid}" if prefix else uid


def hash_string(value: str, algorithm: str = "sha256") -> str:
    """Hash a string using specified algorithm."""
    hasher = getattr(hashlib, algorithm)()
    hasher.update(value.encode('utf-8'))
    return hasher.hexdigest()


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max_length, adding suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


# =============================================================================
# DateTime Utilities
# =============================================================================

def utc_now() -> datetime:
    """Get current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


def iso_format(dt: Optional[datetime] = None) -> str:
    """Format datetime as ISO string. Uses current time if not provided."""
    if dt is None:
        dt = utc_now()
    return dt.isoformat()


def parse_iso(iso_string: str) -> datetime:
    """Parse ISO format string to datetime."""
    return datetime.fromisoformat(iso_string.replace('Z', '+00:00'))


# =============================================================================
# Dictionary Utilities
# =============================================================================

def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries. Override takes precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten nested dictionary with dot notation keys."""
    items: List = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_nested(d: Dict, path: str, default: Any = None, sep: str = '.') -> Any:
    """Get nested value from dictionary using dot notation path."""
    keys = path.split(sep)
    result = d
    try:
        for key in keys:
            result = result[key]
        return result
    except (KeyError, TypeError):
        return default


def set_nested(d: Dict, path: str, value: Any, sep: str = '.') -> None:
    """Set nested value in dictionary using dot notation path."""
    keys = path.split(sep)
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


# =============================================================================
# List Utilities
# =============================================================================

def chunk_list(lst: List[T], size: int) -> List[List[T]]:
    """Split list into chunks of specified size."""
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def unique(lst: List[T]) -> List[T]:
    """Return list with unique elements, preserving order."""
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def first(lst: List[T], default: Optional[T] = None) -> Optional[T]:
    """Get first element of list or default."""
    return lst[0] if lst else default


def last(lst: List[T], default: Optional[T] = None) -> Optional[T]:
    """Get last element of list or default."""
    return lst[-1] if lst else default


# =============================================================================
# Async Utilities
# =============================================================================

async def run_with_timeout(
    coro: Any,
    timeout_seconds: float,
    default: T = None
) -> T:
    """Run coroutine with timeout, returning default if timeout exceeded."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout_seconds}s")
        return default


async def gather_with_concurrency(
    limit: int,
    *coros: Any
) -> List[Any]:
    """Run coroutines with limited concurrency."""
    semaphore = asyncio.Semaphore(limit)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))


# =============================================================================
# Decorator Utilities
# =============================================================================

def retry(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for retrying failed operations with exponential backoff."""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            delay = delay_seconds

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}"
                        )
                        await asyncio.sleep(delay)
                        delay *= backoff

            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            import time
            last_exception = None
            delay = delay_seconds

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}"
                        )
                        time.sleep(delay)
                        delay *= backoff

            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def log_execution(func: Callable):
    """Decorator to log function execution time."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = datetime.now(timezone.utc)
        try:
            result = await func(*args, **kwargs)
            duration = (datetime.now(timezone.utc) - start).total_seconds()
            logger.debug(f"{func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds()
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = datetime.now(timezone.utc)
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now(timezone.utc) - start).total_seconds()
            logger.debug(f"{func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds()
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


# =============================================================================
# Validation Utilities
# =============================================================================

def is_valid_email(email: str) -> bool:
    """Basic email validation."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def is_valid_uuid(value: str) -> bool:
    """Check if string is valid UUID."""
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, AttributeError):
        return False


def is_valid_url(url: str) -> bool:
    """Basic URL validation."""
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url, re.IGNORECASE))


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # String
    "generate_id",
    "hash_string",
    "truncate",
    "slugify",
    "camel_to_snake",
    "snake_to_camel",
    # DateTime
    "utc_now",
    "iso_format",
    "parse_iso",
    # Dictionary
    "deep_merge",
    "flatten_dict",
    "get_nested",
    "set_nested",
    # List
    "chunk_list",
    "unique",
    "first",
    "last",
    # Async
    "run_with_timeout",
    "gather_with_concurrency",
    # Decorators
    "retry",
    "log_execution",
    # Validation
    "is_valid_email",
    "is_valid_uuid",
    "is_valid_url",
]
