"""
Authentication_V1 - Session Manager

Session lifecycle management.
"""

import secrets
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Session model"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    device_info: Optional[Dict[str, str]] = None
    ip_address: Optional[str] = None

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "device_info": self.device_info,
            "ip_address": self.ip_address,
        }


class SessionManager:
    """
    Session lifecycle manager.

    Features:
    - Session creation and validation
    - Activity tracking
    - Multi-device support
    - Session expiration
    """

    def __init__(
        self,
        session_ttl: int = 3600,  # 1 hour
        max_sessions_per_user: int = 5,
    ):
        self.session_ttl = session_ttl
        self.max_sessions_per_user = max_sessions_per_user
        self._sessions: Dict[str, Session] = {}
        self._user_sessions: Dict[str, list] = {}

    async def create_session(
        self,
        user_id: str,
        device_info: Optional[Dict[str, str]] = None,
        ip_address: Optional[str] = None,
    ) -> Session:
        """Create new session for user"""
        # Clean expired sessions
        self._cleanup_expired()

        # Check session limit
        user_sessions = self._user_sessions.get(user_id, [])
        if len(user_sessions) >= self.max_sessions_per_user:
            # Remove oldest session
            oldest = min(user_sessions, key=lambda s: self._sessions[s].created_at)
            await self.invalidate_session(oldest)

        session_id = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)

        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            expires_at=now + timedelta(seconds=self.session_ttl),
            device_info=device_info,
            ip_address=ip_address,
        )

        self._sessions[session_id] = session

        if user_id not in self._user_sessions:
            self._user_sessions[user_id] = []
        self._user_sessions[user_id].append(session_id)

        logger.info(f"Session created for user {user_id}")

        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        session = self._sessions.get(session_id)

        if session and session.is_expired():
            await self.invalidate_session(session_id)
            return None

        return session

    async def update_activity(self, session_id: str) -> bool:
        """Update session last activity"""
        session = await self.get_session(session_id)

        if session:
            session.last_activity = datetime.now(timezone.utc)
            session.expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.session_ttl)
            return True

        return False

    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate session"""
        session = self._sessions.pop(session_id, None)

        if session:
            if session.user_id in self._user_sessions:
                self._user_sessions[session.user_id].remove(session_id)
            logger.info(f"Session invalidated: {session_id}")
            return True

        return False

    async def invalidate_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for user"""
        session_ids = self._user_sessions.get(user_id, []).copy()

        for session_id in session_ids:
            await self.invalidate_session(session_id)

        return len(session_ids)

    async def get_user_sessions(self, user_id: str) -> list:
        """Get all active sessions for user"""
        self._cleanup_expired()

        session_ids = self._user_sessions.get(user_id, [])
        sessions = [self._sessions[sid] for sid in session_ids if sid in self._sessions]

        return sessions

    def _cleanup_expired(self):
        """Remove expired sessions"""
        expired = [
            sid for sid, session in self._sessions.items()
            if session.is_expired()
        ]

        for sid in expired:
            session = self._sessions.pop(sid)
            if session.user_id in self._user_sessions:
                self._user_sessions[session.user_id] = [
                    s for s in self._user_sessions[session.user_id] if s != sid
                ]

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        self._cleanup_expired()

        return {
            "total_sessions": len(self._sessions),
            "total_users": len(self._user_sessions),
            "avg_sessions_per_user": len(self._sessions) / max(1, len(self._user_sessions)),
        }
