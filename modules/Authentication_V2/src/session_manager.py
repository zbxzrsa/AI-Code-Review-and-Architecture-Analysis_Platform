"""
Authentication_V2 - Enhanced Session Manager

Production session management with device tracking and security.
"""

import secrets
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Device information"""
    device_id: str
    device_type: str  # web, mobile, api
    user_agent: Optional[str] = None
    os: Optional[str] = None
    browser: Optional[str] = None


@dataclass
class Session:
    """Enhanced session with device tracking"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    device: Optional[DeviceInfo] = None
    ip_address: Optional[str] = None
    location: Optional[str] = None
    is_trusted: bool = False

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id[:8] + "...",  # Masked
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "device": {
                "type": self.device.device_type,
                "os": self.device.os,
                "browser": self.device.browser,
            } if self.device else None,
            "ip_address": self.ip_address,
            "location": self.location,
            "is_trusted": self.is_trusted,
        }


class SessionManager:
    """
    Production session manager.

    V2 Features:
    - Device fingerprinting
    - Concurrent session limits
    - Trusted device management
    - Session security alerts
    """

    def __init__(
        self,
        session_ttl: int = 3600,
        max_sessions_per_user: int = 5,
        max_sessions_per_device: int = 3,
    ):
        self.session_ttl = session_ttl
        self.max_sessions_per_user = max_sessions_per_user
        self.max_sessions_per_device = max_sessions_per_device

        self._sessions: Dict[str, Session] = {}
        self._user_sessions: Dict[str, List[str]] = {}
        self._trusted_devices: Dict[str, List[str]] = {}

    async def create_session(
        self,
        user_id: str,
        device_info: Optional[DeviceInfo] = None,
        ip_address: Optional[str] = None,
        trust_device: bool = False,
    ) -> Session:
        """Create session with device tracking"""
        self._cleanup_expired()

        # Check and enforce session limits
        await self._enforce_session_limits(user_id, device_info)

        session_id = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)

        # Check if device is trusted
        is_trusted = False
        if device_info and user_id in self._trusted_devices:
            is_trusted = device_info.device_id in self._trusted_devices[user_id]

        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            expires_at=now + timedelta(seconds=self.session_ttl),
            device=device_info,
            ip_address=ip_address,
            is_trusted=is_trusted,
        )

        self._sessions[session_id] = session

        if user_id not in self._user_sessions:
            self._user_sessions[user_id] = []
        self._user_sessions[user_id].append(session_id)

        # Trust device if requested
        if trust_device and device_info:
            await self.trust_device(user_id, device_info.device_id)

        logger.info(f"Session created: user={user_id}, device={device_info.device_type if device_info else 'unknown'}")

        return session

    async def _enforce_session_limits(
        self,
        user_id: str,
        device_info: Optional[DeviceInfo],
    ):
        """Enforce session limits"""
        user_sessions = self._user_sessions.get(user_id, [])
        active_sessions = [
            sid for sid in user_sessions
            if sid in self._sessions and not self._sessions[sid].is_expired()
        ]

        # User limit
        while len(active_sessions) >= self.max_sessions_per_user:
            oldest = min(active_sessions, key=lambda s: self._sessions[s].created_at)
            await self.invalidate_session(oldest)
            active_sessions.remove(oldest)

        # Device limit
        if device_info:
            device_sessions = [
                sid for sid in active_sessions
                if self._sessions[sid].device and
                   self._sessions[sid].device.device_id == device_info.device_id
            ]

            while len(device_sessions) >= self.max_sessions_per_device:
                oldest = min(device_sessions, key=lambda s: self._sessions[s].created_at)
                await self.invalidate_session(oldest)
                device_sessions.remove(oldest)

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        session = self._sessions.get(session_id)

        if session and session.is_expired():
            await self.invalidate_session(session_id)
            return None

        return session

    async def update_activity(
        self,
        session_id: str,
        ip_address: Optional[str] = None,
    ) -> bool:
        """Update session activity"""
        session = await self.get_session(session_id)

        if session:
            # Check for suspicious activity (IP change)
            if ip_address and session.ip_address and ip_address != session.ip_address:
                if not session.is_trusted:
                    logger.warning(f"IP change detected for session: {session_id[:8]}")

            session.last_activity = datetime.now(timezone.utc)
            session.expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.session_ttl)

            if ip_address:
                session.ip_address = ip_address

            return True

        return False

    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate session"""
        session = self._sessions.pop(session_id, None)

        if session:
            if session.user_id in self._user_sessions:
                self._user_sessions[session.user_id] = [
                    s for s in self._user_sessions[session.user_id] if s != session_id
                ]
            logger.info(f"Session invalidated: {session_id[:8]}")
            return True

        return False

    async def invalidate_all_sessions(self, user_id: str, except_session: Optional[str] = None) -> int:
        """Invalidate all user sessions except one"""
        session_ids = self._user_sessions.get(user_id, []).copy()
        count = 0

        for session_id in session_ids:
            if session_id != except_session:
                await self.invalidate_session(session_id)
                count += 1

        return count

    async def trust_device(self, user_id: str, device_id: str):
        """Mark device as trusted"""
        if user_id not in self._trusted_devices:
            self._trusted_devices[user_id] = []

        if device_id not in self._trusted_devices[user_id]:
            self._trusted_devices[user_id].append(device_id)
            logger.info(f"Device trusted: user={user_id}, device={device_id[:8]}")

    async def untrust_device(self, user_id: str, device_id: str):
        """Remove device from trusted list"""
        if user_id in self._trusted_devices:
            self._trusted_devices[user_id] = [
                d for d in self._trusted_devices[user_id] if d != device_id
            ]

    async def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all active sessions for user"""
        self._cleanup_expired()

        session_ids = self._user_sessions.get(user_id, [])
        sessions = [
            self._sessions[sid] for sid in session_ids
            if sid in self._sessions
        ]

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
            "trusted_devices": sum(len(d) for d in self._trusted_devices.values()),
        }
