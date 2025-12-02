"""
Development Auth Server / 开发认证服务器

A simple standalone authentication server for frontend development.
用于前端开发的简单独立认证服务器。

Run with: python dev-auth-server.py
运行命令: python dev-auth-server.py

Endpoints:
- POST /auth/login - Login with email/password
- POST /auth/register - Register with invitation code
- POST /auth/logout - Logout
- POST /auth/refresh - Refresh token
- GET /auth/me - Get current user
"""

import secrets
import hashlib
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, validator
import uvicorn

# ============================================
# Configuration / 配置
# ============================================

# Admin invitation code / 管理员邀请码
ADMIN_INVITATION_CODE = "ZBXzbx123"

# In-memory user database for development / 开发用内存用户数据库
users_db: Dict[str, Dict[str, Any]] = {}

# Token storage / 令牌存储
tokens_db: Dict[str, str] = {}  # token -> user_email

# ============================================
# Models / 模型
# ============================================

class LoginRequest(BaseModel):
    """Login request / 登录请求"""
    email: EmailStr
    password: str
    invitation_code: Optional[str] = None


class RegisterRequest(BaseModel):
    """Register request / 注册请求"""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    name: str = Field(..., min_length=2, max_length=50)
    invitation_code: str

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain a lowercase letter')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain an uppercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain a number')
        return v


class UserResponse(BaseModel):
    """User response / 用户响应"""
    id: str
    email: str
    name: str
    role: str
    avatar: Optional[str] = None
    created_at: datetime


class AuthResponse(BaseModel):
    """Auth response / 认证响应"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    user: UserResponse


class RefreshRequest(BaseModel):
    """Refresh request / 刷新请求"""
    refresh_token: str


# ============================================
# Helper Functions / 辅助函数
# ============================================

def hash_password(password: str) -> str:
    """Hash password / 哈希密码"""
    return hashlib.sha256(password.encode()).hexdigest()


def generate_token() -> str:
    """Generate token / 生成令牌"""
    return secrets.token_urlsafe(32)


def get_user_by_email(email: str) -> Optional[Dict]:
    """Get user by email / 通过邮箱获取用户"""
    return users_db.get(email)


def create_user(email: str, password: str, name: str, role: str = "user") -> Dict:
    """Create user / 创建用户"""
    user_id = f"user_{secrets.token_hex(8)}"
    user = {
        "id": user_id,
        "email": email,
        "password_hash": hash_password(password),
        "name": name,
        "role": role,
        "avatar": None,
        "created_at": datetime.utcnow()
    }
    users_db[email] = user
    return user


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password / 验证密码"""
    return hash_password(password) == password_hash


# ============================================
# FastAPI App / FastAPI 应用
# ============================================

app = FastAPI(
    title="Dev Auth Server / 开发认证服务器",
    description="Development authentication server for frontend testing",
    version="1.0.0"
)

# CORS - Allow all origins for development
# CORS - 开发环境允许所有来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Endpoints / 端点
# ============================================

@app.get("/")
async def root():
    """Root endpoint / 根端点"""
    return {
        "service": "Dev Auth Server",
        "version": "1.0.0",
        "status": "running",
        "invitation_code": ADMIN_INVITATION_CODE
    }


@app.get("/health")
async def health():
    """Health check / 健康检查"""
    return {"status": "healthy"}


@app.post("/auth/register", response_model=AuthResponse)
async def register(request: RegisterRequest, response: Response):
    """
    Register new user / 注册新用户
    
    Invitation code: ZBXzbx123
    邀请码: ZBXzbx123
    """
    # Validate invitation code / 验证邀请码
    if request.invitation_code != ADMIN_INVITATION_CODE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid invitation code. Use: {ADMIN_INVITATION_CODE}"
        )
    
    # Check if email exists / 检查邮箱是否存在
    if get_user_by_email(request.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )
    
    # Create user / 创建用户
    user = create_user(
        email=request.email,
        password=request.password,
        name=request.name,
        role="user"
    )
    
    # Generate tokens / 生成令牌
    access_token = generate_token()
    refresh_token = generate_token()
    
    # Store tokens / 存储令牌
    tokens_db[access_token] = request.email
    tokens_db[refresh_token] = request.email
    
    # Set cookie / 设置Cookie
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=3600,
        samesite="lax"
    )
    
    return AuthResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserResponse(
            id=user["id"],
            email=user["email"],
            name=user["name"],
            role=user["role"],
            avatar=user["avatar"],
            created_at=user["created_at"]
        )
    )


@app.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest, response: Response):
    """Login user / 用户登录"""
    # Find user / 查找用户
    user = get_user_by_email(request.email)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Verify password / 验证密码
    if not verify_password(request.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Generate tokens / 生成令牌
    access_token = generate_token()
    refresh_token = generate_token()
    
    # Store tokens / 存储令牌
    tokens_db[access_token] = request.email
    tokens_db[refresh_token] = request.email
    
    # Set cookie / 设置Cookie
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=3600,
        samesite="lax"
    )
    
    return AuthResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserResponse(
            id=user["id"],
            email=user["email"],
            name=user["name"],
            role=user["role"],
            avatar=user["avatar"],
            created_at=user["created_at"]
        )
    )


@app.post("/auth/logout")
async def logout(response: Response):
    """Logout user / 用户登出"""
    response.delete_cookie("access_token")
    return {"message": "Logged out successfully"}


@app.post("/auth/refresh", response_model=AuthResponse)
async def refresh(request: RefreshRequest):
    """Refresh token / 刷新令牌"""
    email = tokens_db.get(request.refresh_token)
    
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    user = get_user_by_email(email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    # Generate new tokens / 生成新令牌
    access_token = generate_token()
    new_refresh_token = generate_token()
    
    # Update token storage / 更新令牌存储
    del tokens_db[request.refresh_token]
    tokens_db[access_token] = email
    tokens_db[new_refresh_token] = email
    
    return AuthResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        user=UserResponse(
            id=user["id"],
            email=user["email"],
            name=user["name"],
            role=user["role"],
            avatar=user["avatar"],
            created_at=user["created_at"]
        )
    )


@app.get("/auth/me", response_model=UserResponse)
async def get_current_user(request: Request):
    """Get current user / 获取当前用户"""
    # Get token from header or cookie
    auth_header = request.headers.get("Authorization", "")
    token = None
    
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
    else:
        token = request.cookies.get("access_token")
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    email = tokens_db.get(token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    user = get_user_by_email(email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return UserResponse(
        id=user["id"],
        email=user["email"],
        name=user["name"],
        role=user["role"],
        avatar=user["avatar"],
        created_at=user["created_at"]
    )


# ============================================
# Main / 主程序
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Dev Auth Server Starting...")
    print("=" * 50)
    print(f"📧 Invitation Code: {ADMIN_INVITATION_CODE}")
    print(f"🌐 Server: http://localhost:8001")
    print(f"📖 Docs: http://localhost:8001/docs")
    print("=" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
