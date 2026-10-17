from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api.schemas import LoginRequest, TokenResponse, UserCreate, UserRead
from src.core.security import create_access_token, decode_access_token
from src.db.models import User
from src.db.repositories import (
    authenticate_user,
    create_user,
    get_user_by_email,
    get_user_by_id,
)


router = APIRouter(prefix="/auth", tags=["auth"])
bearer_scheme = HTTPBearer(auto_error=False)


def _user_read(user: User) -> UserRead:
    return UserRead(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_admin=user.is_admin,
        created_at=user.created_at,
    )


def _credentials_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired access token",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _credentials_error()

    subject = decode_access_token(credentials.credentials)
    if subject is None:
        raise _credentials_error()

    try:
        user_id = UUID(subject)
    except ValueError as error:
        raise _credentials_error() from error

    user = get_user_by_id(user_id)
    if user is None or not user.is_active:
        raise _credentials_error()
    return user


def get_optional_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> User | None:
    if credentials is None:
        return None
    return get_current_user(credentials)


@router.post("/register", response_model=UserRead, status_code=201)
def register(request: UserCreate) -> UserRead:
    email = request.email.strip().lower()
    if get_user_by_email(email) is not None:
        raise HTTPException(status_code=400, detail="Email is already registered")

    try:
        user = create_user(
            email=email,
            password=request.password,
            full_name=request.full_name,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    if user is None:
        if get_user_by_email(email) is not None:
            raise HTTPException(status_code=400, detail="Email is already registered")
        raise HTTPException(status_code=503, detail="User database is unavailable")
    return _user_read(user)


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest) -> TokenResponse:
    user = authenticate_user(request.email, request.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return TokenResponse(
        access_token=create_access_token(str(user.id)),
        token_type="bearer",
        user=_user_read(user),
    )


@router.get("/me", response_model=UserRead)
def me(current_user: User = Depends(get_current_user)) -> UserRead:
    return _user_read(current_user)