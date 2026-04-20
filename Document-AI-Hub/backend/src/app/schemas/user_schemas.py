from pydantic import BaseModel, EmailStr
from typing import Optional


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    role: Optional[str] = "user"


class UserOut(BaseModel):
    id: str
    email: EmailStr
    role: str
    is_active: bool
