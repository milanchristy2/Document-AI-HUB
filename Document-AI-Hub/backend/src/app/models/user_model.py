import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID

from app.infra.db.session import Base


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    role = Column(String, default="user")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def check_active(self) -> bool:
        return bool(self.is_active)

    MODE_ROLES = {
        "general": {"user", "admin"},
        "legal": {"admin"},
    }

    def can_access_mode(self, mode: str) -> bool:
        allowed = self.MODE_ROLES.get(mode, {"user", "admin"})
        return self.role in allowed
