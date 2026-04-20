from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError
from typing import Any, cast
import logging

from app.models.user_model import User
from app.core.security import hash_password, verify_password, create_access_token

logger = logging.getLogger(__name__)


class AuthService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def signup(self, email: str, password: str, role: str = "user") -> User:
        """Register a new user with email and password.
        
        Raises:
            ValueError: If email already exists
            Exception: For other database errors
        """
        hashed = hash_password(password)
        user = User(email=email, password=hashed, role=role)
        
        try:
            self.db.add(user)
            await self.db.flush()
            return user
        except IntegrityError as e:
            await self.db.rollback()
            # Check if it's a duplicate email constraint
            if "email" in str(e).lower() or "unique" in str(e).lower():
                logger.warning(f"Signup failed: email '{email}' already in use")
                raise ValueError(f"Email '{email}' is already in use. Please try a different email.")
            # Other integrity errors
            raise Exception(f"Database error: {str(e)}")
        except OperationalError as e:
            await self.db.rollback()
            logger.error(f"Database operational error during signup: {e}")
            raise Exception("Database error: users table may not exist. Please contact support.")
        except Exception as e:
            await self.db.rollback()
            logger.exception(f"Unexpected error during signup: {e}")
            raise

    async def login(self, email: str, password: str):
        """Authenticate user and return access token.
        
        Raises:
            ValueError: If credentials are invalid
            Exception: For database errors
        """
        try:
            q = select(User).where(User.email == email)
            res = await self.db.execute(q)
            user = res.scalars().first()
            
            if not user or not verify_password(password, str(user.password)):
                logger.debug(f"Login failed for email: {email} (invalid credentials)")
                raise ValueError("Invalid credentials")
            
            token = create_access_token({"sub": user.id, "email": user.email, "role": user.role})
            logger.info(f"Login successful for email: {email}")
            return {"access_token": token, "token_type": "bearer", "role": user.role}
        
        except OperationalError as e:
            logger.error(f"Database operational error during login: {e}")
            raise Exception("Database error: users table may not exist. Please contact support.")
        except ValueError:
            # Re-raise validation errors (invalid credentials)
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during login: {e}")
            raise

    async def get_user_by_id(self, user_id: str) -> User | None:
        """Get user by ID."""
        try:
            q = select(User).where(User.id == user_id)
            res = await self.db.execute(q)
            return res.scalars().first()
        except OperationalError:
            logger.error(f"Database error while fetching user {user_id}")
            return None

    async def change_password(self, user_id: str, old_pw: str, new_pw: str):
        """Change user password after validating old password."""
        try:
            user = await self.get_user_by_id(user_id)
            if not user or not verify_password(old_pw, str(user.password)):
                raise ValueError("Invalid credentials")
            
            user.password = cast(Any, hash_password(new_pw))
            self.db.add(user)
            await self.db.flush()
            return True
        except OperationalError as e:
            logger.error(f"Database error while changing password: {e}")
            raise Exception("Database error. Please contact support.")
        except ValueError:
            raise

