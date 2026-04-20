from datetime import datetime, timedelta
from typing import Any, Dict

try:
    import bcrypt
except Exception:
    bcrypt = None

import jwt

from app.config.config import settings
from app.exceptions.auth_exceptions import TokenExpiredException, TokenInvalidException


def hash_password(plain: str) -> str:
    if bcrypt:
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(plain.encode("utf-8"), salt).decode("utf-8")
    # fallback lightweight hashing (not as strong as bcrypt)
    import hashlib, os, base64

    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, 100000)
    return base64.b64encode(salt + dk).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    if bcrypt:
        try:
            return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
        except Exception:
            return False
    # fallback verification for pbkdf2 fallback
    import hashlib, base64

    try:
        data = base64.b64decode(hashed.encode("utf-8"))
        salt = data[:16]
        dk = data[16:]
        new = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, 100000)
        return new == dk
    except Exception:
        return False


def create_access_token(data: Dict[str, Any], expires_minutes: int | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=(expires_minutes or settings.JWT_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
    return token


def decode_jwt(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError as e:
        raise TokenExpiredException("Token expired") from e
    except Exception as e:
        raise TokenInvalidException("Token invalid") from e
