from enum import Enum


class Role(str, Enum):
    """Canonical roles referenced in design docs."""
    ADMIN = "admin"
    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"
    ANNOTATOR = "annotator"
    GUEST = "guest"
    LAWYER = "lawyer"
    DOCTOR = "doctor"
    RESEARCHER = "researcher"


__all__ = ["Role"]
