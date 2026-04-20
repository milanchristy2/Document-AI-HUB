"""Role-based access control (RBAC) helper.

Provides a simple permission matrix and helpers to check permissions.
"""
from typing import Set, Dict, Optional
from ..constants.roles import Role
from ..constants.constants import ACTIONS


class PermissionDenied(Exception):
    pass


# A minimal, explicit permission matrix. Keys are Role values.
# Values are sets of allowed action strings from `ACTIONS`.
_PERMISSION_MATRIX: Dict[str, Set[str]] = {
    Role.ADMIN: set(ACTIONS),
    Role.OWNER: set(["ingest", "query", "search", "delete", "summarize", "annotate"]),
    Role.EDITOR: set(["ingest", "query", "search", "annotate", "summarize"]),
    Role.ANNOTATOR: set(["annotate", "query", "search"]),
    Role.VIEWER: set(["query", "search", "summarize"]),
    Role.GUEST: set(["search"]),
    Role.LAWYER: set(["query", "search", "summarize", "annotate"]),
    Role.DOCTOR: set(["query", "search", "summarize"]),
    Role.RESEARCHER: set(["query", "search", "summarize"]),
    "user": set(["query", "search", "summarize"]),  # Default role assigned to new users
}


def is_allowed(role: Optional[str], action: str, resource: Optional[str] = None) -> bool:
    """Return True if the given role is allowed to perform action on resource.

    - `role` may be a Role enum, value string, or None.
    - `action` must be one of the actions in `ACTIONS`.
    - `resource` is presently not used for fine-grained rules but accepted for future extension.
    """
    if action not in ACTIONS:
        return False
    if not role:
        return False
    role_key = role if isinstance(role, str) else role.value
    allowed = _PERMISSION_MATRIX.get(role_key)
    if not allowed:
        return False
    return action in allowed


def require_permission(role: Optional[str], action: str, resource: Optional[str] = None) -> None:
    """Raise PermissionDenied if not allowed."""
    if not is_allowed(role, action, resource):
        raise PermissionDenied(f"role={role} not permitted to perform {action} on {resource}")


def permissions_for(role: Optional[str]) -> Set[str]:
    if not role:
        return set()
    return set(_PERMISSION_MATRIX.get(role if isinstance(role, str) else role.value, set()))


__all__ = ["is_allowed", "require_permission", "permissions_for", "PermissionDenied"]
