"""Application-level constants for RAG and RBAC."""

ACTIONS = [
    "ingest",
    "query",
    "search",
    "delete",
    "manage_users",
    "annotate",
    "summarize",
]

RESOURCES = ["document", "collection", "user"]

DEFAULTS = {
    "RAG_TOP_K": 5,
}
