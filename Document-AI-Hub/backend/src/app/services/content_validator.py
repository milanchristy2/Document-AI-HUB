"""Content validation by user role and document type."""
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Define allowed document types by role
ROLE_DOCUMENT_RESTRICTIONS = {
    "lawyer": {
        "allowed_types": ["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
        "allowed_keywords": ["contract", "agreement", "legal", "clause", "law", "court", "litigation", "attorney", "statute", "ordinance"],
        "description": "Legal documents only (contracts, agreements, laws, court documents)"
    },
    "doctor": {
        "allowed_types": ["application/pdf", "text/plain", "image/png", "image/jpeg"],
        "allowed_keywords": ["patient", "medical", "diagnosis", "treatment", "hospital", "clinical", "health", "disease", "prescription", "symptoms"],
        "description": "Medical documents only (patient records, prescriptions, clinical notes)"
    },
    "researcher": {
        "allowed_types": ["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
        "allowed_keywords": ["research", "study", "paper", "experiment", "data", "analysis", "methodology", "conclusion", "abstract", "hypothesis"],
        "description": "Research papers and academic documents"
    },
    "analyst": {
        "allowed_types": ["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
        "allowed_keywords": ["report", "analysis", "data", "financial", "summary", "metric", "trend", "forecast", "insight"],
        "description": "Analytics and business reports"
    },
    "user": {
        "allowed_types": ["application/pdf", "text/plain", "image/png", "image/jpeg", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "audio/mpeg", "video/mp4"],
        "allowed_keywords": [],  # No restrictions for general users
        "description": "All document types allowed"
    }
}


def validate_document_for_role(filename: str, content_type: str, extracted_text: Optional[str], user_role: Optional[str] = None) -> Tuple[bool, str]:
    """Validate if document is appropriate for user's role.
    
    Args:
        filename: Original filename
        content_type: MIME type of document
        extracted_text: First 500 chars of extracted text (optional)
        user_role: User's role (lawyer, doctor, researcher, etc.)
    
    Returns:
        (is_valid, message)
    """
    # If no role specified or user is admin, allow everything
    if not user_role or user_role == "admin":
        return True, "Document allowed for admin user"
    
    # Normalize role
    user_role = user_role.lower().strip()
    
    # Check if role has restrictions
    if user_role not in ROLE_DOCUMENT_RESTRICTIONS:
        user_role = "user"  # Default to general user restrictions
    
    restrictions = ROLE_DOCUMENT_RESTRICTIONS[user_role]
    
    # Check 1: Content type validation
    if restrictions["allowed_types"] and content_type not in restrictions["allowed_types"]:
        return False, f"File type '{content_type}' not allowed for {user_role}. Allowed types: {', '.join(restrictions['allowed_types'])}"
    
    # Check 2: Keyword validation (if text is available)
    if extracted_text and restrictions["allowed_keywords"]:
        text_lower = extracted_text.lower()
        has_relevant_keyword = any(keyword in text_lower for keyword in restrictions["allowed_keywords"])
        
        if not has_relevant_keyword:
            keywords_str = ", ".join(restrictions["allowed_keywords"])
            return False, f"Document does not appear to be {restrictions['description']}. Expected keywords: {keywords_str}"
    
    return True, f"Document valid for {user_role}"


def get_role_description(user_role: Optional[str] = None) -> str:
    """Get document type description for a role."""
    if not user_role:
        user_role = "user"
    
    user_role = user_role.lower().strip()
    restrictions = ROLE_DOCUMENT_RESTRICTIONS.get(user_role, ROLE_DOCUMENT_RESTRICTIONS["user"])
    return restrictions["description"]


def get_allowed_types(user_role: Optional[str] = None) -> list:
    """Get allowed file types for a role."""
    if not user_role:
        user_role = "user"
    
    user_role = user_role.lower().strip()
    restrictions = ROLE_DOCUMENT_RESTRICTIONS.get(user_role, ROLE_DOCUMENT_RESTRICTIONS["user"])
    return restrictions["allowed_types"]
