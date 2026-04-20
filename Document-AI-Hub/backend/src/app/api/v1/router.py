from fastapi import APIRouter

from app.api.v1.routes.auth.auth_router import router as auth_router
from app.api.v1.routes.documents.documents_router import router as documents_router
from app.api.v1.routes.chat.chat_router import router as chat_router
from app.api.v1.routes.summarization.summarization_router import router as summarization_router
from app.api.v1.routes.ocr.ocr_router import router as ocr_router
from app.api.v1.routes.transcription.transcription_router import router as transcription_router
from app.api.v1.routes.rag.rag_router import router as rag_router
from app.api.v1.routes.users.users_router import router as users_router

v1_router = APIRouter(prefix="/api/v1")

v1_router.include_router(auth_router, prefix="/auth", tags=["auth"])
v1_router.include_router(documents_router, prefix="/documents", tags=["documents"])
v1_router.include_router(chat_router, prefix="/chat", tags=["chat"])
v1_router.include_router(summarization_router, prefix="/summaries", tags=["summarization"])
v1_router.include_router(ocr_router, prefix="/ocr", tags=["ocr"])
v1_router.include_router(transcription_router, prefix="/transcribe", tags=["transcription"])
v1_router.include_router(rag_router, prefix="/rag", tags=["rag"])
v1_router.include_router(users_router, prefix="/users", tags=["users"])
