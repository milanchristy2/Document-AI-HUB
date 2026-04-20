from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.deps.deps import get_db, get_token_payload
from app.models.chat_model import ChatMessage
from app.services.memory_service import memory_service
from app.chains.rag_chain import call_llm_str

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str
    content: str
    provider: str | None = None

@router.post('/message')
async def send_message(req: ChatRequest, payload: dict = Depends(get_token_payload), db: AsyncSession = Depends(get_db)):
    user_id = payload.get('sub')
    
    # Save user message to database
    user_msg = ChatMessage(user_id=user_id, session_id=req.session_id, role='user', content=req.content)
    db.add(user_msg)
    await db.flush()

    # Load conversation history from fast Redis cache
    history = await memory_service.load(user_id, req.session_id)
    chat_history_str = memory_service.format_for_prompt(history)

    # Build prompt with history
    prompt = f"You are a helpful AI assistant.\n\nConversation History:\n{chat_history_str}\n\nUser: {req.content}\nAssistant: "

    # Call LLM backend
    response_text = await call_llm_str(prompt, provider=req.provider)

    # Append to Redis
    await memory_service.append(user_id, req.session_id, 'user', req.content)
    await memory_service.append(user_id, req.session_id, 'assistant', response_text)

    # Save assistant message to database
    assistant_msg = ChatMessage(user_id=user_id, session_id=req.session_id, role='assistant', content=response_text)
    db.add(assistant_msg)
    await db.commit()

    return {"id": assistant_msg.id, "content": assistant_msg.content, "session_id": req.session_id, "role": "assistant"}
