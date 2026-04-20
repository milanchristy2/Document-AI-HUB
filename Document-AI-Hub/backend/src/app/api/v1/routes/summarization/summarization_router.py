from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.v1.deps.deps import get_token_payload

router = APIRouter()


class SummarizeRequest(BaseModel):
    text: str


@router.post('/summarize')
async def summarize(req: SummarizeRequest, payload: dict = Depends(get_token_payload)):
    # placeholder summarization
    return {"summary": req.text[:200]}
