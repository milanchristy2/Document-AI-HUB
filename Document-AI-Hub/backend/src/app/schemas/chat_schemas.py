from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    query: str
    file_id: Optional[str]
    session_id: Optional[str]
    mode: Optional[str] = "general"
    provider: Optional[str] = "ollama"
    cot: Optional[bool] = False
    cite: Optional[bool] = False
    decompose: Optional[bool] = False
    use_hyde: Optional[bool] = False
    rewrite: Optional[bool] = False
    refine: Optional[bool] = False


class SummarizeRequest(BaseModel):
    file_id: str
    mode: Optional[str] = "general"
    style: Optional[str] = "general"
    provider: Optional[str] = "ollama"
    max_length: Optional[int] = 512
    stream: Optional[bool] = False
    sequential: Optional[bool] = False


class SummarizeResponse(BaseModel):
    summary: str
    key_facts: Optional[str]
    action_items: Optional[str]
