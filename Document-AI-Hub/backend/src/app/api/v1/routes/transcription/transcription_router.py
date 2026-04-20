from fastapi import APIRouter, Depends, UploadFile, File

from app.api.v1.deps.deps import get_token_payload
from app.services.transcription_service import transcription_service

router = APIRouter()


@router.post('/transcribe')
async def transcribe(file: UploadFile = File(...), payload: dict = Depends(get_token_payload)):
    data = await file.read()
    res = await transcription_service.transcribe_bytes(data)
    return {"result": res}
