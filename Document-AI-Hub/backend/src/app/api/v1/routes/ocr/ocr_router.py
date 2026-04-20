from fastapi import APIRouter, Depends, UploadFile, File

from app.api.v1.deps.deps import get_token_payload
from app.services.ocr_service import ocr_service

router = APIRouter()


@router.post('/ocr')
async def ocr(file: UploadFile = File(...), payload: dict = Depends(get_token_payload)):
    data = await file.read()
    res = await ocr_service.ocr_bytes(data)
    return {"result": res}
