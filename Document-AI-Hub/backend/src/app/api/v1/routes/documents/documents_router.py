from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.deps.deps import get_db, get_token_payload
from app.services.document_service import DocumentService
from app.services.content_validator import validate_document_for_role
from app.middleware.rate_limiter import check_rate_limit
from app.processors.extractors.extractors import extract_text

router = APIRouter()


@router.post('/upload')
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...), payload: dict = Depends(get_token_payload), db: AsyncSession = Depends(get_db)):
    user_id = payload.get('sub', 'unknown')
    user_role = payload.get('role', 'user')
    
    # Check rate limit
    allowed, msg, remaining = check_rate_limit(str(user_id), "upload", str(user_role))
    if not allowed:
        raise HTTPException(status_code=429, detail=msg)
    
    data = await file.read()
    filename = file.filename or "document"
    content_type = file.content_type or "application/octet-stream"
    
    # Extract text for validation
    try:
        extracted_text = extract_text(filename, content_type)[:500]
    except Exception:
        extracted_text = ""
    
    # Validate document for user's role
    is_valid, validation_msg = validate_document_for_role(filename, content_type, extracted_text, str(user_role))
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Upload validation failed: {validation_msg}")
    
    service = DocumentService(db)
    try:
        doc = await service.upload(data, filename, content_type, str(user_id)) #type:ignore
        # schedule ingestion asynchronously
        try:
            import asyncio
            from app.ingestion.pipeline import IngestPipeline
            asyncio.create_task(IngestPipeline(None).run(doc.id, doc.storage_path, content_type, filename)) #type:ignore
        except Exception:
            pass
        return {
            "id": doc.id,
            "filename": doc.filename,
            "status": str(doc.status),
            "message": f"Upload successful. {validation_msg}",
            "rate_limit_remaining": remaining
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get('/{doc_id}')
async def get_doc(doc_id: str, payload: dict = Depends(get_token_payload), db: AsyncSession = Depends(get_db)):
    user_id = payload.get('sub')
    service = DocumentService(db)
    doc = await service.get_by_id(doc_id, user_id) #type:ignore
    if not doc:
        raise HTTPException(status_code=404, detail='Not found')
    return {"id": doc.id, "filename": doc.filename, "status": str(doc.status)}


@router.get('/')
async def list_docs(payload: dict = Depends(get_token_payload), db: AsyncSession = Depends(get_db)):
    user_id = payload.get('sub')
    service = DocumentService(db)
    rows = await service.list_documents(user_id) #type:ignore
    return rows


@router.delete('/{doc_id}')
async def delete_doc(doc_id: str, payload: dict = Depends(get_token_payload), db: AsyncSession = Depends(get_db)):
    user_id = payload.get('sub')
    service = DocumentService(db)
    ok = await service.delete(doc_id, user_id) #type:ignore
    if not ok:
        raise HTTPException(status_code=404, detail='Not found')
    return {"deleted": True}
