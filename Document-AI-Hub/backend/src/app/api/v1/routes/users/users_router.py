from typing import List
from fastapi import APIRouter, Depends, HTTPException

from app.api.v1.deps.deps import get_token_payload

router = APIRouter()


@router.get('/')
async def list_users(payload: dict = Depends(get_token_payload)):
    # very small demo: in production, fetch from DB with proper pagination
    # ensure only admins can list users
    if payload.get('role') != 'admin':
        raise HTTPException(status_code=403, detail='forbidden')
    users = [
        {"id": "1", "username": "alice", "email": "alice@example.com", "role": "admin"},
        {"id": "2", "username": "bob", "email": "bob@example.com", "role": "viewer"},
    ]
    return {"users": users}


@router.get('/{user_id}')
async def get_user(user_id: str, payload: dict = Depends(get_token_payload)):
    # demo: allow admins or requesting user
    if payload.get('role') != 'admin' and payload.get('sub') != user_id:
        raise HTTPException(status_code=403, detail='forbidden')
    # return mock
    return {"id": user_id, "username": f"user-{user_id}", "email": f"user+{user_id}@example.com", "role": "viewer"}


@router.post('/create')
async def create_user(payload_in: dict, payload: dict = Depends(get_token_payload)):
    # only admin can create users in this demo
    if payload.get('role') != 'admin':
        raise HTTPException(status_code=403, detail='forbidden')
    # in prod: validate, hash password, store in DB
    new = {"id": "99", "username": payload_in.get('username'), "email": payload_in.get('email'), "role": payload_in.get('role', 'viewer')}
    return {"user": new}


@router.delete('/{user_id}')
async def delete_user(user_id: str, payload: dict = Depends(get_token_payload)):
    if payload.get('role') != 'admin':
        raise HTTPException(status_code=403, detail='forbidden')
    return {"deleted": True, "id": user_id}
