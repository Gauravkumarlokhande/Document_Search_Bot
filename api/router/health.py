
from fastapi import FastAPI, APIRouter, Form
from validation_models.schemas import health


router=APIRouter()

@router.post("/health")
async def root(request:health):
    return {"message": f"Hello {request.name}, I'm alive"}

