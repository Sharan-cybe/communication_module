from fastapi import APIRouter, UploadFile, File
from app.core.pipeline import run_pipeline

router = APIRouter()

@router.post("/evaluate")
async def evaluate(audio: UploadFile = File(...), question: str = ""):
    result = await run_pipeline(audio, question)
    return result