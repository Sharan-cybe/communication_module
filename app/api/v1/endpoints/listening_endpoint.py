from fastapi import APIRouter, UploadFile, File, Form, Body
from app.core.listening_pipeline import (
    generate_listening_clips,
    evaluate_clip_response,
    aggregate_session,
)

router = APIRouter()


@router.get("/clips")
async def get_clips():
    clips = await generate_listening_clips()
    return {"clips": clips}


@router.post("/respond")
async def respond(
    audio: UploadFile = File(...),
    session_id: str = Form(...),
    clip_id: str = Form(...),
    question_index: int = Form(0),
):
    return await evaluate_clip_response(
        session_id=session_id,
        clip_id=clip_id,
        audio_file=audio,
        question_index=question_index,
    )

@router.post("/aggregate")
async def aggregate(clip_results: list = Body(...)):
    return aggregate_session(clip_results)