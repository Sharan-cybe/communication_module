from fastapi import APIRouter, Body, UploadFile, File, Form
from app.services.speaking.question_generator import generate_speaking_questions
from app.core.scoring_engine import aggregate_speaking_session
from app.core.pipeline import run_pipeline

router = APIRouter()

@router.post("/evaluate")
async def evaluate(audio: UploadFile = File(...), question: str = Form("")):
    result = await run_pipeline(audio, question)
    return result

@router.post("/speaking/evaluate_all")
async def evaluate_all_speaking(
    audio_1: UploadFile = File(...),
    question_1: str = Form(...),
    audio_2: UploadFile = File(...),
    question_2: str = Form(...),
    audio_3: UploadFile = File(...),
    question_3: str = Form(...),
):
    from app.core.pipeline import run_session_pipeline
    audio_files = [
        (audio_1, question_1),
        (audio_2, question_2),
        (audio_3, question_3),
    ]
    results = await run_session_pipeline(audio_files)
    return aggregate_speaking_session(results)

@router.get("/speaking/questions")
async def get_speaking_questions():
    """
    Returns 3 speaking questions:
    - Q1: static  ("Tell me about yourself")
    - Q2, Q3: dynamically generated via LLM
    """
    dynamic = generate_speaking_questions()

    questions = [
        "Tell me about yourself.",
        dynamic[0],
        dynamic[1],
    ]

    return {"questions": questions}


@router.post("/speaking/aggregate")
async def aggregate_speaking(clip_results: list = Body(...)):
    """
    Receives an array of speaking evaluation results and aggregates them
    into a single overall score and summary.
    """
    return aggregate_speaking_session(clip_results)
