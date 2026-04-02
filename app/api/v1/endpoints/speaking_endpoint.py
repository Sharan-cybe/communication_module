from fastapi import APIRouter, Body
from app.services.speaking.question_generator import generate_speaking_questions
from app.core.scoring_engine import aggregate_speaking_session

router = APIRouter()


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
