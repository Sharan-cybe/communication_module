from fastapi import APIRouter
from app.services.speaking.question_generator import generate_speaking_questions

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
