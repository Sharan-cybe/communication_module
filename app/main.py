from fastapi import FastAPI
from app.api.v1.endpoints import evaluation

app = FastAPI()

app.include_router(evaluation.router, prefix="/api/v1")