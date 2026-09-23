from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from graph import AnswerResponse, AssistantGraph


class AskRequest(BaseModel):
    query: str


assistant = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global assistant
    assistant = AssistantGraph()
    yield


app = FastAPI(title="Zepto Policy Support Assistant", version="1.0.0", lifespan=lifespan)


@app.post("/ask", response_model=AnswerResponse)
def ask(request: AskRequest) -> AnswerResponse:
    assert assistant is not None
    return assistant.ask(request.query)
