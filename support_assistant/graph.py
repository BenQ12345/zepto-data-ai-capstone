from __future__ import annotations

import json
import os
from typing import Literal, TypedDict

import chromadb
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
from langgraph.graph import END, StateGraph

from prompt_template import build_prompt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "zepto_policy"
MODEL_NAME = "all-MiniLM-L6-v2"
KEYWORDS = [
    "delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"
]


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class GraphState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved: list[dict]
    response: dict


class AssistantGraph:
    def __init__(self) -> None:
        self.mock = os.getenv("MOCK_LLM", "1") != "0"
        self.embedding_model = SentenceTransformer(MODEL_NAME)
        client = chromadb.PersistentClient(path=DB_DIR)
        self.collection = client.get_collection(COLLECTION_NAME)
        self._groq = None
        if not self.mock:
            from langchain_groq import ChatGroq
            self._groq = ChatGroq(
                model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                temperature=0,
                api_key=os.environ["GROQ_API_KEY"],
            )

    def classify_intent(self, state: GraphState) -> GraphState:
        query = state["query"]
        if self.mock:
            q = query.lower()
            intent = "policy_question" if any(k in q for k in KEYWORDS) else "general_question"
        else:
            # Optional real-LLM branch; routing still returns one of the two allowed labels.
            prompt = (
                "Classify the user query as exactly policy_question or general_question. "
                "Return only the label. Query: " + query
            )
            raw = self._groq.invoke(prompt).content.strip().lower()
            intent = "policy_question" if "policy_question" in raw else "general_question"
        return {**state, "intent": intent}

    def retrieve_and_answer(self, state: GraphState) -> GraphState:
        query = state["query"]
        q_embedding = self.embedding_model.encode([query], normalize_embeddings=True).tolist()
        result = self.collection.query(
            query_embeddings=q_embedding,
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        distances = result["distances"][0]
        retrieved = [
            {
                "id": metas[i].get("document_id", "unknown"),
                "text": docs[i],
                "distance": distances[i],
            }
            for i in range(len(docs))
        ]
        if self.mock:
            snippet = retrieved[0]["text"][:200]
            response = AnswerResponse(
                answer=f"Based on the retrieved context: {snippet}",
                sources=[x["id"] for x in retrieved],
                confidence=1.0,
            ).model_dump()
        else:
            context = "\n\n".join(f"[{x['id']}] {x['text']}" for x in retrieved)
            response = self._generate_structured_real(query, context, [x["id"] for x in retrieved])
        return {**state, "retrieved": retrieved, "response": response}

    def _generate_structured_real(self, query: str, context: str, source_ids: list[str]) -> dict:
        last_error = ""
        for attempt in range(3):
            prompt = build_prompt(context, query)
            if last_error:
                prompt += f"\nCorrection required: your previous output failed schema validation: {last_error}. Return JSON only."
            raw = self._groq.invoke(prompt).content
            try:
                obj = AnswerResponse.model_validate_json(raw)
                return obj.model_dump()
            except ValidationError as exc:
                last_error = str(exc)
        return AnswerResponse(
            answer="Error: the real LLM response could not be validated after 3 attempts.",
            sources=source_ids,
            confidence=0.0,
        ).model_dump()

    def direct_answer(self, state: GraphState) -> GraphState:
        if self.mock:
            response = AnswerResponse(
                answer="I can only answer questions about Zepto policies right now.",
                sources=[],
                confidence=1.0,
            ).model_dump()
        else:
            response = self._generate_structured_real(state["query"], "", [])
        return {**state, "response": response}

    @staticmethod
    def route(state: GraphState) -> str:
        return "retrieve_and_answer" if state.get("intent") == "policy_question" else "direct_answer"

    def compile(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("classify_intent", self.classify_intent)
        workflow.add_node("retrieve_and_answer", self.retrieve_and_answer)
        workflow.add_node("direct_answer", self.direct_answer)
        workflow.set_entry_point("classify_intent")
        workflow.add_conditional_edges(
            "classify_intent",
            self.route,
            {
                "retrieve_and_answer": "retrieve_and_answer",
                "direct_answer": "direct_answer",
            },
        )
        workflow.add_edge("retrieve_and_answer", END)
        workflow.add_edge("direct_answer", END)
        return workflow.compile()

    def ask(self, query: str) -> AnswerResponse:
        graph = self.compile()
        result = graph.invoke({"query": query})
        return AnswerResponse.model_validate(result["response"])
