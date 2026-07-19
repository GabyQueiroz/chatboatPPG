import random
import re
import unicodedata
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.rag import (
    ask_question,
    find_quick_match,
    get_vector_store,
    ingestion,
    is_insufficient_answer,
    quick_context,
    retrieve,
    sources_for_quick_match,
)
from src.rag.quick_answers import load_quick_answers

app = FastAPI(title="Chatbot Acadêmico PPGD/UEPG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

ingestion.ingest_data()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QuestionRequest(BaseModel):
    query: str
    history: list[ChatMessage] = Field(default_factory=list)


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_accents.lower()


def _tokens(text: str) -> set[str]:
    folded = _fold(text)
    return set(re.findall(r"[a-z0-9]+", folded))


def _recent_user_context(history: list[ChatMessage], limit: int = 4) -> str:
    user_messages = [item.content for item in history if item.role == "user" and item.content.strip()]
    return " ".join(user_messages[-limit:])


def _resolve_follow_up(query: str, history: list[ChatMessage]) -> tuple[str, str | None]:
    """Detecta pergunta de acompanhamento MAS nunca deve alterar a query usada
    para busca (quick match / retrieval). Serve só para decidir se vale
    reforçar o contexto conversacional que já é passado separadamente ao LLM."""
    query_tokens = _tokens(query)
    recent_context = _recent_user_context(history)
    context_tokens = _tokens(recent_context)

    if not recent_context:
        return query, None

    is_short_follow_up = len(query_tokens) <= 5
    asks_hours = bool({"hora", "horas", "carga", "horaria"} & query_tokens)
    previous_credit_topic = bool({"credito", "creditos"} & context_tokens)

    if is_short_follow_up and asks_hours and previous_credit_topic and not ({"credito", "creditos"} & query_tokens):
        resolved = "Quantas horas correspondem ao total de créditos do mestrado?"
        return resolved, query

    return query, None


def _source_label(metadata: dict) -> str:
    source = str(metadata.get("source", "documento")).replace("\\", "/")
    page = metadata.get("page")
    if page is None or page == "":
        return source
    return f"{source}, página {int(page) + 1}"


def _build_response(query: str, history: list[ChatMessage] | None = None):
    history = history or []
    resolved_query, original_follow_up = _resolve_follow_up(query, history)

    quick_match = find_quick_match(resolved_query)
    if quick_match and quick_match.mode == "direct":
        return {
            "results": quick_match.answer.answer,
            "sources": sources_for_quick_match(quick_match),
            "context": [quick_context(quick_match)],
            "answer_mode": "quick",
            "similarity": quick_match.score,
            "resolved_query": resolved_query,
            "original_query": original_follow_up or query,
        }

    context_docs = retrieve(resolved_query)
    context_text = "\n\n---\n\n".join(doc.page_content for doc in context_docs)
    if quick_match and quick_match.mode == "assist":
        context_text = f"{quick_context(quick_match)}\n\n---\n\n{context_text}"

    if history:
        conversation_context = "\n".join(
            f"{item.role}: {item.content}" for item in history[-6:] if item.content.strip()
        )
        context_text = f"Histórico recente da conversa:\n{conversation_context}\n\n---\n\n{context_text}"

    result = ask_question(resolved_query, context=context_text)
    sources = []
    seen = set()
    if quick_match and quick_match.mode == "assist":
        sources.extend(sources_for_quick_match(quick_match))
        seen.add(sources[0]["source"])

    for doc in context_docs:
        label = _source_label(doc.metadata)
        if label in seen:
            continue
        seen.add(label)
        sources.append(
            {
                "source": label,
                "excerpt": doc.page_content[:420],
            }
        )

    if is_insufficient_answer(str(result)) and quick_match and quick_match.mode == "suggest":
        result = (
            "Não encontrei uma resposta exata para essa pergunta. "
            f"Talvez você queira perguntar: \"{quick_match.answer.canonical_question}\". "
            f"Resposta relacionada: {quick_match.answer.answer}"
        )
        sources = sources_for_quick_match(quick_match) + sources

    return {
        "results": str(result),
        "sources": sources,
        "context": ([quick_context(quick_match)] if quick_match else []) + [doc.page_content for doc in context_docs],
        "answer_mode": quick_match.mode if quick_match else "rag",
        "similarity": quick_match.score if quick_match else None,
        "resolved_query": resolved_query,
        "original_query": original_follow_up or query,
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"message": "ok"}


@app.get("/update")
def update_vector_store():
    ingestion.ingest_data()
    vector_store = get_vector_store()
    total_docs = len(vector_store._collection.get()["ids"])
    return {"message": f"Vector store updated. Total documents: {total_docs}"}


@app.get("/query")
def query_vector_store(query: str):
    return _build_response(query)


@app.post("/api/query")
def query_vector_store_post(payload: QuestionRequest):
    return _build_response(payload.query, payload.history)


@app.get("/api/suggestions")
def suggestions():
    questions = []
    seen = set()
    for answer in load_quick_answers():
        question = answer.canonical_question.strip()
        folded = _fold(question)
        if question and folded not in seen:
            seen.add(folded)
            questions.append(question)

    fallback = [
        "Quantos créditos compõem a grade curricular?",
        "Quais são os requisitos para fazer a qualificação?",
        "Qual o prazo para entregar a versão final após a defesa?",
        "Quais documentos preciso para solicitar a defesa?",
        "Como funcionam as atividades complementares?",
        "Quais disciplinas são de formação geral?",
    ]
    pool = questions or fallback
    amount = min(4, len(pool))
    return {"suggestions": random.sample(pool, amount)}
