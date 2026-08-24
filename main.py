import random
import re
import os
import threading
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
    get_vector_store_document_count,
    ingestion,
    is_insufficient_answer,
    quick_context,
    retrieve,
    sources_for_quick_match,
)
from src.rag.intents import preferred_source_fragments, should_expand_with_history
from src.rag.quick_answers import load_quick_answers
from src.rag.structured_answers import resolve_structured_answer

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

AUTO_INGEST = os.getenv("AUTO_INGEST", "true").strip().lower() in {"1", "true", "yes", "on"}

_INGESTION_STATE = {
    "running": False,
    "ready": False,
    "error": "",
}


def _mark_ready_from_store() -> None:
    try:
        _INGESTION_STATE["ready"] = get_vector_store_document_count() > 0
    except Exception:
        _INGESTION_STATE["ready"] = False


def _ingest_in_background() -> None:
    if _INGESTION_STATE["running"]:
        return

    def worker():
        _INGESTION_STATE["running"] = True
        _INGESTION_STATE["error"] = ""
        try:
            ingestion.ingest_data()
            _INGESTION_STATE["ready"] = get_vector_store_document_count() > 0
        except Exception as exc:
            _INGESTION_STATE["error"] = str(exc)
            _INGESTION_STATE["ready"] = False
        finally:
            _INGESTION_STATE["running"] = False

    threading.Thread(target=worker, daemon=True).start()


_mark_ready_from_store()
if AUTO_INGEST and not _INGESTION_STATE["ready"]:
    _ingest_in_background()


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


def _recent_user_context(history: list[ChatMessage], limit: int = 4) -> str:
    user_messages = [item.content for item in history if item.role == "user" and item.content.strip()]
    return " ".join(user_messages[-limit:])


def _resolve_follow_up(query: str, history: list[ChatMessage]) -> tuple[str, str]:
    history_context = _recent_user_context(history, limit=2)
    if history_context and should_expand_with_history(query):
        return f"{query} {history_context}".strip(), history_context
    return query, history_context


def _source_label(metadata: dict) -> str:
    source = str(metadata.get("source", "documento")).replace("\\", "/")
    page = metadata.get("page")
    if page is None or page == "" or page == "merged":
        return source
    return f"{source}, página {int(page) + 1}"


def _collect_sources(context_docs, quick_match=None) -> list[dict]:
    sources = []
    seen = set()

    if quick_match:
        for item in sources_for_quick_match(quick_match):
            if item["source"] in seen:
                continue
            seen.add(item["source"])
            sources.append(item)

    for doc in context_docs:
        label = _source_label(doc.metadata)
        if label in seen:
            continue
        seen.add(label)
        sources.append({"source": label, "excerpt": doc.page_content[:420]})

    return sources


def _retrieve_context_docs(query: str, history: list[ChatMessage], answer_source: str = "", k: int = 8):
    search_query, history_hint = _resolve_follow_up(query, history)
    preferred_sources = preferred_source_fragments(query, history_hint, answer_source=answer_source)
    return retrieve(search_query, k=k, preferred_sources=preferred_sources)


def _build_response(query: str, history: list[ChatMessage] | None = None):
    history = history or []
    structured_answer = resolve_structured_answer(query)
    if structured_answer:
        return {
            "results": structured_answer.answer,
            "sources": [{"source": structured_answer.source, "excerpt": structured_answer.context[:420]}],
            "context": [structured_answer.context],
            "answer_mode": "structured",
            "similarity": 1.0,
            "resolved_query": query,
            "original_query": query,
        }

    quick_match = find_quick_match(query)

    if quick_match and quick_match.mode == "direct":
        context_docs = _retrieve_context_docs(
            f"{quick_match.answer.canonical_question} {query}",
            history,
            answer_source=quick_match.answer.source,
            k=5,
        )
        return {
            "results": quick_match.answer.answer,
            "sources": _collect_sources(context_docs, quick_match=quick_match),
            "context": [quick_context(quick_match)] + [doc.page_content for doc in context_docs],
            "answer_mode": "quick",
            "similarity": quick_match.score,
            "resolved_query": query,
            "original_query": query,
        }

    if not _INGESTION_STATE["ready"]:
        status_message = (
            "A base documental ainda está sendo preparada para consulta. "
            "Tente novamente em alguns instantes."
        )
        if _INGESTION_STATE["error"]:
            status_message = (
                "A base documental não ficou pronta para consulta. "
                "Verifique a configuração do Ollama e atualize a base novamente."
            )
        return {
            "results": status_message,
            "sources": [],
            "context": [],
            "answer_mode": "status",
            "similarity": None,
            "resolved_query": query,
            "original_query": query,
        }

    context_docs = _retrieve_context_docs(
        quick_match.answer.canonical_question if quick_match and quick_match.mode == "assist" else query,
        history,
        answer_source=quick_match.answer.source if quick_match else "",
        k=8,
    )
    context_text = "\n\n---\n\n".join(doc.page_content for doc in context_docs)

    if quick_match and quick_match.mode == "assist":
        context_text = f"{quick_context(quick_match)}\n\n---\n\n{context_text}" if context_text else quick_context(quick_match)

    if history:
        conversation_context = "\n".join(
            f"{item.role}: {item.content}" for item in history[-6:] if item.content.strip()
        )
        context_text = f"Histórico recente da conversa:\n{conversation_context}\n\n---\n\n{context_text}" if context_text else conversation_context

    result = ask_question(query, context=context_text)

    if is_insufficient_answer(str(result)) and quick_match and quick_match.mode == "suggest":
        suggestion_docs = _retrieve_context_docs(quick_match.answer.canonical_question, history, answer_source=quick_match.answer.source, k=4)
        return {
            "results": (
                "Não encontrei uma resposta exata para essa pergunta. "
                f'Talvez você queira perguntar: "{quick_match.answer.canonical_question}". '
                f"Resposta relacionada: {quick_match.answer.answer}"
            ),
            "sources": _collect_sources(suggestion_docs, quick_match=quick_match),
            "context": [quick_context(quick_match)] + [doc.page_content for doc in suggestion_docs],
            "answer_mode": "suggest",
            "similarity": quick_match.score,
            "resolved_query": query,
            "original_query": query,
        }

    return {
        "results": str(result),
        "sources": _collect_sources(context_docs, quick_match=quick_match if quick_match and quick_match.mode == "assist" else None),
        "context": ([quick_context(quick_match)] if quick_match and quick_match.mode == "assist" else []) + [doc.page_content for doc in context_docs],
        "answer_mode": quick_match.mode if quick_match else "rag",
        "similarity": quick_match.score if quick_match else None,
        "resolved_query": query,
        "original_query": query,
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "message": "ok",
        "ready": _INGESTION_STATE["ready"],
        "running": _INGESTION_STATE["running"],
        "error": _INGESTION_STATE["error"],
        "documents": get_vector_store_document_count() if _INGESTION_STATE["ready"] else 0,
    }


@app.get("/api/status")
def status():
    return health()


@app.get("/update")
def update_vector_store():
    if _INGESTION_STATE["running"]:
        return {"message": "A atualização já está em andamento.", "ready": _INGESTION_STATE["ready"]}
    _ingest_in_background()
    return {"message": "Atualização da base iniciada.", "ready": _INGESTION_STATE["ready"]}


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
