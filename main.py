from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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


class QuestionRequest(BaseModel):
    query: str


def _source_label(metadata: dict) -> str:
    source = str(metadata.get("source", "documento")).replace("\\", "/")
    page = metadata.get("page")
    if page is None or page == "":
        return source
    return f"{source}, página {int(page) + 1}"


def _build_response(query: str):
    quick_match = find_quick_match(query)
    if quick_match and quick_match.mode == "direct":
        return {
            "results": quick_match.answer.answer,
            "sources": sources_for_quick_match(quick_match),
            "context": [quick_context(quick_match)],
            "answer_mode": "quick",
            "similarity": quick_match.score,
        }

    context_docs = retrieve(query)
    context_text = "\n\n---\n\n".join(doc.page_content for doc in context_docs)
    if quick_match and quick_match.mode == "assist":
        context_text = f"{quick_context(quick_match)}\n\n---\n\n{context_text}"

    result = ask_question(query, context=context_text)
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
    return _build_response(payload.query)
