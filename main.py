from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.rag import ask_question, get_vector_store, ingestion, retrieve

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
    context_docs = retrieve(query)
    context_text = "\n\n---\n\n".join(doc.page_content for doc in context_docs)
    result = ask_question(query, context=context_text)
    sources = []
    seen = set()
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
    return {"results": str(result), "sources": sources, "context": [doc.page_content for doc in context_docs]}


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
