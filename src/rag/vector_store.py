import os
from threading import Lock

from langchain_chroma import Chroma

from .embeddings import get_embeddings

_VECTOR_STORE = None
_LOCK = Lock()
PERSIST_DIRECTORY = os.getenv("CHROMA_DIR", "db/chroma")


def get_vector_store():
    global _VECTOR_STORE

    if _VECTOR_STORE is None:
        with _LOCK:
            if _VECTOR_STORE is None:
                _VECTOR_STORE = Chroma(
                    collection_name="documents",
                    persist_directory=PERSIST_DIRECTORY,
                    embedding_function=get_embeddings(),
                )
    return _VECTOR_STORE


def get_vector_store_document_count() -> int:
    vector_store = get_vector_store()
    return len(vector_store._collection.get().get("ids", []))
