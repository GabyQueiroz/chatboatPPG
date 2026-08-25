import os

from langchain_ollama import OllamaEmbeddings

EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
EMBED_OLLAMA_HOST = os.getenv("EMBED_OLLAMA_HOST", "http://127.0.0.1:11434").strip()
EMBED_OLLAMA_API_KEY = os.getenv("EMBED_OLLAMA_API_KEY", "").strip()


def get_embeddings():
    kwargs = {"model": EMBED_MODEL, "base_url": EMBED_OLLAMA_HOST}
    if EMBED_OLLAMA_API_KEY:
        kwargs["client_kwargs"] = {
            "headers": {"Authorization": f"Bearer {EMBED_OLLAMA_API_KEY}"}
        }
    return OllamaEmbeddings(**kwargs)
