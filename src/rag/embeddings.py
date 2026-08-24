import os

from langchain_ollama import OllamaEmbeddings

EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "").strip()
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()


def get_embeddings():
    kwargs = {"model": EMBED_MODEL}
    if OLLAMA_HOST:
        kwargs["base_url"] = OLLAMA_HOST
    if OLLAMA_API_KEY:
        kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}"}}
    return OllamaEmbeddings(**kwargs)
