import os

from langchain_ollama import OllamaEmbeddings

EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "").strip()


def get_embeddings():
    kwargs = {"model": EMBED_MODEL}
    if OLLAMA_HOST:
        kwargs["base_url"] = OLLAMA_HOST
    return OllamaEmbeddings(**kwargs)
