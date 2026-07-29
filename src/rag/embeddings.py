from langchain_ollama import OllamaEmbeddings

#EMBED_MODEL = "nomic-embed-text"
EMBED_MODEL = "bge-m3"

def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL)