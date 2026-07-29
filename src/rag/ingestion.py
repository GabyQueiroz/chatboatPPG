import json
import os

from ..rag import load_docs, chunk_data, get_vector_store

MANIFEST_PATH = os.path.join("db", "chroma", "manifest.json")
INGESTION_VERSION = 3  # bump: chunker.py agora divide por artigo + overlap reduzido (300->150)


def build_manifest():
    entries = []
    sources = [
        ("docs/pdfs", ".pdf", "pdf"),
        ("docs/texts", ".txt", "text"),
    ]

    for base_dir, ext, file_type in sources:
        if not os.path.exists(base_dir):
            continue

        for root, _, files in os.walk(base_dir):
            for name in files:
                if not name.lower().endswith(ext):
                    continue

                full_path = os.path.join(root, name)
                stat = os.stat(full_path)
                rel_path = os.path.relpath(full_path, ".").replace("\\", "/")
                entries.append(
                    {
                        "path": rel_path,
                        "size": stat.st_size,
                        "mtime": int(stat.st_mtime),
                        "type": file_type,
                    }
                )

    entries.sort(key=lambda item: item["path"])
    return {"version": INGESTION_VERSION, "entries": entries}


def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return None

    with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest(manifest):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=True, indent=2)


def ingest_data():
    manifest = build_manifest()
    previous_manifest = load_manifest()

    if previous_manifest == manifest:
        print("Manifest unchanged. Skipping ingestion.")
        return

    documents = load_docs()
    chunked_data = chunk_data(documents)

    # Defesa contra IDs duplicados: pode acontecer (ex: um PDF sendo
    # carregado/mesclado mais de uma vez, ou dois artigos com texto
    # byte-a-byte identico) e o Chroma rejeita o upsert inteiro se achar
    # qualquer duplicata. Deduplicamos aqui por chunk_id, mantendo a
    # primeira ocorrencia, em vez de deixar o processo de ingestao quebrar.
    seen_ids = set()
    deduped_chunks = []
    for doc in chunked_data:
        chunk_id = doc.metadata["chunk_id"]
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        deduped_chunks.append(doc)

    if len(deduped_chunks) != len(chunked_data):
        print(f"Aviso: {len(chunked_data) - len(deduped_chunks)} chunk(s) duplicado(s) removido(s) antes da ingestão.")

    chunked_data = deduped_chunks
    ids = [doc.metadata["chunk_id"] for doc in chunked_data]

    vector_store = get_vector_store()

    existing_ids = vector_store._collection.get()["ids"]
    if existing_ids:
        vector_store._collection.delete(ids=existing_ids)

    if chunked_data:
        vector_store.add_documents(chunked_data, ids=ids)

    save_manifest(manifest)
    total_docs = len(vector_store._collection.get()["ids"])
    print(f"Ingested {len(chunked_data)} documents. Total documents in vector store: {total_docs}")


if __name__ == "__main__":
    ingest_data()
