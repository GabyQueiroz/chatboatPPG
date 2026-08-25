import json
import os
import time

from ..rag import chunk_data, get_vector_store, load_docs

PERSIST_DIRECTORY = os.getenv("CHROMA_DIR", "db/chroma")
MANIFEST_PATH = os.path.join(PERSIST_DIRECTORY, "manifest.json")
INGEST_BATCH_SIZE = max(1, int(os.getenv("INGEST_BATCH_SIZE", "8")))
INGESTION_VERSION = 3


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

    seen_ids = set()
    deduped_chunks = []
    for doc in chunked_data:
        chunk_id = doc.metadata["chunk_id"]
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        deduped_chunks.append(doc)

    if len(deduped_chunks) != len(chunked_data):
        removed = len(chunked_data) - len(deduped_chunks)
        print(f"{removed} chunk(s) duplicado(s) removido(s) antes da ingestão.")

    ids = [doc.metadata["chunk_id"] for doc in deduped_chunks]
    vector_store = get_vector_store()

    existing_ids = vector_store._collection.get()["ids"]
    if existing_ids:
        vector_store._collection.delete(ids=existing_ids)

    total_chunks = len(deduped_chunks)
    total_batches = (total_chunks + INGEST_BATCH_SIZE - 1) // INGEST_BATCH_SIZE
    print(
        f"Iniciando ingestão de {total_chunks} chunks "
        f"em {total_batches} lote(s) de até {INGEST_BATCH_SIZE}.",
        flush=True,
    )

    for batch_number, start in enumerate(
        range(0, total_chunks, INGEST_BATCH_SIZE), start=1
    ):
        batch_docs = deduped_chunks[start:start + INGEST_BATCH_SIZE]
        batch_ids = ids[start:start + INGEST_BATCH_SIZE]
        started_at = time.perf_counter()
        vector_store.add_documents(batch_docs, ids=batch_ids)
        completed_chunks = min(start + len(batch_docs), total_chunks)
        elapsed_seconds = time.perf_counter() - started_at
        percentage = (completed_chunks / total_chunks * 100) if total_chunks else 100
        print(
            f"Ingestão: {completed_chunks}/{total_chunks} chunks "
            f"({percentage:.1f}%) | lote {batch_number}/{total_batches} "
            f"em {elapsed_seconds:.1f}s",
            flush=True,
        )

    save_manifest(manifest)
    total_docs = len(vector_store._collection.get()["ids"])
    print(
        f"Ingestão concluída: {total_chunks} chunks processados. "
        f"Total no banco vetorial: {total_docs}",
        flush=True,
    )


if __name__ == "__main__":
    ingest_data()
