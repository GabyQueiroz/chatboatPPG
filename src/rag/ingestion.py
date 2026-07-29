import json
import os

from ..rag import chunk_data, get_vector_store, load_docs

MANIFEST_PATH = os.path.join("db", "chroma", "manifest.json")
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

    if deduped_chunks:
        vector_store.add_documents(deduped_chunks, ids=ids)

    save_manifest(manifest)
    total_docs = len(vector_store._collection.get()["ids"])
    print(f"Ingested {len(deduped_chunks)} documents. Total documents in vector store: {total_docs}")


if __name__ == "__main__":
    ingest_data()
