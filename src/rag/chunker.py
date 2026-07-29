import hashlib
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1400
CHUNK_OVERLAP = 150

ARTICLE_START = re.compile(r"(?=Art\.?\s*\d+[ºo°]?[\.\s])")
ARTICLE_COUNT = re.compile(r"Art\.?\s*\d+[ºo°]?[\.\s]")

_fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    add_start_index=True,
)


def chunk_id(doc):
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", ""))
    start = str(doc.metadata.get("start_index", ""))
    text = " ".join(doc.page_content.split()).strip().lower()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{source}|{page}|{start}|{digest}"


def _is_legal_instrument(text: str) -> bool:
    return len(ARTICLE_COUNT.findall(text)) >= 3


def _split_by_article(text: str) -> list[str]:
    parts = ARTICLE_START.split(text)
    return [part.strip() for part in parts if part.strip()]


def chunk_data(documents):
    chunked_data = []

    for doc in documents:
        text = doc.page_content

        if _is_legal_instrument(text):
            start_index = 0
            for article_text in _split_by_article(text):
                if len(article_text) <= CHUNK_SIZE:
                    new_doc = Document(page_content=article_text, metadata=dict(doc.metadata))
                    new_doc.metadata["start_index"] = start_index
                    chunked_data.append(new_doc)
                else:
                    chunked_data.extend(
                        _fallback_splitter.create_documents([article_text], metadatas=[dict(doc.metadata)])
                    )
                start_index += len(article_text)
        else:
            chunked_data.extend(_fallback_splitter.split_documents([doc]))

    for doc in chunked_data:
        doc.metadata["chunk_id"] = chunk_id(doc)

    return chunked_data
