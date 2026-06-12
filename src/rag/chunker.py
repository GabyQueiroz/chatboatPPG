import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_id(doc):
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", ""))
    start = str(doc.metadata.get("start_index", ""))
    text = " ".join(doc.page_content.split()).strip().lower()
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{source}|{page}|{start}|{h}"

def chunk_data(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1400,
                                                    chunk_overlap=300,
                                                    separators=["\n\n", "\n", ". ", " ", ""], add_start_index=True)
    chunked_data = text_splitter.split_documents(documents)
    for doc in chunked_data:
        doc.metadata["chunk_id"] = chunk_id(doc)
    return chunked_data