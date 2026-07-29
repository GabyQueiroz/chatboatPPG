import hashlib
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Overlap reduzido de 300 para 150: o overlap alto gerava muitos chunks
# quase-duplicados (o mesmo trecho aparecendo em 2-3 chunks vizinhos),
# diluindo o context_precision - varios "slots" do top-k eram gastos com
# conteudo redundante em vez de informacao nova.
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 150

# Detecta o inicio de um artigo de lei/instrucao normativa (ex: "Art. 34",
# "Art.34", "Art. 4º"). Usado tanto para decidir se um documento e' uma
# instrucao normativa (varias ocorrencias) quanto para split-por-artigo.
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
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{source}|{page}|{start}|{h}"


def _is_legal_instrument(text: str) -> bool:
    """Heurística: um documento com pelo menos 3 ocorrências de 'Art. N' é
    tratado como instrução normativa/regulamento e ganha split por artigo
    em vez de split por tamanho de caractere fixo."""
    return len(ARTICLE_COUNT.findall(text)) >= 3


def _split_by_article(text: str) -> list[str]:
    parts = ARTICLE_START.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_data(documents):
    chunked_data = []

    for doc in documents:
        text = doc.page_content

        if _is_legal_instrument(text):
            # Cada artigo vira um chunk inteiro, mantendo a disposicao legal
            # completa e coesa (nao corta um artigo no meio, nao mistura o
            # fim de um artigo com o comeco de outro sem relacao).
            start_index = 0
            for article_text in _split_by_article(text):
                if len(article_text) <= CHUNK_SIZE:
                    new_doc = Document(page_content=article_text, metadata=dict(doc.metadata))
                    new_doc.metadata["start_index"] = start_index
                    chunked_data.append(new_doc)
                else:
                    # Artigo excepcionalmente longo: ainda particiona por
                    # tamanho, mas so dentro do proprio artigo (nao mistura
                    # com outros artigos).
                    sub_docs = _fallback_splitter.create_documents(
                        [article_text], metadatas=[dict(doc.metadata)]
                    )
                    chunked_data.extend(sub_docs)
                start_index += len(article_text)
        else:
            chunked_data.extend(_fallback_splitter.split_documents([doc]))

    for doc in chunked_data:
        doc.metadata["chunk_id"] = chunk_id(doc)

    return chunked_data
