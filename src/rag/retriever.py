import re
import unicodedata
from typing import Iterable, List, Optional

from .vector_store import get_vector_store

STOPWORDS = {
    "a", "ao", "aos", "as", "ate", "com", "como", "da", "das", "de", "do",
    "dos", "e", "em", "eu", "me", "minha", "meu", "na", "nas", "no", "nos",
    "o", "os", "ou", "para", "por", "qual", "quais", "que", "sao", "se",
    "sobre", "um", "uma",
}


def _normalize_query(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _expand_query(text: str) -> str:
    folded = _fold(text)
    expansions = []
    if "prazo" in folded and any(term in folded for term in ["conclusao", "concluir", "finalizacao", "terminar"]):
        expansions.append("grade curricular integralizada 24 vinte e quatro meses possibilidade 06 seis meses artigo 34")
    if "qualificacao" in folded:
        expansions.append("exame de qualificação artigo 40 créditos suficiência língua estrangeira projeto artigo")
    if "suficiencia" in folded or "lingua estrangeira" in folded:
        expansions.append("suficiência língua estrangeira inglês espanhol final do 1º semestre")
    if "defesa" in folded and any(term in folded for term in ["versao final", "deposito", "entregar", "prazo"]):
        expansions.append("60 sessenta dias após defesa pública versão final dissertação trabalho final")
    if not expansions:
        return text
    return f"{text} {' '.join(expansions)}"


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_accents.lower()


def _extract_keywords(text: str) -> List[str]:
    words = re.findall(r"[\w@#./:-]+", _fold(text))
    return [w for w in words if len(w) >= 4 and w not in STOPWORDS]


def _contains_keywords(text: str, keywords: List[str]) -> bool:
    if not keywords:
        return False
    lowered = _fold(text)
    return any(kw in lowered for kw in keywords)


def _keyword_score(text: str, keywords: Iterable[str]) -> int:
    lowered = _fold(text)
    return sum(1 for kw in keywords if kw in lowered)


def _dedupe_documents(documents: Iterable) -> List:
    seen = set()
    deduped = []
    for doc in documents:
        key = doc.metadata.get("chunk_id") or (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.page_content[:80],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def retrieve(
    query: str,
    k: int = 10,
    fetch_k: int = 80,
    use_mmr: bool = True,
    lambda_mult: float = 0.35,
    max_distance: Optional[float] = 0.6,
) -> List:
    vector_store = get_vector_store()
    normalized_query = _normalize_query(query)
    search_query = _expand_query(normalized_query)
    keywords = _extract_keywords(search_query)

    if use_mmr:
        semantic_results = vector_store.max_marginal_relevance_search(
            search_query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )
        if not keywords:
            return semantic_results

        scored = vector_store.similarity_search_with_score(search_query, k=max(fetch_k, 200))
        keyword_hits = [doc for doc, _ in scored if _contains_keywords(doc.page_content, keywords)]
        combined = _dedupe_documents(keyword_hits[:k] + semantic_results + [doc for doc, _ in scored[:k]])
        combined.sort(key=lambda doc: _keyword_score(doc.page_content, keywords), reverse=True)
        return combined[:k]

    scored = vector_store.similarity_search_with_score(search_query, k=fetch_k)
    if max_distance is not None:
        filtered = [doc for doc, score in scored if score <= max_distance]
        if filtered:
            return filtered[:k]

    return [doc for doc, _ in scored[:k]]
