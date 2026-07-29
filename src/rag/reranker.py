from typing import List

RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

_reranker = None
_reranker_failed = False


def _get_reranker():
    global _reranker, _reranker_failed
    if _reranker is not None or _reranker_failed:
        return _reranker

    try:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    except Exception as exc:
        print(f"[rerank] não foi possível carregar o modelo ({exc}); seguindo sem reranking.")
        _reranker_failed = True
        _reranker = None

    return _reranker


def rerank(query: str, documents: List, top_k: int) -> List:
    if not documents:
        return documents

    reranker = _get_reranker()
    if reranker is None:
        return documents[:top_k]

    pairs = [(query, doc.page_content) for doc in documents]
    try:
        scores = reranker.predict(pairs)
    except Exception as exc:
        print(f"[rerank] erro ao pontuar candidatos ({exc}); seguindo sem reranking.")
        return documents[:top_k]

    ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]
