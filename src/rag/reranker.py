"""Reranking com cross-encoder real, para substituir a heuristica de
keyword-score (que ja mostrou varias vezes ser fragil: palavras genericas
demais, sinonimos faltando, termos raros sem peso certo).

Um cross-encoder recebe o par (pergunta, chunk) e da uma nota de relevancia
de verdade (nao e' so similaridade de embeddings pre-computados como no MMR -
ele "le" os dois textos juntos), o que costuma ser bem mais preciso para
escolher, entre os candidatos, quais realmente respondem a pergunta.

Modelo escolhido: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 - multilingue
(inclui portugues), pequeno o suficiente para rodar em CPU sem virar gargalo.
"""

from functools import lru_cache
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
    except Exception as exc:  # modelo nao baixado, sem internet, etc.
        print(f"[rerank] não foi possível carregar o cross-encoder ({exc}); seguindo sem reranking.")
        _reranker_failed = True
        _reranker = None

    return _reranker


def rerank(query: str, documents: List, top_k: int) -> List:
    """Reordena `documents` pela relevância real (query, chunk) via
    cross-encoder e devolve os `top_k` melhores. Se o modelo não estiver
    disponível (falha ao carregar), devolve os documentos como vieram, sem
    quebrar o fluxo — reranking é uma melhoria opcional, nunca um requisito
    para o sistema funcionar."""
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
