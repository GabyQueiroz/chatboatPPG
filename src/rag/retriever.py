import re
import unicodedata
from typing import Iterable, List, Optional

from langchain_core.documents import Document

from .vector_store import get_vector_store
from .reranker import rerank

STOPWORDS = {
    "a", "ao", "aos", "as", "ate", "com", "como", "da", "das", "de", "do",
    "dos", "e", "em", "eu", "me", "minha", "meu", "na", "nas", "no", "nos",
    "o", "os", "ou", "para", "por", "qual", "quais", "que", "sao", "se",
    "sobre", "um", "uma",
    "pode", "pod", "poderia", "podem", "gostaria", "queria", "quero",
    "preciso", "precisar", "necessito", "consigo", "consegue", "conseguir",
    "saber", "sabe", "fazer", "favor", "obrigado", "obrigada", "voce",
    "voces", "informar", "dizer", "existe", "tem", "esta", "isso", "aqui",
    "ali", "bem", "muito", "mais", "menos", "assim", "entao", "tambem",
    "ainda", "apenas", "algum", "alguma", "algumas", "alguns", "outro",
    "outra", "outros", "outras", "onde", "quando", "porque", "pois",
    "isto", "aquele", "aquela", "este", "essa", "esse", "vou", "vai",
    "vamos", "nao", "sim", "todo", "toda", "todos", "todas", "cada",
    "meus", "minhas", "seu", "sua", "seus", "suas", "eles", "elas", "ele",
    "ela", "aquilo", "tudo", "nada", "algo", "alguem", "mim", "lhe",
    "dele", "dela", "deles", "delas", "passar", "mandar", "manda",
    "falar", "fala", "explicar", "explica",
}


def _normalize_query(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _expand_query(text: str) -> str:
    folded = _fold(text)
    expansions = []
    if any(t in folded for t in ["prazo", "periodo", "prorrog"]) and any(term in folded for term in ["conclusao", "concluir", "finalizacao", "terminar"]):
        expansions.append("grade curricular integralizada 24 vinte e quatro meses possibilidade prorrogação 06 seis meses artigo 34 instrução normativa 06 2024")
    if "qualificacao" in folded:
        expansions.append("exame de qualificação artigo 40 créditos suficiência língua estrangeira projeto artigo")
    if "suficiencia" in folded or "lingua estrangeira" in folded:
        expansions.append("suficiência língua estrangeira inglês espanhol final do 1º semestre")
    if "defesa" in folded and any(term in folded for term in ["versao final", "deposito", "entregar", "prazo"]):
        expansions.append("60 sessenta dias após defesa pública versão final dissertação trabalho final")
    if "credito" in folded and "disciplina" not in folded and any(term in folded for term in ["quantos", "total", "tenho", "preciso", "exigidos"]):
        expansions.append("artigo 34 totalizam-se 33 trinta e três créditos composição curricular grade curricular")
    if "credito" in folded and "disciplina" in folded:
        expansions.append("disciplinas créditos carga horária 45 3 três")
    if "estagio" in folded and any(t in folded for t in ["lugar", "local", "instituicao", "unidade"]):
        expansions.append("mais de uma unidade supervisora pluralidade dos campos de estágio termos de compromisso declarações de carga horária distintas")
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


_CORPUS_CACHE: Optional[List[Document]] = None


def _get_corpus_documents(vector_store) -> List[Document]:
    """Carrega e cacheia (por processo) todos os chunks do Chroma, para
    permitir busca textual completa e calculo de frequencia de termos sem
    reescanear o banco a cada pergunta."""
    global _CORPUS_CACHE
    if _CORPUS_CACHE is not None:
        return _CORPUS_CACHE

    try:
        raw = vector_store.get(include=["documents", "metadatas"])
    except Exception:
        return []

    contents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or [{}] * len(contents)
    _CORPUS_CACHE = [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(contents, metadatas)
        if content
    ]
    return _CORPUS_CACHE


""" def _keyword_document_frequencies(corpus: List[Document], keywords: List[str]) -> dict:
    \""" Para cada keyword, conta em quantos chunks do corpus ela aparece.
    Usado para pesar termos raros (ex: "fomento", "secijur", "teap") mais
    do que termos genericos que aparecem em quase todo documento (ex:
    "vagas", "disciplina", "programa") - sem isso, um chunk irrelevante que
    so bate num termo comum pode empatar ou superar o chunk certo, que so
    bate no termo especifico da pergunta. \"""
    freqs = {kw: 0 for kw in keywords}
    for doc in corpus:
        lowered = _fold(doc.page_content)
        for kw in keywords:
            if kw in lowered:
                freqs[kw] += 1
    return freqs
 """

def _full_corpus_keyword_search(vector_store, keywords: List[str], limit: int = 20) -> List:
    """Necessario porque termos raros/siglas (ex: "TEAP", nomes de
    exames, numeros de instrucao normativa)
    """
    if not keywords:
        return []

    corpus = _get_corpus_documents(vector_store)
    if not corpus:
        return []

    hits = [doc for doc in corpus if _contains_keywords(doc.page_content, keywords)]
    hits.sort(key=lambda doc: _keyword_score(doc.page_content, keywords), reverse=True)
    return hits[:limit]


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

    # Gate de confiança: antes de tudo, olha só o melhor match semântico. Se
    # nem o resultado MAIS próximo bate um limiar minimo de similaridade,
    # a pergunta provavelmente não tem resposta no corpus (ex: "O que é a
    # UEPG?" - genérico demais, sempre "acha" algo por keyword/MMR mesmo sem
    # ter nada realmente relevante). Sem esse gate, o sistema nunca recusava
    # por falta de relevância de verdade - só quando a busca vinha vazia.
    """ if max_distance is not None:
        try:
            top_match = vector_store.similarity_search_with_score(search_query, k=1)
        except Exception:
            top_match = []
        if top_match and top_match[0][1] > max_distance:
            return [] """

    if use_mmr:
        # Pool de candidatos maior que k: reunimos mais opcoes do que o
        # necessario (MMR + keyword hits) e deixamos o cross-encoder (rerank)
        # escolher os k melhores de verdade no final, em vez de confiar so na
        # heuristica de contagem de keyword para o corte final.
        candidate_pool_size = min(max(k * 2, 20), fetch_k)
        semantic_results = vector_store.max_marginal_relevance_search(
            search_query,
            k=candidate_pool_size,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )

        if keywords:
            keyword_hits = _full_corpus_keyword_search(vector_store, keywords, limit=max(k, 10))
            candidates = _dedupe_documents(keyword_hits[:candidate_pool_size] + semantic_results + keyword_hits)
            candidates.sort(key=lambda doc: _keyword_score(doc.page_content, keywords), reverse=True)
        else:
            candidates = semantic_results

        # Reranking com cross-encoder real (query, chunk) por cima do pool de
        # candidatos - ver reranker.py. Se o modelo nao estiver disponivel,
        # rerank() degrada de volta para os top-k do ranking anterior, sem
        # quebrar o fluxo.
        return rerank(normalized_query, candidates[:candidate_pool_size], top_k=k)

    scored = vector_store.similarity_search_with_score(search_query, k=fetch_k)
    if max_distance is not None:
        filtered = [doc for doc, score in scored if score <= max_distance]
        if filtered:
            return rerank(normalized_query, filtered, top_k=k)

    return rerank(normalized_query, [doc for doc, _ in scored], top_k=k)
