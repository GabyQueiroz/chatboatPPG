import re
import unicodedata
from typing import Iterable, List, Optional

from langchain_core.documents import Document

from .reranker import rerank
from .vector_store import get_vector_store

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

_CORPUS_CACHE: Optional[List[Document]] = None


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_accents.lower()


def _normalize_query(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _expand_query(text: str) -> str:
    folded = _fold(text)
    expansions = []
    if any(term in folded for term in ["telefone", "ramal", "ligar", "contato", "email", "e-mail"]):
        expansions.append("contato email ramal telefone secretaria mestrado ppgd")
    if any(term in folded for term in ["site", "link", "instagram", "secijur", "pagina", "redes sociais"]):
        expansions.append("site oficial url secijur instagram redes sociais direito uepg")
    if any(term in folded for term in ["prazo", "periodo", "prorrog"]) and any(
        term in folded for term in ["conclusao", "concluir", "finalizacao", "terminar"]
    ):
        expansions.append(
            "grade curricular integralizada 24 vinte e quatro meses possibilidade prorrogacao 06 seis meses artigo 34 instrucao normativa 06 2024"
        )
    if "fomento" in folded:
        expansions.append("vagas de fomento agosto ano letivo coordenador colegiado instituicoes requerentes")
    if "qualificacao" in folded:
        expansions.append("exame de qualificacao artigo 40 creditos suficiencia lingua estrangeira projeto artigo")
    if "suficiencia" in folded or "lingua estrangeira" in folded:
        expansions.append("suficiencia lingua estrangeira ingles espanhol final do primeiro semestre")
    if "michigan" in folded or "ecce" in folded:
        expansions.append("michigan ecce 650 seiscentos e cinquenta pontos suficiencia lingua estrangeira")
    if "defesa" in folded and any(term in folded for term in ["versao final", "deposito", "entregar", "prazo"]):
        expansions.append("60 sessenta dias apos defesa publica versao final dissertacao trabalho final")
    if "credito" in folded and "disciplina" not in folded and any(
        term in folded for term in ["quantos", "total", "tenho", "preciso", "exigidos"]
    ):
        expansions.append("artigo 34 totalizam-se 33 trinta e tres creditos composicao curricular grade curricular")
    if "credito" in folded and "disciplina" in folded:
        expansions.append("disciplinas creditos carga horaria 45 3 tres")
    if "docencia" in folded:
        expansions.append("estagio de docencia 2 creditos")
    if any(term in folded for term in ["laboratorio", "chave", "agendamento", "reserva"]):
        expansions.append("laboratorio secretaria ppgd agendamento 7 dias antecedencia horario funcionamento")
    if "estagio" in folded and any(term in folded for term in ["lugar", "local", "instituicao", "unidade"]):
        expansions.append(
            "mais de uma unidade supervisora pluralidade dos campos de estagio termos de compromisso declaracoes de carga horaria distintas"
        )
    if not expansions:
        return text
    return f"{text} {' '.join(expansions)}"


def _extract_keywords(text: str) -> List[str]:
    words = re.findall(r"[\w@#./:-]+", _fold(text))
    return [word for word in words if len(word) >= 4 and word not in STOPWORDS]


def _contains_keywords(text: str, keywords: List[str]) -> bool:
    if not keywords:
        return False
    lowered = _fold(text)
    return any(keyword in lowered for keyword in keywords)


def _keyword_score(text: str, keywords: Iterable[str]) -> int:
    lowered = _fold(text)
    return sum(1 for keyword in keywords if keyword in lowered)


def _get_corpus_documents(vector_store) -> List[Document]:
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


def _full_corpus_keyword_search(vector_store, keywords: List[str], limit: int = 20) -> List[Document]:
    if not keywords:
        return []

    corpus = _get_corpus_documents(vector_store)
    if not corpus:
        return []

    hits = [doc for doc in corpus if _contains_keywords(doc.page_content, keywords)]
    hits.sort(key=lambda doc: _keyword_score(doc.page_content, keywords), reverse=True)
    return hits[:limit]


def _source_matches(doc, preferred_sources: Iterable[str]) -> bool:
    if not preferred_sources:
        return False
    source = _fold(str(doc.metadata.get("source", "")))
    return any(_fold(fragment) in source for fragment in preferred_sources if fragment)


def _apply_source_preference(documents: Iterable[Document], preferred_sources: Iterable[str], keywords: List[str]) -> List[Document]:
    documents = list(documents)
    if not preferred_sources:
        return documents
    return sorted(
        documents,
        key=lambda doc: (
            1 if _source_matches(doc, preferred_sources) else 0,
            _keyword_score(doc.page_content, keywords),
        ),
        reverse=True,
    )


def _dedupe_documents(documents: Iterable[Document]) -> List[Document]:
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
    preferred_sources: Optional[Iterable[str]] = None,
) -> List[Document]:
    vector_store = get_vector_store()
    normalized_query = _normalize_query(query)
    search_query = _expand_query(normalized_query)
    keywords = _extract_keywords(search_query)
    preferred_sources = tuple(preferred_sources or ())

    if use_mmr:
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
            candidates = _apply_source_preference(candidates, preferred_sources, keywords)
        else:
            candidates = _apply_source_preference(semantic_results, preferred_sources, keywords)

        return rerank(normalized_query, candidates[:candidate_pool_size], top_k=k)

    scored = vector_store.similarity_search_with_score(search_query, k=fetch_k)
    if max_distance is not None:
        filtered = [doc for doc, score in scored if score <= max_distance]
        if filtered:
            filtered = _apply_source_preference(filtered, preferred_sources, keywords)
            return rerank(normalized_query, filtered, top_k=k)

    docs = _apply_source_preference([doc for doc, _ in scored], preferred_sources, keywords)
    return rerank(normalized_query, docs, top_k=k)
