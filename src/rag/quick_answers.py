import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from openpyxl import load_workbook

ROOT_DIR = Path(__file__).resolve().parents[2]
QUESTION_BANK_PATH = ROOT_DIR / "base_perguntas_respostas_ppgd.xlsx"
FAQ_PATH = ROOT_DIR / "static" / "FAQ.md"

DIRECT_THRESHOLD = 0.86
ASSIST_THRESHOLD = 0.70
SUGGEST_THRESHOLD = 0.58

STOPWORDS = {
    "a", "ao", "aos", "as", "ate", "com", "como", "da", "das", "de", "do",
    "dos", "e", "em", "eu", "me", "minha", "meu", "na", "nas", "no", "nos",
    "o", "os", "ou", "para", "por", "qual", "quais", "que", "sao", "se",
    "sobre", "um", "uma", "voce", "pode", "gostaria", "saber",
}

IMPORTANT_TERMS = {
    "credito", "creditos", "disciplina", "disciplinas", "prazo", "prazos",
    "defesa", "qualificacao", "suficiencia", "email", "telefone", "ramal",
    "instagram", "fomento", "dissertacao", "atividades", "complementares",
    "matricula", "aproveitamento", "lingua", "estrangeira",
}


@dataclass(frozen=True)
class QuickAnswer:
    question: str
    canonical_question: str
    answer: str
    source: str
    category: str
    intent: str
    response_type: str
    fallback_answer: str


@dataclass(frozen=True)
class QuickMatch:
    answer: QuickAnswer
    score: float
    mode: str


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_accents.lower()


def _normalize(text: str) -> str:
    text = _fold(text)
    text = re.sub(r"[^a-z0-9@./:-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

#Stem corta sufixos para melhorar busca semântica da mesma variação

def _stem(word: str) -> str:
    suffixes = ("ador", "adora", "acao", "ação", "ativo", "ativa", "mente", "s")
    for suffix in sorted(suffixes, key=len, reverse=True):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _tokens(text: str) -> set[str]:
    raw_tokens = {word for word in _normalize(text).split() if len(word) >= 3 and word not in STOPWORDS}
    return {_stem(word) for word in raw_tokens}


def _clean_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text or "")
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _score(query: str, candidate: str) -> float:
    normalized_query = _normalize(query)
    normalized_candidate = _normalize(candidate)
    if not normalized_query or not normalized_candidate:
        return 0.0
    if normalized_query == normalized_candidate:
        return 1.0

    seq_score = SequenceMatcher(None, normalized_query, normalized_candidate).ratio()
    query_tokens = _tokens(query)
    candidate_tokens = _tokens(candidate)
    if not query_tokens or not candidate_tokens:
        return seq_score

    query_important = query_tokens & IMPORTANT_TERMS
    candidate_important = candidate_tokens & IMPORTANT_TERMS
    if query_important and not (query_important & candidate_important):
        seq_score = min(seq_score, 0.45)

    overlap = len(query_tokens & candidate_tokens)
    containment = overlap / max(1, len(query_tokens))
    jaccard = overlap / max(1, len(query_tokens | candidate_tokens))
    return max(seq_score, containment * 0.9, jaccard)


def _shares_important_term(query: str, answer: QuickAnswer) -> bool:
    query_terms = _tokens(query) & IMPORTANT_TERMS
    answer_terms = (_tokens(answer.question) | _tokens(answer.canonical_question) | _tokens(answer.answer)) & IMPORTANT_TERMS
    return bool(query_terms and query_terms & answer_terms)


def _parse_faq_entries() -> list[QuickAnswer]:
    if not FAQ_PATH.exists():
        return []

    text = FAQ_PATH.read_text(encoding="utf-8")
    matches = list(re.finditer(r"^###\s+(.+)$", text, flags=re.MULTILINE))
    entries = []

    for index, match in enumerate(matches):
        question = re.sub(r"^\d+\.\s*", "", match.group(1)).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        source_match = re.search(r"Fonte[s]?:\s*(.+)", block, flags=re.IGNORECASE)
        source = source_match.group(1).strip() if source_match else "static/FAQ.md"
        answer = re.split(r"Fonte[s]?:", block, maxsplit=1, flags=re.IGNORECASE)[0]
        answer = _clean_markdown(" ".join(line.strip() for line in answer.splitlines() if line.strip()))
        if question and answer:
            entries.append(
                QuickAnswer(
                    question=question,
                    canonical_question=question,
                    answer=answer,
                    source=source,
                    category="faq",
                    intent=_normalize(question).replace(" ", "_")[:48],
                    response_type="faq",
                    fallback_answer="Desculpe, não tenho informações suficientes para responder a essa pergunta.",
                )
            )
    return entries


def _load_from_workbook() -> list[QuickAnswer]:
    if not QUESTION_BANK_PATH.exists():
        return []

    workbook = load_workbook(QUESTION_BANK_PATH, read_only=True, data_only=True)
    sheet = workbook["Perguntas_Respostas"]
    headers = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
    index = {name: position for position, name in enumerate(headers)}
    entries = []

    for row in sheet.iter_rows(min_row=2, values_only=True):
        if not row or not row[index["pergunta_variacao"]]:
            continue
        entries.append(
            QuickAnswer(
                question=str(row[index["pergunta_variacao"]]),
                canonical_question=str(row[index["pergunta_base"]]),
                answer=_clean_markdown(str(row[index["resposta_esperada"]])),
                source=str(row[index["fonte"]]),
                category=str(row[index["categoria"]]),
                intent=str(row[index["intencao"]]),
                response_type=str(row[index["tipo_resposta"]]),
                fallback_answer=_clean_markdown(str(row[index["resposta_quando_nao_entender"]])),
            )
        )
    return entries


@lru_cache(maxsize=1)
def load_quick_answers() -> tuple[QuickAnswer, ...]:
    entries = _load_from_workbook()
    if not entries:
        entries = _parse_faq_entries()
    return tuple(entries)


def find_quick_match(query: str, threshold: float = SUGGEST_THRESHOLD) -> Optional[QuickMatch]:
    query_tokens = _tokens(query)
    asks_credit_hours = (
        bool({"hora", "horas", "horaria", "carga"} & query_tokens)
        and bool({"credito", "creditos"} & query_tokens)
    )
    if asks_credit_hours:
        for answer in load_quick_answers():
            folded_answer = _fold(answer.answer)
            folded_question = _fold(answer.canonical_question)
            if ("33" in folded_answer and "credito" in folded_answer) or (
                "creditos compoem a grade curricular" in folded_question
            ):
                enriched_answer = QuickAnswer(
                    question=answer.question,
                    canonical_question="Quantas horas correspondem ao total de créditos do mestrado?",
                    answer=(
                        "Considerando a equivalência apresentada nas disciplinas do Programa, em que 3 créditos "
                        "correspondem a 45 horas, cada crédito equivale a 15 horas. Assim, os 33 créditos da grade "
                        "curricular correspondem a 495 horas."
                    ),
                    source=answer.source,
                    category=answer.category,
                    intent="horas_creditos",
                    response_type=answer.response_type,
                    fallback_answer=answer.fallback_answer,
                )
                return QuickMatch(answer=enriched_answer, score=1.0, mode="direct")

    asks_total_credits = (
        "creditos" in query_tokens or "credito" in query_tokens
    ) and bool({"quantos", "total", "integralizar", "preciso", "ter"} & query_tokens)
    if asks_total_credits:
        for answer in load_quick_answers():
            folded_answer = _fold(answer.answer)
            folded_question = _fold(answer.canonical_question)
            if ("33" in folded_answer and "credito" in folded_answer) or (
                "creditos compoem a grade curricular" in folded_question
            ):
                enriched_answer = QuickAnswer(
                    question=answer.question,
                    canonical_question=answer.canonical_question,
                    answer=(
                        "A grade curricular totaliza 33 (trinta e três) créditos: 9 de formação geral, "
                        "6 de aprofundamento específico na linha de pesquisa, 6 de aprofundamento específico "
                        "de livre escolha, 4 de imersão prático-institucional, 4 de discussão e disseminação "
                        "do conhecimento e 4 de pesquisa/escrita acadêmica."
                    ),
                    source=answer.source,
                    category=answer.category,
                    intent=answer.intent,
                    response_type=answer.response_type,
                    fallback_answer=answer.fallback_answer,
                )
                return QuickMatch(answer=enriched_answer, score=1.0, mode="direct")

    best_answer = None
    best_score = 0.0

    for answer in load_quick_answers():
        score = max(_score(query, answer.question), _score(query, answer.canonical_question) * 0.96)
        if score > best_score:
            best_score = score
            best_answer = answer

    if not best_answer or best_score < threshold:
        return None

    if best_score >= DIRECT_THRESHOLD or (best_score >= 0.58 and _shares_important_term(query, best_answer)):
        mode = "direct"
    elif best_score >= ASSIST_THRESHOLD:
        mode = "assist"
    else:
        mode = "suggest"

    return QuickMatch(answer=best_answer, score=round(best_score, 3), mode=mode)


def quick_context(match: QuickMatch) -> str:
    answer = match.answer
    return (
        "Resposta provável vinda da base rápida de FAQ/planilha:\n"
        f"Pergunta semelhante: {answer.canonical_question}\n"
        f"Resposta esperada: {answer.answer}\n"
        f"Fonte: {answer.source}\n"
        f"Similaridade: {match.score}"
    )


def is_insufficient_answer(text: str) -> bool:
    return "não tenho informações suficientes" in _fold(text or "")


def sources_for_quick_match(match: QuickMatch) -> list[dict]:
    answer = match.answer
    return [
        {
            "source": f"Base rápida: {answer.source}",
            "excerpt": f"Pergunta semelhante: {answer.canonical_question} (similaridade {match.score})",
        }
    ]
