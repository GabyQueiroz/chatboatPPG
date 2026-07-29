import re
import unicodedata


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_accents.lower()


INTENT_PATTERNS = {
    "contact": (
        r"\b(contato|telefone|ramal|ligar|email|e-mail|correio eletronico)\b",
    ),
    "url": (
        r"\b(site|link|pagina|página|instagram|rede social|redes sociais|url|acessar|acesso)\b",
    ),
    "funding": (
        r"\b(fomento|bolsa|vagas de fomento)\b",
    ),
    "deadline": (
        r"\b(prazo|prorrog|integraliza|conclusao|conclusão|terminar|ate quando|até quando)\b",
    ),
    "proficiency": (
        r"\b(suficiencia|suficiência|lingua estrangeira|língua estrangeira|michigan|ecce|teap)\b",
    ),
    "lab": (
        r"\b(laboratorio|laboratório|lab|chave|reserva|agendamento)\b",
    ),
    "internship_teaching": (
        r"\b(estagio de docencia|estágio de docência|docencia|docência)\b",
    ),
}


INTENT_SOURCES = {
    "contact": ("docs/texts/contatos.txt", "static/FAQ.md"),
    "url": ("docs/texts/contatos.txt", "docs/texts/redes_sociais.txt", "static/FAQ.md"),
    "funding": ("IN-Normativa-12-2025-Vagas-de-Fomento-1.pdf", "static/FAQ.md"),
    "deadline": (
        "Instrucao_Normativa_n._6_2024__sobre_contagem_de_prazos_para_conclusao_do_Mestrado___24_06_2024.pdf",
        "Regulamento-do-Programa-de-Mestrado-Profissional-em-Direito",
        "docs/texts/estrutura_curricular.txt",
        "static/FAQ.md",
    ),
    "proficiency": ("Instrucao-Normativa-n°-10-suficiencia-em-linguas.pdf", "static/FAQ.md"),
    "lab": ("Proposta-Ordem-de-Sevic", "docs/texts/infraestrutura.txt", "static/FAQ.md"),
    "internship_teaching": (
        "static/FAQ.md",
        "base_perguntas_respostas_ppgd.xlsx",
    ),
}


def detect_intents(query: str, history_text: str = "") -> tuple[str, ...]:
    text = _fold(f"{query} {history_text}")
    found = []
    for intent, patterns in INTENT_PATTERNS.items():
        if any(re.search(pattern, text, flags=re.I) for pattern in patterns):
            found.append(intent)
    return tuple(found)


def preferred_source_fragments(query: str, history_text: str = "", answer_source: str = "") -> tuple[str, ...]:
    intents = detect_intents(query, history_text)
    preferred = []
    for intent in intents:
        preferred.extend(INTENT_SOURCES.get(intent, ()))

    source_text = answer_source.strip().strip("`")
    if source_text:
        preferred.append(source_text)

    deduped = []
    seen = set()
    for item in preferred:
        folded = _fold(item)
        if folded in seen:
            continue
        seen.add(folded)
        deduped.append(item)
    return tuple(deduped)


def should_expand_with_history(query: str) -> bool:
    folded = _fold(query)
    vague_terms = (
        "la",
        "lá",
        "ali",
        "isso",
        "essa",
        "esse",
        "essas",
        "esses",
        "aquele",
        "aquela",
        "aquelas",
        "aqueles",
    )
    token_count = len(re.findall(r"[a-z0-9]+", folded))
    return token_count <= 5 or any(term in folded.split() for term in vague_terms)
