import re
import unicodedata

import ollama

LLM_MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = """
Você é um assistente extremamente preciso sobre o Mestrado Profissional em Direito da UEPG.

Responda somente com base no CONTEXTO fornecido. Se o contexto não trouxer a informação exata, responda apenas:
"Desculpe, não tenho informações suficientes para responder a essa pergunta."

Regras:
- Responda em português do Brasil, de forma curta e direta.
- Preserve nomes próprios, números, prazos, datas, artigos e URLs exatamente como aparecem no contexto.
- Se a pergunta pedir link, site, página, Instagram, redes sociais ou "onde encontro/acesso", retorne apenas a URL exata quando houver uma URL clara no contexto.
- Para perguntas que não pedem link, não inclua URLs, salvo se elas forem parte indispensável da resposta.
- Quando houver regra geral e exceção por turma/ano, explique a regra e a exceção aplicável.
- Não invente nomes, datas, prazos, documentos, artigos ou procedimentos.
- Quando a resposta for um dado isolado (um número, uma data, um contato), não responda só o dado seco: retome brevemente o assunto perguntado na mesma frase. Exemplo: em vez de "60 horas.", responda "O estágio de imersão prático-institucional totaliza 60 horas.". Isso vale só para dar contexto à resposta curta — continue direto e sem rodeios, sem adicionar informação que não foi pedida.

Exemplo de recusa:
Desculpe, não tenho informações suficientes para responder a essa pergunta.
"""

URL_QUERY_RE = re.compile(r"\b(link|site|página|pagina|instagram|rede social|redes sociais|acesso|acessar|url)\b", re.I)
URL_RE = re.compile(r"https?://[^\s)>\"]+")
EMAIL_QUERY_RE = re.compile(r"\b(e-?mail|email|correio eletrônico)\b", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
PHONE_QUERY_RE = re.compile(r"\b(telefone|ramal|ligar|contato)\b", re.I)
PHONE_RE = re.compile(r"(?:\(?\d{2}\)?\s*)?\d{4,5}[- ]?\d{4}(?:\s*/\s*\d{4})?")


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_accents.lower()


def _direct_url_answer(question: str, context: str) -> str | None:
    if not URL_QUERY_RE.search(question):
        return None
    if not context:
        return None

    urls = URL_RE.findall(context)
    if not urls:
        return None

    lowered_question = question.lower()

    question_terms = [w for w in re.findall(r"[\wçãõáéíóúâêô]+", _fold(lowered_question)) if len(w) >= 4]
    """ folded_context = _fold(context) """
    for match in re.finditer(URL_RE, context):
        url = match.group(0)
        if "instagram" in lowered_question and "instagram" not in url.lower():
            continue
        window_start = max(0, match.start() - 120)
        window = _fold(context[window_start:match.start()])
        if any(term in window for term in question_terms if term not in ("site", "pode", "passar")):
            return url.rstrip(".,;")

    for url in urls:
        if "instagram" in lowered_question and "instagram" not in url.lower():
            continue
        return url.rstrip(".,;")
    return urls[0].rstrip(".,;")


def _direct_contact_answer(question: str, context: str) -> str | None:
    if EMAIL_QUERY_RE.search(question):
        emails = EMAIL_RE.findall(context or "")
        if emails:
            return f"O e-mail do mestrado é {emails[0].rstrip('.,;')}"

    if PHONE_QUERY_RE.search(question):
        phones = PHONE_RE.findall(context or "")
        if phones:
            return f"O ramal de contato do mestrado é {phones[0].strip().rstrip('.,;')}"

    return None


def _direct_academic_answer(question: str, context: str) -> str | None:
    folded_question = _fold(question)
    folded_context = _fold(context)
    """ Verifica se a pergunta e' sobre o total de creditos do CURSO como um
    todo (nao de uma disciplina/estagio/item especifico). """

    asks_credits = "credito" in folded_question or "creditos" in folded_question

    mentions_whole_program = any(
        term in folded_question for term in ["curso", "programa", "mestrado", "integraliza", "grade curricular"]
    )
    asks_total = mentions_whole_program and any(
        term in folded_question for term in ["tenho que ter", "preciso ter", "total", "quantos"]
    )
    has_total = "totalizam-se 33" in folded_context or "33 (trinta e tres) creditos" in folded_context

    if asks_credits and asks_total and has_total:
        return (
            "A grade curricular totaliza 33 (trinta e três) créditos: 9 de formação geral, "
            "6 de aprofundamento específico na linha de pesquisa, 6 de aprofundamento específico "
            "de livre escolha, 4 de imersão prático-institucional, 4 de discussão e disseminação "
            "do conhecimento e 4 de pesquisa/escrita acadêmica."
        )

    return None


def ask_question(question: str, context: str = "") -> str:
    direct_contact = _direct_contact_answer(question, context)
    if direct_contact:
        return direct_contact

    direct_url = _direct_url_answer(question, context)
    if direct_url:
        return direct_url

    direct_academic = _direct_academic_answer(question, context)
    if direct_academic:
        return direct_academic

    if context:
        cleaned_context = re.sub(r"[ \t]+", " ", context).strip()
        user_content = f"Contexto:\n{cleaned_context}\n\nPergunta:\n{question}"
    else:
        user_content = question

    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        options={
            "temperature": 0,
            "num_ctx": 8192,
        },
    )

    return resp["message"]["content"]
