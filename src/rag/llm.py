import re

import ollama

LLM_MODEL = "gemma3:4b"

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

Exemplo de recusa:
Desculpe, não tenho informações suficientes para responder a essa pergunta.
"""

URL_QUERY_RE = re.compile(r"\b(link|site|página|pagina|instagram|rede social|redes sociais|acesso|acessar|url)\b", re.I)
URL_RE = re.compile(r"https?://[^\s)>\"]+")
EMAIL_QUERY_RE = re.compile(r"\b(e-?mail|email|correio eletrônico)\b", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
PHONE_QUERY_RE = re.compile(r"\b(telefone|ramal|ligar|contato)\b", re.I)
PHONE_RE = re.compile(r"(?:\(?\d{2}\)?\s*)?\d{4,5}[- ]?\d{4}(?:\s*/\s*\d{4})?")


def _direct_url_answer(question: str, context: str) -> str | None:
    if not URL_QUERY_RE.search(question):
        return None
    urls = URL_RE.findall(context or "")
    if not urls:
        return None

    lowered_question = question.lower()
    for url in urls:
        if "instagram" in lowered_question and "instagram" not in url.lower():
            continue
        return url.rstrip(".,;")
    return urls[0].rstrip(".,;")


def _direct_contact_answer(question: str, context: str) -> str | None:
    if EMAIL_QUERY_RE.search(question):
        emails = EMAIL_RE.findall(context or "")
        if emails:
            return emails[0].rstrip(".,;")

    if PHONE_QUERY_RE.search(question):
        phones = PHONE_RE.findall(context or "")
        if phones:
            return phones[0].strip().rstrip(".,;")

    return None


def ask_question(question: str, context: str = "") -> str:
    direct_contact = _direct_contact_answer(question, context)
    if direct_contact:
        return direct_contact

    direct_url = _direct_url_answer(question, context)
    if direct_url:
        return direct_url

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
        },
    )

    return resp["message"]["content"]
