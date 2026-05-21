import ollama

LLM_MODEL = "phi4-mini"

SYSTEM_PROMPT = """
você é um assistente de perguntas e respostas. Responda às perguntas com base no contexto fornecido. Se o contexto não for suficiente para responder à pergunta, responda com "Desculpe, não tenho informações suficientes para responder a essa pergunta.""
"""
def ask_question(question: str, context: str = "") -> str:
    user_content = f"Contexto:\n{context}\n\nPergunta:\n{question}" if context else question

    

    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return resp["message"]["content"]