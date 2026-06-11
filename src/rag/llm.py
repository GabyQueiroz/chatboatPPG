import re
import ollama

LLM_MODEL = "phi4-mini"

SYSTEM_PROMPT = """
voce e um assistente de perguntas e respostas. use somente o contexto fornecido.

regras:
- nao use conhecimento externo.
- Responda apenas com o conteúdo exato do contexto quando for uma URL ou informação literal.
- se a informacao nao estiver claramente no contexto, responda exatamente: "desculpe, nao tenho informacoes suficientes para responder a essa pergunta."
- responda de forma direta e objetiva.
Proibido:
- Corrigir URLs
- Adicionar/remover caracteres
- "Melhorar" a formatação

Caso o contexto não seja suficiente responda com "Desculpe, não posso te responder isso"

"""
def ask_question(question: str, context: str = "") -> str:
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
            'temperature':0
        }
    )
    return resp["message"]["content"]