# Chatbot Acadêmico PPGD/UEPG

Chatbot RAG para consulta aos documentos do Mestrado Profissional em Direito da UEPG.

## Requisitos

- Python 3.10+
- Ollama instalado e em execução

Modelos usados:

```powershell
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

## Instalação

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Rodar localmente

```powershell
.\run.ps1
```

Depois acesse:

- Chat: http://127.0.0.1:8000
- FAQ: http://127.0.0.1:8000/static/faq.html
- Swagger/API: http://127.0.0.1:8000/docs


## Endpoints

- `GET /health`: verifica se a API está ativa.
- `GET /query?query=...`: consulta o chatbot.
- `POST /api/query`: consulta o chatbot via JSON, útil para integrações.
- `GET /update`: reprocessa os documentos e atualiza o índice vetorial.

## Base rápida + RAG

O chatbot usa três camadas em conjunto:

1. **Base rápida**: consulta primeiro `base_perguntas_respostas_ppgd.xlsx`, gerada a partir do FAQ e dos documentos. Quando encontra uma pergunta igual ou muito parecida, responde sem chamar o modelo.
2. **FAQ**: usa `static/FAQ.md` como fonte textual auxiliar para perguntas frequentes.
3. **RAG completo**: quando não há correspondência rápida forte, busca nos documentos em `docs/`, recupera contexto no Chroma e chama o modelo Ollama.

Para regenerar a planilha de perguntas e respostas:

```powershell
python scripts/generate_question_bank.py
```

O arquivo gerado precisa permanecer no repositório para o deploy usar a camada rápida.

## WhatsApp

A integração com WhatsApp é possível via Meta WhatsApp Cloud API ou Twilio. A integração deve receber mensagens por webhook e chamar o endpoint `POST /api/query`.
