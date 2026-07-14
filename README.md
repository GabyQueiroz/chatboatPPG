# Chatbot Acadêmico PPGD/UEPG

Chatbot RAG para consulta aos documentos do Mestrado Profissional em Direito da UEPG.

## Requisitos

- Python 3.10+
- Ollama instalado e em execução

Modelos usados:

```powershell
ollama pull gemma3:4b
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

## Publicar no Render

Este repositório já inclui front e backend juntos:

- Frontend estático em `static/`.
- Backend FastAPI em `main.py`.
- Configuração do Render em `render.yaml`.
- Comando alternativo em `Procfile`.

No Render:

1. Crie um novo **Web Service** a partir deste repositório.
2. Use Python como ambiente.
3. Build command:

```bash
pip install -r requirements.txt
```

4. Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

5. Configure a variável de ambiente `OLLAMA_HOST` apontando para um servidor Ollama acessível pela internet.

Importante: o Render não fornece automaticamente os modelos do Ollama. O chatbot usa:

- `gemma3:4b`
- `nomic-embed-text`

Então o `OLLAMA_HOST` precisa apontar para uma instância Ollama onde esses modelos já estejam baixados. Sem isso, o site sobe, mas as respostas do chatbot não funcionam.

## Endpoints

- `GET /health`: verifica se a API está ativa.
- `GET /query?query=...`: consulta o chatbot.
- `POST /api/query`: consulta o chatbot via JSON, útil para integrações.
- `GET /update`: reprocessa os documentos e atualiza o índice vetorial.

## WhatsApp

A integração com WhatsApp é possível via Meta WhatsApp Cloud API ou Twilio. A integração deve receber mensagens por webhook e chamar o endpoint `POST /api/query`.
