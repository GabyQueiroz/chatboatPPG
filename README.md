# Chatbot Acadêmico PPGD/UEPG

Chatbot RAG para consulta aos documentos do Mestrado Profissional em Direito da UEPG.

## Requisitos

- Python 3.10+
- Ollama instalado e em execução

Modelos usados:

```powershell
ollama pull qwen2.5:7b
ollama pull bge-m3
```

## Instalação

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Para deploy, use preferencialmente:

```powershell
.\.venv\Scripts\python -m pip install -r requirements-runtime.txt
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
- `GET /api/status`: mostra se a base documental já está pronta para consulta.
- `GET /query?query=...`: consulta o chatbot.
- `POST /api/query`: consulta o chatbot via JSON, útil para integrações.
- `GET /update`: inicia a reindexação em segundo plano.

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

## Deploy

### O que foi preparado

- geração e embeddings configuráveis de forma independente;
- indexação automática em segundo plano via `AUTO_INGEST`;
- separação entre dependências de produção e de avaliação;
- endpoint de status para saber se a base já está pronta.

### Variáveis de ambiente

- `LLM_OLLAMA_HOST`: URL do Ollama usado para geração. Para Ollama Cloud: `https://ollama.com`.
- `LLM_OLLAMA_API_KEY`: chave de acesso do Ollama Cloud para geração.
- `LLM_MODEL`: modelo gerador. Padrão: `qwen2.5:7b`.
- `EMBED_OLLAMA_HOST`: URL do Ollama usado para embeddings. Padrão: `http://127.0.0.1:11434`.
- `EMBED_OLLAMA_API_KEY`: chave opcional para embeddings remotos.
- `EMBED_MODEL`: modelo de embeddings. Padrão: `bge-m3`.
- `AUTO_INGEST`: `true` ou `false`. Quando `true`, a base é preparada automaticamente ao subir.
- `CHROMA_DIR`: pasta de persistência do índice vetorial. Padrão: `db/chroma`.
- `INGEST_BATCH_SIZE`: quantidade de fragmentos vetorizados por lote. Use `1` em VPS com 1 GB de RAM.
- `ENABLE_ADMIN_ENDPOINTS`: mantenha `false` em produção; protege `/update` e `/api/ollama-status`.

### Render

O projeto já possui `render.yaml`, mas o link público só funcionará bem se o serviço tiver acesso a um Ollama executando em outro lugar. O Render gratuito não é o melhor lugar para hospedar também o Ollama, porque:

- a instância gratuita entra em sleep;
- não há GPU;
- o disco não é persistente no plano free;
- modelos grandes tendem a deixar o startup lento.

Para uma implantação com LLM na nuvem e embeddings locais, use:

1. apontar `LLM_OLLAMA_HOST` para `https://ollama.com` e configurar `LLM_OLLAMA_API_KEY`;
2. manter `EMBED_OLLAMA_HOST=http://127.0.0.1:11434` e instalar somente o modelo `bge-m3` na VPS;
3. usar um diretório persistente em `CHROMA_DIR`;
4. aguardar o endpoint `/api/status` indicar `ready: true`.

### VPS

Se a ideia for ter um link mais estável, a melhor opção é uma VPS com:

- esta API FastAPI;
- Ollama no mesmo servidor;
- persistência do diretório `db/chroma`.

Nesse cenário, a VPS não executa o modelo gerador, mas ainda precisa ter memória
suficiente para o modelo de embeddings durante a ingestão e as consultas.
