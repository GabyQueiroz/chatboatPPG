import pandas as pd
import time
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from src.rag import (
    ask_question,
    find_quick_match,
    is_insufficient_answer,
    quick_context,
    retrieve,
)


def _answer_like_production(pergunta: str) -> tuple[str, list[str]]:
    """Reproduz a mesma decisao do _build_response() do main.py: primeiro
    tenta o atalho de FAQ (quick_match), depois cai para retrieval + LLM.
    Sem isso, a avaliacao testa um pipeline (so retrieve+ask_question) que
    nao e o que a aplicacao real usa em producao."""
    quick_match = find_quick_match(pergunta)

    if quick_match and quick_match.mode == "direct":
        return quick_match.answer.answer, [quick_context(quick_match)]

    context_docs = retrieve(pergunta)
    context_list = [doc.page_content for doc in context_docs]
    context_text = "\n\n---\n\n".join(context_list)

    if quick_match and quick_match.mode == "assist":
        context_text = f"{quick_context(quick_match)}\n\n---\n\n{context_text}"

    resposta = ask_question(pergunta, context=context_text)

    if is_insufficient_answer(str(resposta)) and quick_match and quick_match.mode == "suggest":
        resposta = (
            "Não encontrei uma resposta exata para essa pergunta. "
            f"Talvez você queira perguntar: \"{quick_match.answer.canonical_question}\". "
            f"Resposta relacionada: {quick_match.answer.answer}"
        )

    return str(resposta), context_list

def run_batch_evaluation(input_csv_path: str, output_csv_path: str):
    print(f"1. Lendo arquivo de entrada: {input_csv_path}...")

    try:
        df_input = pd.read_csv(input_csv_path, sep=';')
    except FileNotFoundError:
        print(f"Erro: O arquivo {input_csv_path} não foi encontrado na raiz do projeto.")
        return

    if 'categoria' not in df_input.columns:
        df_input['categoria'] = 'respondivel'

    user_inputs = []
    retrieved_contexts = []
    responses = []
    references = []
    categorias = []

    print(f"2. Consultando o sistema RAG para {len(df_input)} perguntas...")
    start_generation = time.time()

    for index, row in df_input.iterrows():
        # Os nomes são case sensitive
        pergunta = str(row['pergunta'])
        referencia = str(row['referencia'])
        categoria = str(row['categoria']).strip().lower()

        print(f"   -> Processando ({index+1}/{len(df_input)}): {pergunta[:40]}...")

        resposta_bot, context_list = _answer_like_production(pergunta)

        user_inputs.append(pergunta)
        retrieved_contexts.append(context_list)
        responses.append(resposta_bot)
        references.append(referencia)
        categorias.append(categoria)

    gen_time = time.time() - start_generation
    print(f"   Geração concluída em {gen_time:.2f} segundos.")

    # --------------------------------------------------------------
    # Perguntas fora de escopo devem ser recusadas;
    # perguntas respondíveis NÃO devem ser recusadas. O RAGAS penaliza
    # recusas corretas como se fossem falha de geração, o que
    # distorce as 4 métricas principais - por isso essa checagem roda
    # separada, e o RAGAS só é calculado sobre as perguntas respondíveis.
    # --------------------------------------------------------------
    print("\n3. Calculando acurácia de recusa (fora de escopo vs respondível)...")
    refusal_rows = []
    for pergunta, resposta, categoria in zip(user_inputs, responses, categorias):
        refused = is_insufficient_answer(str(resposta))
        if categoria == 'fora_de_escopo':
            correto = refused
        else:
            correto = not refused
        refusal_rows.append({
            'pergunta': pergunta,
            'categoria': categoria,
            'resposta': resposta,
            'recusou': refused,
            'correto': correto,
        })
    df_refusal = pd.DataFrame(refusal_rows)
    df_refusal.to_csv('resultados_recusas.csv', index=False, encoding='utf-8-sig')

    fora_escopo_mask = df_refusal['categoria'] == 'fora_de_escopo'
    respondivel_mask = ~fora_escopo_mask

    if fora_escopo_mask.any():
        acc_fora_escopo = df_refusal.loc[fora_escopo_mask, 'correto'].mean()
        print(f"   Recusa correta em perguntas fora de escopo: {acc_fora_escopo:.1%} "
              f"({df_refusal.loc[fora_escopo_mask, 'correto'].sum()}/{fora_escopo_mask.sum()})")

    if respondivel_mask.any():
        acc_respondivel = df_refusal.loc[respondivel_mask, 'correto'].mean()
        falsas_recusas = (~df_refusal.loc[respondivel_mask, 'correto']).sum()
        print(f"   Não-recusa correta em perguntas respondíveis: {acc_respondivel:.1%} "
              f"({respondivel_mask.sum() - falsas_recusas}/{respondivel_mask.sum()}, "
              f"{falsas_recusas} recusa(s) indevida(s))")
        if falsas_recusas:
            print("   Perguntas respondíveis recusadas indevidamente:")
            for _, r in df_refusal[respondivel_mask & ~df_refusal['correto']].iterrows():
                print(f"     - {r['pergunta'][:70]}")

    idx_respondivel = [i for i, c in enumerate(categorias) if c != 'fora_de_escopo']
    user_inputs_r = [user_inputs[i] for i in idx_respondivel]
    retrieved_contexts_r = [retrieved_contexts[i] for i in idx_respondivel]
    responses_r = [responses[i] for i in idx_respondivel]
    references_r = [references[i] for i in idx_respondivel]

    print("\n4. Inicializando juízes locais (llama3.1:8b)...")
    local_llm = ChatOllama(model="llama3.1:8b", temperature=0)
    local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    ragas_llm = LangchainLLMWrapper(local_llm)
    ragas_emb = LangchainEmbeddingsWrapper(local_embeddings)

    data = {
        "user_input": user_inputs_r,
        "retrieved_contexts": retrieved_contexts_r,
        "response": responses_r,
        "reference": references_r,
    }
    dataset = Dataset.from_dict(data)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    run_configs = RunConfig(
        max_workers=2,
        timeout=360
    )

    print(f"\n5. Executando avaliação Ragas em lote sobre {len(user_inputs_r)} perguntas respondíveis "
          f"(as {len(user_inputs) - len(user_inputs_r)} fora de escopo entraram só na acurácia de recusa acima)...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=run_configs
    )

    print("\n6. Consolidando e exportando resultados...")
    df_results = result.to_pandas()

    df_results.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    
    print(f"\n========================================================")
    print(f"Sucesso! Avaliação concluída.")
    print(f"Resultados salvos em: {output_csv_path}\n")

    #print(result)

if __name__ == "__main__":
    INPUT_FILE = "perguntas.csv"
    OUTPUT_FILE = "resultados_ragas.csv"
    
    run_batch_evaluation(INPUT_FILE, OUTPUT_FILE)