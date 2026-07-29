import time

import pandas as pd
from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from main import _build_response
from src.rag import is_insufficient_answer


def _answer_like_production(pergunta: str) -> tuple[str, list[str]]:
    payload = _build_response(pergunta, [])
    return str(payload.get("results", "")), list(payload.get("context", []) or [])


def run_batch_evaluation(input_csv_path: str, output_csv_path: str):
    print(f"1. Lendo arquivo de entrada: {input_csv_path}...")

    try:
        df_input = pd.read_csv(input_csv_path, sep=";")
    except FileNotFoundError:
        print(f"Erro: O arquivo {input_csv_path} nao foi encontrado na raiz do projeto.")
        return

    if "categoria" not in df_input.columns:
        df_input["categoria"] = "respondivel"

    user_inputs = []
    retrieved_contexts = []
    responses = []
    references = []
    categorias = []

    print(f"2. Consultando o sistema RAG para {len(df_input)} perguntas...")
    start_generation = time.time()

    for index, row in df_input.iterrows():
        pergunta = str(row["pergunta"])
        referencia = str(row["referencia"])
        categoria = str(row["categoria"]).strip().lower()

        print(f"   -> Processando ({index + 1}/{len(df_input)}): {pergunta[:40]}...")

        resposta_bot, context_list = _answer_like_production(pergunta)

        user_inputs.append(pergunta)
        retrieved_contexts.append(context_list)
        responses.append(resposta_bot)
        references.append(referencia)
        categorias.append(categoria)

    gen_time = time.time() - start_generation
    print(f"   Geracao concluida em {gen_time:.2f} segundos.")

    print("\n3. Calculando acuracia de recusa (fora de escopo vs respondível)...")
    refusal_rows = []
    for pergunta, resposta, categoria in zip(user_inputs, responses, categorias):
        refused = is_insufficient_answer(str(resposta))
        correto = refused if categoria == "fora_de_escopo" else not refused
        refusal_rows.append(
            {
                "pergunta": pergunta,
                "categoria": categoria,
                "resposta": resposta,
                "recusou": refused,
                "correto": correto,
            }
        )

    df_refusal = pd.DataFrame(refusal_rows)
    df_refusal.to_csv("resultados_recusas.csv", index=False, encoding="utf-8-sig")

    fora_escopo_mask = df_refusal["categoria"] == "fora_de_escopo"
    respondivel_mask = ~fora_escopo_mask

    if fora_escopo_mask.any():
        acc_fora_escopo = df_refusal.loc[fora_escopo_mask, "correto"].mean()
        print(
            f"   Recusa correta em perguntas fora de escopo: {acc_fora_escopo:.1%} "
            f"({df_refusal.loc[fora_escopo_mask, 'correto'].sum()}/{fora_escopo_mask.sum()})"
        )

    if respondivel_mask.any():
        acc_respondivel = df_refusal.loc[respondivel_mask, "correto"].mean()
        falsas_recusas = (~df_refusal.loc[respondivel_mask, "correto"]).sum()
        print(
            f"   Nao-recusa correta em perguntas respondiveis: {acc_respondivel:.1%} "
            f"({respondivel_mask.sum() - falsas_recusas}/{respondivel_mask.sum()}, {falsas_recusas} recusa(s) indevida(s))"
        )
        if falsas_recusas:
            print("   Perguntas respondiveis recusadas indevidamente:")
            for _, row in df_refusal[respondivel_mask & ~df_refusal["correto"]].iterrows():
                print(f"     - {row['pergunta'][:70]}")

    idx_respondivel = [i for i, categoria in enumerate(categorias) if categoria != "fora_de_escopo"]
    user_inputs_r = [user_inputs[i] for i in idx_respondivel]
    retrieved_contexts_r = [retrieved_contexts[i] for i in idx_respondivel]
    responses_r = [responses[i] for i in idx_respondivel]
    references_r = [references[i] for i in idx_respondivel]

    print("\n4. Inicializando juizes locais (llama3.1:8b)...")
    local_llm = ChatOllama(model="llama3.1:8b", temperature=0, config={"device": "cuda"})
    local_embeddings = OllamaEmbeddings(model="bge-m3")
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

    run_configs = RunConfig(max_workers=10, timeout=360)

    print(
        f"\n5. Executando avaliacao Ragas em lote sobre {len(user_inputs_r)} perguntas respondiveis "
        f"(as {len(user_inputs) - len(user_inputs_r)} fora de escopo entram so na acuracia de recusa)..."
    )
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=run_configs,
    )

    print("\n6. Consolidando e exportando resultados...")
    df_results = result.to_pandas()
    df_results.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    print("\n========================================================")
    print("Sucesso! Avaliacao concluida.")
    print(f"Resultados salvos em: {output_csv_path}\n")


if __name__ == "__main__":
    INPUT_FILE = "perguntas.csv"
    OUTPUT_FILE = "resultados_ragas.csv"
    run_batch_evaluation(INPUT_FILE, OUTPUT_FILE)
