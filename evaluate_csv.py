import pandas as pd
import time
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from src.rag.retriever import retrieve
from src.rag.llm import ask_question

def run_batch_evaluation(input_csv_path: str, output_csv_path: str):
    print(f"1. Lendo arquivo de entrada: {input_csv_path}...")
    
    try:
        df_input = pd.read_csv(input_csv_path, sep=';') 
    except FileNotFoundError:
        print(f"Erro: O arquivo {input_csv_path} não foi encontrado na raiz do projeto.")
        return

    user_inputs = []
    retrieved_contexts = []
    responses = []
    references = []

    print(f"2. Consultando o sistema RAG para {len(df_input)} perguntas...")
    start_generation = time.time()
    
    for index, row in df_input.iterrows():
        # Os nomes são case sensitive
        pergunta = str(row['pergunta'])
        referencia = str(row['referencia'])

        print(f"   -> Processando ({index+1}/{len(df_input)}): {pergunta[:40]}...")

        context_docs = retrieve(pergunta)
        context_list = [doc.page_content for doc in context_docs]
        context_text = "\n\n---\n\n".join(context_list)


        resposta_bot = ask_question(pergunta, context_text)

        user_inputs.append(pergunta)
        retrieved_contexts.append(context_list)
        responses.append(resposta_bot)
        references.append(referencia)

    gen_time = time.time() - start_generation
    print(f"   Geração concluída em {gen_time:.2f} segundos.")

    print("\n3. Inicializando juízes locais (llama3.1:8b)...")
    local_llm = ChatOllama(model="llama3.1:8b", temperature=0)
    local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    ragas_llm = LangchainLLMWrapper(local_llm)
    ragas_emb = LangchainEmbeddingsWrapper(local_embeddings)

    data = {
        "user_input": user_inputs,
        "retrieved_contexts": retrieved_contexts,
        "response": responses,
        "reference": references,
    }
    dataset = Dataset.from_dict(data)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    run_configs = RunConfig(
        max_workers=3,
        timeout=300
    )

    print("\n4. Executando avaliação Ragas em lote (isso pode levar vários minutos)...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=run_configs
    )

    print("\n5. Consolidando e exportando resultados...")
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