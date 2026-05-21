import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings

def run_evaluation():
    print("1 - Inicialzando modelo juíz local.")
    
    local_llm = ChatOllama(model="llama3.1:8b", temperature=0)
    local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    ragas_llm = LangchainLLMWrapper(local_llm)
    ragas_emb = LangchainEmbeddingsWrapper(local_embeddings)

    print("2 - Estruturando teste.")

    # Pergunta padrão para teste
    # O reference seria o Ground Truth(resposta ideal)
    test_data = {
        "user_input": [
            "Qual o nome do coordenador do programa de Pós-Graduação de direito da UEPG?"
        ],
        "retrieved_contexts": [
            [
            "PARECER HOMOLOGADO \nPortaria n° 576, publicada no D.O.U. de 9/7/2020, Seção 1, Pág. 127. \nAntonio Freitas – 0483  Documento assinado eletronicamente nos termos da legislação vigente \n \nMINISTÉRIO DA EDUCAÇÃO \nCONSELHO NACIONAL DE EDUCAÇÃO \n \nINTERESSADA: Fundação Coordenação de Aperfeiçoamento de Pessoal de \nNível Superior (Capes) \nUF: DF \nASSUNTO: Reconhecimento dos programas de pós -graduação stricto sensu  (Mestrado e \nDoutorado), recomendados pelo Conselho Técnico -Científico da Educação Superior (CTC -\nES) da Capes, na reunião realizada de 11 a 15 de maio de 2020 (194ª Reunião). \nRELATOR: Antonio de Araujo Freitas Júnior \nPROCESSO Nº: 23001.000483/2020-90 \nPARECER CNE/CES Nº: \n350/2020 \nCOLEGIADO: \nCES \nAPROVADO EM: \n17/6/2020 \n \nI – RELATÓRIO \n \nO presente processo trata de solicitação da Fundação Coordenação de \nAperfeiçoamento de Pessoal de Nível Superior (Capes) à Câmara de Educação Superior do",
            "a aprovação do Conselho de Administração na reunião do dia 21 de setembro de 2020; e,\n \nconsiderando mais, os termos do expediente autuado no Sistema Eletrônico de Informações\nda Universidade Estadual de Ponta Grossa, onde se consubstanciou no Processo no 19.000000268-4, ad\nreferendum do Conselho Universitário,\n  \nR E S O L V E:\n \nArt. 1o Aprovar o Regulamento do Programa Profissional de Pós-Graduação Stricto sensu\nem Direito – Curso de Mestrado Profissional, da Universidade Estadual de Ponta Grossa – UEPG, na forma\ndo Anexo que passa a integrar este ato legal.\n \nArt. 2o Esta Portaria entrará em vigor na data de sua publicação. Reitoria da Universidade\nEstadual de Ponta Grossa.\n \nPonta Grossa, 20 de outubro de 2020.",
            "Universidade Estadual de Ponta Grossa  \n \n \nANEXO DA RESOLUÇÃO CEPE No 2022.13                                                                                                              FL. 21 DE 22 \nArt. 47 Os conceitos finais obtidos pelo disce nte deverão constar do histórico \nescolar. \n \nArt. 48 Disciplinas com características específicas, tais como: Seminários, \nEstágio Orientado de Docência, Atividades Programadas de Pesquisa Orientada e \nOrientação de Dissertação e Trabalho Final, poderão, quando  previsto em projeto, se valer \nde outro instrumento de avaliação, que considere apenas duas categorias: \n \nI - S – Suficiente - aprovação com direito a crédito; \n \nII - NS – Não Suficiente - reprovação sem direito a crédito. \n \nArt. 49 A frequência no PPGD é obrigatória e cabe ao Colegiado do Programa \nautorizar trabalhos especiais ou estágios fora dos lugares indicados na programação. \n \nArt. 5 0 A cada período letivo o discente deverá requerer matrícula nas"
            ]
        ],
        "response": [
            "**ANÁLISE DO DOCUMENTO E IDENTIFICAÇÃO DO COORDONADOR**\n\n**Fundamentos Legais:** O presente parecer homologado (Portaria n° 576/2020) estabelece o reconhecimento do Programa de Pós-Graduação *Stricto Sensu* em Direito da Universidade Estadual de Ponta Grossa (UEPG), autorizado através de resolução do Conselho de Educação Superior (CES). A análise da estrutura do documento, especificamente os artigos 47, 48 e 49, demonstra a regulamentação do funcionamento do programa, incluindo critérios de avaliação e frequência.\n\n**Análise Doutrinária:** A Portaria em questão, por ser um ato normativo da Capes/CNE, alinha-se com a regulamentação padrão de programas de pós-graduação stricto sensu, buscando otimizar a avaliação e o reconhecimento da qualidade acadêmica. A utilização de conceitos como \"S\" e \"NS\" na avaliação das disciplinas visa a formalizar o processo de avaliação e garantir a padronização dos critérios.\n\n**Jurisprudência:** A ausência de referência direta a jurisprudência específica neste contexto é esperada, dada a natureza do documento como ato normativo de reconhecimento institucional. No entanto, a própria estrutura da Capes/CNE, baseada em critérios de avaliação e reconhecimento, pode ser considerada um equivalente jurídico a uma jurisprudência consolidada na área da educação superior.\n\n**Criticas:** A Portaria em si não apresenta espaço para críticas institucional, sendo um ato de reconhecimento. A estrutura do documento, no entanto, permite a análise do processo de avaliação do programa, identificando possíveis pontos de melhoria na metodologia de avaliação.\n\n**Conclusão Técnica:** O parecer homologado (CNE/CES n° 350/2020) formaliza o reconhecimento do Programa de Pós-Graduação *Stricto Sensu* em Direito da Universidade Estadual de Ponta Grossa (UEPG). A análise dos artigos do Anexo revela a existência de normas específicas sobre avaliação, frequência e autorização de atividades complementares.\n\n**Sugestões:** É fundamental que a UEPG mantenha uma comunicação transparente com a Capes/CNE sobre quaisquer alterações no programa de pós-graduação, buscando garantir a contínua avaliação e reconhecimento da qualidade do curso.\n\n**Identificação do Coordenador:**\n\nO documento em questão não especifica o nome do coordenador do programa de Pós-Graduação em Direito da UEPG em 2020. A análise do corpo normativo, por si só, não permite a identificação desse responsável. Seria necessário acessar outros documentos, como a ata da reunião do Conselho Técnico-Científico da UEPG, ou outros registros da universidade, para obter essa informação.\n"
        ],
        "reference": [
            "Foi designado o professor Eliezer Gomes da Silva para as função de Coordenador do Programa Profissional de Pós-Graduação Strcto sensu em Diretivo da Universidade Estadual de Ponta Grossa - UEPG"
        ],
    }

    dataset = Dataset.from_dict(test_data)

    # Definindo métricas para calcular
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    print("3 - Executando Avalição das Métricas com Ragas(Demoraaado)")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb
    )

    print("Resultados da avaliação:")
    print("Quanto mais próximo de zero melhor\n" +
    "Se um ou mais resultados forem NaN, o mais provável é que o modelo juíz precisa ser mais forte.")
    df_results = result.to_pandas()

    print(df_results[["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]])

if __name__ == "__main__":
    run_evaluation()