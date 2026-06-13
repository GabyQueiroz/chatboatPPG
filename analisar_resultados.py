"""
Análise dos resultados do RAGAS (resultados_ragas.csv)

Gera:
- Estatísticas descritivas das 4 métricas
- Histogramas de distribuição de cada métrica
- Heatmap de correlação entre métricas
- Boxplot comparativo das 4 métricas
- Lista das piores perguntas (menor score médio)
- Gráfico de barras das N piores perguntas por métrica
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_CSV = "resultados_ragas.csv"
OUTPUT_DIR = "."  # pasta onde salvar os gráficos

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

sns.set_theme(style="whitegrid")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Garante que as métricas são numéricas (linhas com erro/timeout viram NaN)
    for col in METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def print_summary(df: pd.DataFrame):
    print("=" * 60)
    print("RESUMO GERAL")
    print("=" * 60)
    print(f"Total de perguntas: {len(df)}")

    for col in METRICS:
        n_valid = df[col].notna().sum()
        print(f"\n--- {col} ---")
        print(f"  Avaliações válidas: {n_valid}/{len(df)}")
        if n_valid > 0:
            print(df[col].describe().round(3).to_string())

    print("\n" + "=" * 60)
    print("MÉDIA GERAL POR MÉTRICA")
    print("=" * 60)
    print(df[METRICS].mean().round(3).to_string())


def plot_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, col in zip(axes, METRICS):
        sns.histplot(df[col].dropna(), bins=20, kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribuição: {col}")
        ax.set_xlabel("Score")
        ax.set_xlim(0, 1)

    fig.suptitle("Distribuição das métricas RAGAS", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/distribuicoes_metricas.png", dpi=150)
    plt.close(fig)


def plot_boxplot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))
    melted = df[METRICS].melt(var_name="Métrica", value_name="Score")
    sns.boxplot(data=melted, x="Métrica", y="Score", ax=ax)
    sns.stripplot(data=melted, x="Métrica", y="Score", ax=ax, color="black", alpha=0.3, size=3)
    ax.set_title("Comparação geral entre métricas")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/boxplot_metricas.png", dpi=150)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame):
    corr = df[METRICS].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlação entre métricas")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/correlacao_metricas.png", dpi=150)
    plt.close(fig)


def plot_worst_questions(df: pd.DataFrame, n: int = 10):
    df = df.copy()
    df["score_medio"] = df[METRICS].mean(axis=1)

    worst = df.nsmallest(n, "score_medio")[["user_input", "score_medio"] + METRICS]

    print("\n" + "=" * 60)
    print(f"TOP {n} PIORES PERGUNTAS (menor score médio)")
    print("=" * 60)
    for _, row in worst.iterrows():
        pergunta = row["user_input"]
        pergunta_curta = pergunta if len(pergunta) <= 60 else pergunta[:57] + "..."
        print(f"\n[{row['score_medio']:.3f}] {pergunta_curta}")
        for col in METRICS:
            print(f"    {col}: {row[col]:.3f}" if pd.notna(row[col]) else f"    {col}: N/A")

    # Gráfico de barras horizontal
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.5)))
    labels = [
        q if len(q) <= 50 else q[:47] + "..."
        for q in worst["user_input"]
    ]
    sns.barplot(x=worst["score_medio"], y=labels, ax=ax, color="indianred")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score médio")
    ax.set_title(f"{n} piores perguntas (score médio das 4 métricas)")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/piores_perguntas.png", dpi=150)
    plt.close(fig)


def plot_metric_by_question_heatmap(df: pd.DataFrame, n: int = 20):
    """Heatmap mostrando todas as métricas para as N primeiras perguntas (ou as piores)."""
    df = df.copy()
    df["score_medio"] = df[METRICS].mean(axis=1)
    subset = df.nsmallest(n, "score_medio")

    labels = [
        q if len(q) <= 40 else q[:37] + "..."
        for q in subset["user_input"]
    ]

    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.35)))
    sns.heatmap(
        subset[METRICS].values,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        yticklabels=labels,
        xticklabels=METRICS,
        ax=ax,
    )
    ax.set_title(f"Métricas detalhadas — {n} piores perguntas")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/heatmap_piores_perguntas.png", dpi=150)
    plt.close(fig)


def main():
    df = load_data(INPUT_CSV)

    print_summary(df)

    plot_distributions(df)
    plot_boxplot(df)
    plot_correlation(df)
    plot_worst_questions(df, n=10)
    plot_metric_by_question_heatmap(df, n=15)

    print("\n" + "=" * 60)
    print("Gráficos salvos:")
    print("  - distribuicoes_metricas.png")
    print("  - boxplot_metricas.png")
    print("  - correlacao_metricas.png")
    print("  - piores_perguntas.png")
    print("  - heatmap_piores_perguntas.png")
    print("=" * 60)


if __name__ == "__main__":
    main()