import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analisar_resultados():
    """
    Função que carrega e analisa o arquivo final de previsões.
    """
    try:
        # Tenta carregar o arquivo de resultados
        submission_df = pd.read_csv('titanic_report_rna.csv')
    except FileNotFoundError:
        print("Erro: O arquivo 'titanic_report_rna.csv' não foi encontrado.")
        print("Certifique-se de que este script está na mesma pasta que o arquivo de resultados.")
        return

    print("--- Análise do Arquivo de Resultados Finais ---")
    print(f"\nO arquivo contém {len(submission_df)} previsões para os passageiros do conjunto de teste.")
    
    print("\nPrimeiras 10 previsões geradas pelo modelo:")
    print(submission_df.head(10))

    # Contar o número de previsões para cada classe
    prediction_counts = submission_df['Survived'].value_counts()
    prediction_rate = submission_df['Survived'].value_counts(normalize=True) * 100

    print("\nContagem Geral das Previsões:")
    print(f"Não Sobreviveu (0): {prediction_counts.get(0, 0)}")
    print(f"Sobreviveu (1):     {prediction_counts.get(1, 0)}")
    
    if 1 in prediction_rate:
        print(f"\nO modelo previu que {prediction_rate[1]:.2f}% dos passageiros sobreviveriam.")

    # Gerar o gráfico
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='Survived', data=submission_df, palette='viridis')
    ax.set_title('Distribuição das Previsões Finais do Modelo', fontsize=16)
    ax.set_xlabel('Previsão (0 = Não Sobreviveu, 1 = Sobreviveu)', fontsize=12)
    ax.set_ylabel('Número de Passageiros', fontsize=12)
    ax.set_xticklabels(['Não Sobreviveu', 'Sobreviveu'])

    # Adicionar anotações de contagem no gráfico
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plt.savefig('analise_dos_resultados_finais.png', bbox_inches='tight')
    plt.close()

    print("\nGráfico 'analise_dos_resultados_finais.png' gerado com sucesso na sua pasta!")

if __name__ == "__main__":
    analisar_resultados()