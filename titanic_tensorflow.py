import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

print(f"O script está rodando a partir de: {os.getcwd()}")

# ==============================================================================
# 0. DEFININDO SEED PARA REPRODUTIBILIDADE
# ==============================================================================
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
# ----------------------------------------------------


# ==============================================================================
# 1. ANÁLISE EXPLORATÓRIA DE DADOS (STORYTELLING)
# ==============================================================================
def run_exploratory_data_analysis():
    """
    Função que gera e salva os gráficos e tabelas para o storytelling.
    """
    print("--- INICIANDO ANÁLISE EXPLORATÓRIA (STORYTELLING) ---")

    # Configurações visuais
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

    try:
        df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("Erro: O arquivo 'train.csv' não foi encontrado. Certifique-se de que ele está no mesmo diretório do script.")
        return

    # Criar uma cópia do DataFrame para as plots para não alterar o original
    df_plot = df.copy()
    # Traduzir a coluna 'Sex' para os gráficos, eliminando a necessidade de set_xticklabels
    df_plot['Gênero'] = df_plot['Sex'].map({'female': 'Mulher', 'male': 'Homem'})
    # Criar uma coluna 'Sobreviveu_Label' para os gráficos
    df_plot['Sobreviveu_Label'] = df_plot['Survived'].map({0: 'Não Sobreviveu', 1: 'Sobreviveu'})


    # 1. Análise Geral de Sobrevivência
    print("\n[Storytelling 1/6] Analisando a taxa geral de sobrevivência...")
    plt.figure()
    # CORRIGIDO: Adicionado hue e legend=False
    ax = sns.countplot(data=df_plot, x='Sobreviveu_Label', hue='Sobreviveu_Label', palette='viridis', legend=False)
    ax.set_title('Distribuição Geral de Sobrevivência', fontsize=16)
    ax.set_xlabel('Situação') # O Label já tem a informação (0) e (1)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.savefig('1_geral_sobrevivencia.png', bbox_inches='tight')
    plt.close()

    # 2. Análise por Gênero
    print("[Storytelling 2/6] Analisando a sobrevivência por gênero...")
    plt.figure()
    # CORRIGIDO: Usando a coluna 'Gênero' traduzida e adicionado hue
    ax = sns.barplot(x='Gênero', y='Survived', data=df_plot, hue='Gênero', palette='plasma', estimator=lambda y: sum(y)*100.0/len(y), legend=False)
    ax.set_title('Taxa de Sobrevivência por Gênero', fontsize=16)
    ax.set_ylabel('Taxa de Sobrevivência (%)')
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.savefig('2_sobrevivencia_por_genero.png', bbox_inches='tight')
    plt.close()

    # 3. Análise por Classe Social
    print("[Storytelling 3/6] Analisando a sobrevivência por classe social...")
    plt.figure()
    # CORRIGIDO: Adicionado hue e legend=False
    ax = sns.barplot(x='Pclass', y='Survived', data=df_plot, hue='Pclass', palette='magma', estimator=lambda y: sum(y)*100.0/len(y), legend=False)
    ax.set_title('Taxa de Sobrevivência por Classe Social', fontsize=16)
    ax.set_xlabel('Classe do Passageiro')
    ax.set_ylabel('Taxa de Sobrevivência (%)')
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.savefig('3_sobrevivencia_por_classe.png', bbox_inches='tight')
    plt.close()

    # 4. Análise por Idade
    print("[Storytelling 4/6] Analisando a distribuição de idade...")
    plt.figure()
    # CORRIGIDO: Usando a coluna 'Sobreviveu_Label' para a legenda
    sns.histplot(data=df_plot, x='Age', hue='Sobreviveu_Label', kde=True, multiple='stack', palette='viridis')
    plt.title('Distribuição de Idade por Sobrevivência', fontsize=16)
    plt.legend(title='Situação', labels=['Sobreviveu', 'Não Sobreviveu'])
    plt.savefig('4_sobrevivencia_por_idade.png', bbox_inches='tight')
    plt.close()

    # 5. Análise Combinada: Classe e Gênero
    print("[Storytelling 5/6] Analisando a combinação de classe e gênero...")
    # CORRIGIDO: Usando a coluna 'Sobreviveu_Label' para os títulos das colunas
    g = sns.catplot(x="Pclass", hue="Gênero", col="Sobreviveu_Label", data=df_plot, kind="count", height=6, aspect=.7, palette='plasma',
                    col_order=['Não Sobreviveu', 'Sobreviveu']) # Garante a ordem
    g.fig.suptitle('Contagem de Sobrevivência por Classe e Gênero', y=1.03)
    g.set_axis_labels("Classe Social", "Número de Passageiros")
    g.set_titles("{col_name}") # Usa o valor da coluna diretamente como título
    plt.savefig('5_classe_genero_sobrevivencia.png', bbox_inches='tight')
    plt.close()

    # 6. Matriz de Correlação
    print("[Storytelling 6/6] Gerando matriz de correlação...")
    df_corr = df.copy()
    df_corr['Sex'] = df_corr['Sex'].map({'female': 0, 'male': 1})
    df_corr['Embarked'] = df_corr['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df_corr = df_corr[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].dropna()
    correlation_matrix = df_corr.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlação das Features', fontsize=16)
    plt.savefig('6_matriz_correlacao.png', bbox_inches='tight')
    plt.close()

    print("--- ANÁLISE EXPLORATÓRIA CONCLUÍDA ---")
    print("Gráficos salvos como arquivos .png no diretório.\n")

# ==============================================================================
# 2. FUNÇÕES DE VISUALIZAÇÃO DO MODELO
# ==============================================================================
def plot_training_history(history):
    """
    Plota os gráficos de acurácia e perda durante o treinamento.
    """
    plt.figure(figsize=(12, 5))

    # Gráfico de Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Acurácia de Treino')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title('Histórico de Acurácia', fontsize=16)
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    
    # Gráfico de Perda (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Perda de Treino')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.title('Histórico de Perda', fontsize=16)
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()
    plt.tight_layout()
    plt.savefig('8_grafico_perda_e_acuracia.png', bbox_inches='tight')
    plt.close()
    print("[Visualização] Gráficos de acurácia e perda salvos.")

def plot_confusion_matrix(y_true, y_pred):
    """
    Plota a matriz de confusão.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Não Sobreviveu', 'Sobreviveu'],
                yticklabels=['Não Sobreviveu', 'Sobreviveu'])
    plt.title('Matriz de Confusão', fontsize=16)
    plt.ylabel('Verdadeiro')
    plt.xlabel('Previsto')
    plt.savefig('9_matriz_confusao.png', bbox_inches='tight')
    plt.close()
    print("[Visualização] Matriz de confusão salva.")


# ==============================================================================
# 3. TREINAMENTO DO MODELO E PREVISÃO
# ==============================================================================
def train_and_predict_model():
    """
    Função que pré-processa os dados, treina a RNA e gera o relatório.
    """
    print("--- INICIANDO TREINAMENTO DA REDE NEURAL ---")
    
    # Carregar Dados
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    test_passenger_ids = test_df['PassengerId']

    # Pré-processamento
    print("[RNA 1/7] Pré-processando os dados...")
    def preprocess(df):
        df['Age'] = df['Age'].fillna(df['Age'].mean())
        df['Embarked'] = df['Embarked'].fillna('S')
        df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
        df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
        df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
        df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])
        return df

    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    # Preparar Dados para o Modelo
    X = train_df.drop('Survived', axis=1)
    y = train_df['Survived']
    
    # DIVISÃO EM TREINO E VALIDAÇÃO
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(test_df)

    # Construir e Compilar a RNA
    print("[RNA 2/7] Construindo o modelo...")
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(units=16, activation='relu', name='Camada_Oculta_1'),
        tf.keras.layers.Dense(units=8, activation='relu', name='Camada_Oculta_2'),
        tf.keras.layers.Dense(units=1, activation='sigmoid', name='Camada_de_Saida')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Gerar resumo e fluxograma da arquitetura
    print("[RNA 3/7] Gerando resumo e fluxograma da arquitetura...")
    print("Resumo da Arquitetura do Modelo:")
    model.summary()
    try:
        plot_model(model, to_file='arquitetura_modelo.png', show_shapes=True, show_layer_names=True)
        print("Fluxograma 'arquitetura_modelo.png' gerado com sucesso.")
    except ImportError:
        print("\nAVISO: Não foi possível gerar o fluxograma do modelo.")
        print("Para isso, instale 'pydot' e 'graphviz'.")
        print("Comandos: pip install pydot graphviz")
        print("E instale o programa Graphviz no seu sistema: https://graphviz.org/download/\n")

    # Treinar a RNA
    print("[RNA 4/7] Treinando o modelo...")
    history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, 
                        validation_data=(X_val_scaled, y_val),
                        verbose=0)
    print("[RNA 5/7] Treinamento concluído.")

    # Visualizar Outputs do Treinamento
    plot_training_history(history)
    
    # Avaliar o modelo
    val_loss, val_accuracy = model.evaluate(X_val_scaled, y_val, verbose=0)
    print(f"\nAcurácia no conjunto de validação: {val_accuracy*100:.2f}%")
    
    # Visualizar Matriz de Confusão
    y_pred_val = (model.predict(X_val_scaled) > 0.5).astype(int)
    plot_confusion_matrix(y_val, y_pred_val)
    
    print("\nRelatório de Classificação no Conjunto de Validação:")
    print(classification_report(y_val, y_pred_val, target_names=['Não Sobreviveu', 'Sobreviveu']))

    # Gerar Arquivo Final
    print("[RNA 6/7] Gerando arquivo final...")
    predictions_prob = model.predict(X_test_scaled)
    predictions = (predictions_prob > 0.5).astype(int).flatten()

    submission_df = pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': predictions})
    submission_df.to_csv('titanic_report_rna.csv', index=False)
    
    # Inspecionar o Arquivo de Submissão
    print("[RNA 7/7] Visualizando as primeiras linhas do arquivo final:")
    print(submission_df.head(10))
    
    print("\n--- PROCESSO FINALIZADO ---")
    print("Arquivo 'titanic_report_rna.csv' gerado com sucesso!")


if __name__ == "__main__":
    run_exploratory_data_analysis()
    train_and_predict_model()