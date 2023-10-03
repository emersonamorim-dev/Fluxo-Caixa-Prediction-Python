import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns 

from model_evaluation import evaluate_model
from data_visualization import plot_data

from data_utils import load_data, preprocess_data
from feature_utils import add_features
from visualize import plot_predictions


def main():
    # Carregar dados
    df = load_data('../data/processed/data.csv')  


    # Esta linha para cima para garantir que a coluna 'Ano/Mês' ainda exista no DataFrame
    plot_data(df)

    # Pré-processamento de dados
    # Continuamos removendo a coluna 'Ano/Mês' aqui
    df = preprocess_data(df).drop(columns=['Ano/Mês'], errors='ignore')


    # Usei colunas válidas no DataFrame.
    df = add_features(df, 'Total Entradas (R$)', 'Total Saídas (R$)')

    # Todos os devem dados são numéricos antes do treinamento do modelo
    df = df.select_dtypes(include=['number'])

    # Dividi os dados em conjunto de treino e teste
    X = df.drop(columns=['Fluxo de Caixa Líquido (R$)'], errors='ignore')
    y = df.get('Fluxo de Caixa Líquido (R$)', pd.Series())  
    if y.empty:
        print("Erro: Não há coluna 'Fluxo de Caixa Líquido (R$)' no DataFrame.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar modelo
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Avaliar modelo
    mae = evaluate_model(model, X_test, y_test)
    print(f"Mean Absolute Error: {mae}")

    # Visualizar previsões vs reais
    plot_predictions(y_test, model.predict(X_test))
    

# Certifique-se de carregar o seu DataFrame real
df = pd.DataFrame({
    'Ano/Mês': [f"2023-{i:02d}" for i in range(1, 13)],
    'Receitas': [i * 1000 for i in range(1, 13)],
    'Custos': [i * 500 for i in range(1, 13)],
    'Despesas Operacionais': [i * 300 for i in range(1, 13)]
})

# Filtra o DataFrame para mostrar dados a cada 3 meses
df = df.iloc[::3]

# Definindo um estilo usando seaborn
sns.set_theme(style="whitegrid")

# Mostra uma DRE em gráfico a cada 3 meses
def plot_dre_data(df):
    required_columns = ['Ano/Mês', 'Receitas', 'Custos', 'Despesas Operacionais']
    
    if not all(column in df.columns for column in required_columns):
        print("Algumas colunas necessárias não existem no DataFrame.")
        return

    plt.figure(figsize=(12, 8))
    
    # Plotando os dados com estilos diferentes para cada linha
    sns.lineplot(x='Ano/Mês', y='Receitas', data=df, marker="o", label='Receitas', dashes=False)
    sns.lineplot(x='Ano/Mês', y='Custos', data=df, marker="x", label='Custos', dashes=False)
    sns.lineplot(x='Ano/Mês', y='Despesas Operacionais', data=df, marker="s", label='Despesas Operacionais', dashes=False)

    plt.title('Demonstrativo de Resultados do Exercício (2023)')  
    plt.xlabel('Mês')  
    plt.ylabel('Valores em R$')  

    # Exibindo o gráfico com um estilo melhorado
    sns.despine(left=True)

    # Anotando valores numéricos nos pontos
    for i, (mes, receitas, custos, despesas) in enumerate(
        zip(df['Ano/Mês'], df['Receitas'], df['Custos'], df['Despesas Operacionais'])):

        plt.annotate(f'{receitas:,.0f}', 
                     (i, receitas), 
                     textcoords="offset points", 
                     xytext=(0, 10), 
                     ha='center', 
                     fontsize=9)

        plt.annotate(f'{custos:,.0f}', 
                     (i, custos), 
                     textcoords="offset points", 
                     xytext=(0, -20), 
                     ha='center', 
                     fontsize=9)

        plt.annotate(f'{despesas:,.0f}', 
                     (i, despesas), 
                     textcoords="offset points", 
                     xytext=(0, -35), 
                     ha='center', 
                     fontsize=9)

    plt.xticks(rotation=45)
    plt.legend(loc='upper left')  
    plt.tight_layout()
    plt.show()

# Chamada da função
plot_dre_data(df)


if __name__ == "__main__":
    main()

