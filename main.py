import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

from scripts.data_collection import collect_from_csv, preprocess_data
from scripts.train_model import train_model, load_model, save_model, predict
from scripts.data_preprocessing import DataPreprocessor

from scripts.data_cleaning import clean_data
from scripts.feature_engineering import create_features

               
data = collect_from_csv()

print("Iniciando o processo de coleta de dados...")
               
if data is not None:
        print("Iniciando o processo de pré-processamento de dados...")
        data = preprocess_data(data)
        print("Dados pré-processados com sucesso!")
        
         # Adicione os nomes das colunas categóricas aqui
        config = {
            'categorical_columns': [] 
        }
        
        preprocessor = DataPreprocessor(config)
        data = preprocessor.preprocess(data)       
                   
        print(data.head())  # Imprimir os primeiros registros dos dados pré-processados
else:
        print("Falha na coleta de dados.")
        
        
        # Dividindo os dados em treinamento e teste
        X = data.drop('target_column', axis=1) 
        y = data['target_column']  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Iniciando o treinamento do modelo...")
        
        # Treinando o modelo
        model = train_model(X_train, y_train)  
        save_model(model)  
        

       # Carregando o modelo
        loaded_model = load_model() 
        
        # Fazendo previsões
        predictions = predict(loaded_model, X_test) 
        

cleaned_data = clean_data('data/raw/raw_data.csv') if data is not None else None
    
if cleaned_data is not None:
        print("Dados limpos e prontos para serem processados!")
        
        # Visualização dos dados
        #plot_data(cleaned_data)

else:
  print("Falha na coleta ou limpeza dos dados.")
       
                      
# Carregar os dados reais do arquivo CSV
file_path = 'data/processed/data.csv'
df = pd.read_csv(file_path)

# Limpando os dados
cleaned_data_path = clean_data(file_path)

# Criando novas features
featured_data = create_features(cleaned_data)


# Verificar se a coluna de datas está presente, senão, usar o índice como representação do tempo
if 'Data' in df.columns:
    df['Mes'] = pd.DatetimeIndex(df['Data']).month
    df['Ano'] = pd.DatetimeIndex(df['Data']).year
else:
    df['Mes'] = df.index % 12 + 1
    df['Ano'] = df.index // 12 + 2020  # Ajuste o ano base conforme necessário

# Preparar os dados para treinamento
features = ['Entradas - Vendas (R$)', 'Entradas - Outros (R$)', 'Saídas - Salários (R$)',
            'Saídas - Despesas Operacionais (R$)', 'Saídas - Investimentos (R$)', 'Mes', 'Ano']


# Treina um modelo para cada recurso com Polynomial Features
future_inputs = pd.DataFrame()
for feature in features[:-2]:
    X = df[features[-2:]]  
    y = df[feature]

    model = make_pipeline(PolynomialFeatures(2), RandomForestRegressor(n_estimators=100, random_state=0))
    model.fit(X, y)

    # Faz as previsões para os próximos 6 meses
    future_months = pd.DataFrame({'Mes': [1, 2, 3, 4, 5, 6], 'Ano': [2024] * 6})
    future_predictions = model.predict(future_months)

    future_inputs[feature] = future_predictions

# Calcular 'Total Saídas (R$)'
future_inputs['Total Saídas (R$)'] = future_inputs[['Saídas - Salários (R$)',
                                                    'Saídas - Despesas Operacionais (R$)',
                                                    'Saídas - Investimentos (R$)']].sum(axis=1)

# Adiciona as colunas 'Mes' e 'Ano' ao future_inputs para corresponder às features de treinamento
future_inputs['Mes'] = [1, 2, 3, 4, 5, 6]
future_inputs['Ano'] = [2024, 2024, 2024, 2024, 2024, 2024]

# Agora, treina o modelo para 'Fluxo de Caixa Líquido (R$)'
X = df[features + ['Total Saídas (R$)']]
y = df['Fluxo de Caixa Líquido (R$)']

model = make_pipeline(PolynomialFeatures(2), RandomForestRegressor(n_estimators=100, random_state=0))
model.fit(X, y)

# Fazer previsões para os próximos 6 meses
cash_flow_predictions = model.predict(future_inputs[features + ['Total Saídas (R$)']])

# Visualizar as previsões
months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun']
plt.figure(figsize=(10, 6))
plt.plot(months, cash_flow_predictions, marker='o', linestyle='-', color='b', label='Fluxo de Caixa Previsto')

# Adiciona previsões de fluxo de caixa ao gráfico
for i, txt in enumerate(cash_flow_predictions):
    plt.annotate(f"R${txt:.2f}", (months[i], cash_flow_predictions[i]), textcoords="offset points", xytext=(0, 10),
                 ha='center')

plt.xlabel('Meses em 2024')
plt.ylabel('Fluxo de Caixa Previsto (R$)')
plt.title('Fluxo de Caixa Previsto para os próximos 6 meses')
plt.grid(True)
plt.legend()
plt.show()

# Imprimi as previsões no console também, se necessário
for month, pred in zip(months, cash_flow_predictions):
    print(f"Fluxo de Caixa Previsto para 2024/{month}: R${pred:.2f}")