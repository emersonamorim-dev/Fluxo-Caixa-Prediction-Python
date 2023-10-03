# scripts/data_cleaning.py

import pandas as pd

def clean_data(file_path):
    """
    Limpa os dados carregados a partir de um arquivo CSV.

    Args:
    - file_path (str): O caminho para o arquivo de dados brutos.

    Returns:
    - DataFrame: Retorna os dados limpos como um DataFrame.
    """
    print(f"Limpeza dos dados de {file_path} em andamento...")
    
    try:
        df = pd.read_csv(file_path)
        df = df.dropna()
        df['Total Entradas (R$)'] = df['Total Entradas (R$)'].astype(float)

        print("Dados limpos com sucesso!")
        return df
    
    except Exception as e:
        print(f"Erro durante a limpeza dos dados: {e}")
        return None
