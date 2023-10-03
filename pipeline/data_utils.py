import pandas as pd

def load_data(file_path):
    """
    Carrega os dados de um arquivo CSV para um DataFrame do Pandas.
    
    Args:
    - file_path (str): o caminho para o arquivo CSV.
    
    Returns:
    - DataFrame: os dados carregados.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Realiza pré-processamento nos dados carregados.
    
    Args:
    - df (DataFrame): o DataFrame carregado para ser pré-processado.
    
    Returns:
    - DataFrame: os dados pré-processados.
    """

    return df
