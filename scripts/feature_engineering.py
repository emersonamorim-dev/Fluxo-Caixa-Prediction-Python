import pandas as pd

def create_features(df):
    """
    Adiciona novos recursos ao DataFrame.

    Args:
    df (DataFrame): DataFrame contendo os dados.

    Returns:
    DataFrame: DataFrame com novos recursos adicionados.
    """
    # Adiciona novos recursos, como média móvel de entradas/saídas, etc.
    df['Media Movel Entradas'] = df['Total Entradas (R$)'].rolling(window=3).mean()
    
    return df
