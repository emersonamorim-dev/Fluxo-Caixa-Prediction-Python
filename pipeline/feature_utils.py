import pandas as pd

def add_features(df, col1, col2):
    if col1 not in df.columns or col2 not in df.columns:
        print(f"As colunas '{col1}' e '{col2}' não existem no DataFrame.")
        return df

    df['nova_feature'] = df[col1] - df[col2]
    return df

