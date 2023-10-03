import pandas as pd
import chardet

def collect_from_csv(file_path="data/raw/raw_data.csv"):
    """
    Coleta dados de um arquivo CSV.
    """
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        
        data = pd.read_csv(file_path, encoding=result['encoding'])
        print(f"Dados coletados com sucesso de {file_path}")
        return data
    except Exception as e:
        print(f"Erro ao coletar dados do arquivo CSV: {e}")
        return None

def preprocess_data(data):
    """
    Realiza pré-processamento básico nos dados.
    """
    data.drop_duplicates(inplace=True)
    
    for column in data.columns:
        if data[column].dtype == 'O':
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            data[column].fillna(data[column].median(), inplace=True)
    
    threshold = 0.7 * len(data)
    data.dropna(axis=1, thresh=threshold, inplace=True)
    
    data = pd.get_dummies(data, drop_first=True)
    
    return data
