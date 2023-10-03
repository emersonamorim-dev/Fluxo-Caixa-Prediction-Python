import pandas as pd
from joblib import load

# Carregar modelo treinado
model = load('models/model.joblib')

# Carregar dados de teste ou novos dados
data = pd.read_csv('data/processed/test_data.csv')

# Fazer previsões
predictions = model.predict(data)

print("Previsões:", predictions)
