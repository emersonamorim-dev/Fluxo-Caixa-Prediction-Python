import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

# Carregar dados
df = pd.read_csv('data/processed/data.csv')

# Separar features e target
X = df.drop(columns='target')
y = df['target']

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir modelo
model = RandomForestRegressor(random_state=42)

# Treinar modelo
model.fit(X_train, y_train)

# Salvar modelo treinado
dump(model, 'models/model.joblib')

print("Modelo treinado e salvo com sucesso.")
