from sklearn.linear_model import LinearRegression
import joblib

def train_model(X_train, y_train):   
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path='trained_model.pkl'):
    try:
        joblib.dump(model, model_path)
        print(f"Modelo salvo com sucesso em {model_path}")
    except Exception as e:
        print(f"Erro ao salvar o modelo: {e}")

def load_model(model_path='trained_model.pkl'):
    try:
        model = joblib.load(model_path)
        print(f"Modelo carregado com sucesso de {model_path}")
        return model
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None

def predict(model, data):
    try:
        predictions = model.predict(data)
        return predictions
    except Exception as e:
        print(f"Erro ao fazer previsões: {e}")
        return None
