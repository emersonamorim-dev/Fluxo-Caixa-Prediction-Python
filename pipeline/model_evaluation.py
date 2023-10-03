import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f'Mean Absolute Error: {mae}') 
    return mae

