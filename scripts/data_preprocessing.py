import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.encoders = {}
        self.train_columns = None 

    def preprocess(self, data, is_training=True):
        processed_data = data.copy()

        for col in self.config['categorical_columns']:
            if col in processed_data.columns:
                if col not in self.encoders:
                    self.encoders[col] = OneHotEncoder(sparse_output=False, drop='first')
                    encoded_data = self.encoders[col].fit_transform(processed_data[[col]])
                else:
                    encoded_data = self.encoders[col].transform(processed_data[[col]])

                encoded_cols = [f"{col}_{category}" for category in self.encoders[col].categories_[0][1:]]
                encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols)

                processed_data = pd.concat([processed_data, encoded_df], axis=1)
                processed_data.drop(col, axis=1, inplace=True)

        if is_training:
            self.train_columns = processed_data.columns
        else:
            if self.train_columns is None:
                raise ValueError("O preprocessador deve ser usado no modo de treinamento antes do modo de previsão.")
            
            missing_cols = set(self.train_columns) - set(processed_data.columns)
            missing_data = {c: [0] * len(processed_data) for c in missing_cols}
            missing_df = pd.DataFrame(missing_data)
            processed_data = pd.concat([processed_data, missing_df], axis=1)
            processed_data = processed_data[self.train_columns]

        return processed_data
