import pandas as pd
import numpy as np


class Preprocessor:
    def __init__(self, filepath=None, dataframe=None, delimiter=';'):
        self.filepath = filepath
        self.delimiter = delimiter
        self.df = dataframe
        self.to_drop = None

    def load_data(self):
        self.df = pd.read_csv(self.filepath, delimiter=self.delimiter)

    def preprocess_data(self):
        # Find high correlation features
        corr_matrix = self.df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1)
                                  .astype(np.bool_))
        self.to_drop = [column for column in upper.columns if any(upper[column] 
                                                                  > 0.8)]

        # Drop high correlation features
        self.df = self.df.drop(columns=self.to_drop)

    def split_data(self, target_column, test_size=0.3, random_state=42):
        X = self.df.drop([target_column], axis=1)
        y = self.df[target_column]
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size,
                                random_state=random_state, stratify=y)

    def get_to_drop(self):
        return self.to_drop
       
    def get_dataFrame(self):
        return self.df
