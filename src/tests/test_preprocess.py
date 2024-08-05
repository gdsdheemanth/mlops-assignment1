import pytest
import pandas as pd
from src.preprocess import Preprocessor


@pytest.fixture
def preprocessor():
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [5, 6, 7, 8, 9],
        'C': [9, 10, 11, 12, 13]
    }
    df = pd.DataFrame(data)
    return Preprocessor(dataframe=df)


def test_preprocess_data(preprocessor):
    preprocessor.preprocess_data()
    to_drop = preprocessor.get_to_drop()
    assert 'C' in to_drop


def test_split_data(preprocessor):
    preprocessor.df['target'] = [0, 1, 0, 1, 0]
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        target_column='target')
    assert len(X_train) == 3
    assert len(X_test) == 2
    assert len(y_train) == 3
    assert len(y_test) == 2
