import pandas as pd
import re

def load_valid_data(file_path, target):
    df = pd.read_csv(file_path)
    df = df.dropna(axis=0, how='any', subset=[target])
    df = df.drop_duplicates()

    return df


def feature_engineering_method(self, X):
    X['exclamation_and_question_mark_frequency'] = X['text'].str.count(r'!|?')

    return X


# Testing purposes only
df = load_valid_data('dataset/train.csv', target='label')
X, y = df.drop(columns='label'), df['label']
X['exclamation_and_question_mark_frequency'] = X['text'].str.count(r'!|\?')
print()
