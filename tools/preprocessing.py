import pandas as pd
import re
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


def load_valid_data(file_path, target):
    df = pd.read_csv(file_path)
    df = df.dropna(axis=0, how='any', subset=[target])
    df = df.drop_duplicates()

    return df.drop(columns=[target]), df[target]


def feature_engineering_method(X):
    X['exclamation_and_question_mark_frequency'] = X['text'].str.count(r'!|\?')

    X['title_word_count'] = X['title'].str.split().str.len()

    title_baseline_word_count = 8  # same as that used by the paper
    X['long_title'] = X['title_word_count'] > title_baseline_word_count

    # lemmatise text
    w_tokenizer = WhitespaceTokenizer()
    lemmatizer = WordNetLemmatizer()

    def lemmatize_text(text):
        for token, tag in pos_tag(w_tokenizer.tokenize((str(text)))):
            wntag = tag[0].lower()
            wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
            if not wntag:
                return token
            else:
                return lemmatizer.lemmatize(token, wntag)
        # return ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(str(text))])

    # X['text_lemmatized'] = X['text'].apply(lemmatize_text)

    # remove stop words
    # do something...

    # remove punctuations

    X = X.drop(columns=['id', 'author', 'title', 'text'])

    return X


# Testing purposes only
X, y = load_valid_data('dataset/train.csv', target='label')
X = feature_engineering_method(X)
print()
