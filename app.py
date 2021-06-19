import json
import re

import pandas as pd
import numpy as np

from newsfetch.news import newspaper
from torch.nn.functional import softmax
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from flask import Flask, request, jsonify


app = Flask(__name__)
model = RobertaForSequenceClassification.from_pretrained('saved-models/roBERTa-base-2/')
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length=512)


def get_prediction(df_in):
    """
        Predicts whether or not each news article is reliable.

        :param df_in: pandas dataframe object containing the news articles
        :return: a list of 1s and 0s indicating if each input article is reliable
        """

    df_in = df_in.fillna('')

    inputs = tokenizer(
        df_in['text'].tolist(),
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    outputs = model(**inputs)
    predictions = softmax(outputs.logits, dim=-1).detach().numpy()
    class_predictions = np.argmax(predictions, axis=-1)
    return class_predictions


def is_valid(url):
    pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(pattern, url) is not None


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for predicting if news articles are reliable. Takes the json object from request as input and returns
    a json object containing the input url, prediction and special flags.

    :return: json object containing the input url, prediction and special flags. If article is not in English, flag
    would be 1, else if article is too short, flag would be 2.
    """

    if request.method == 'POST':
        data = request.get_json(force=True)
        urls = data['urls']

        flags = [0] * len(urls)
        texts = [None] * len(urls)
        for i in range(len(urls)):
            url = urls[i]

            if not is_valid(url):
                raise ValueError(f'Invalid URL: {url}')

            # Scrape news article
            news = newspaper(url)

            # Flag unsupported language
            if news.language != 'en':
                flags[i] = 1

            # Flag articles behind paywall
            elif len(news.article.split()) < 80:
                flags[i] = 2

            else:
                texts[i] = news.headline + '\n\n' + news.article
        df = pd.DataFrame({'text': texts})
        predictions = get_prediction(df)
        return jsonify({'url': urls, 'reliability': predictions.tolist(), 'flag': flags})


if __name__ == '__main__':
    app.run()
