import pandas as pd
import numpy as np
from io import StringIO
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


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        df = pd.json_normalize(data)
        predictions = get_prediction(df)
        return jsonify({'reliability': predictions})


if __name__ == '__main__':
    app.run()
