import pandas as pd
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from torch.nn.functional import softmax
import numpy as np
import argparse

"""
Either import this file and use the predict_reliability() function directly in another python script e.g.
from predict import predict_reliability
df_in = ... # a pandas dataframe object
predictions = predict_reliability(df_in)
    Note: you can also pass no input to the function in which case it will read from the default input csv file path
or run in cmd via
predict --input_dir [input path] --output_dir [output put]

You would need to have installed everything in the requirements.txt in both cases via command
pip install -r requirements.txt
"""


parser = argparse.ArgumentParser(description='FakeNews-roBERTa')
parser.add_argument('--input_path', type=str, default='dataset/test-simple.csv')
parser.add_argument('--output_path', type=str, default='dataset/predictions.csv')
parser.add_argument('--model_dir', type=str, default='saved-models/roBERTa-base-2/')
args = parser.parse_args()


def predict_reliability(model=None, tokenizer=None, df_in=None, should_save_csv=False):
    """
    Predicts whether or not a news article is reliable.

    :param df_in: pandas dataframe object containing the news articles. If None, will read .csv file from disk
    :param should_save_csv: if True, will also save the output as a .csv file
    :return: a list of 1s and 0s indicating if each input article is reliable
    """

    if df_in is None:
        df_in = pd.read_csv(args.input_path)
    df_in = df_in.fillna('')

    if model is None:
        model = RobertaForSequenceClassification.from_pretrained(args.model_dir)
    if tokenizer is None:
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length=512)

    inputs = tokenizer(
        df_in['text'].tolist(),
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    outputs = model(**inputs)
    predictions = softmax(outputs.logits, dim=-1).detach().numpy()
    class_predictions = np.argmax(predictions, axis=-1)

    if should_save_csv:
        df_out = df_in[['id']]
        df_out['labels'] = class_predictions
        df_out.to_csv(args.output_path, index=False)

    return class_predictions


if __name__ == '__main__':
    predict_reliability(should_save_csv=True)
