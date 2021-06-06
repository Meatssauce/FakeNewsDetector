import pandas as pd
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from torch.nn.functional import softmax
import numpy as np


def predict(df_in=None, should_save_csv=False):
    """
    Predicts whether or not a news article is reliable.

    :param df_in: pandas dataframe object containing the news articles. If None, will read .csv file from disk
    :param should_save_csv: if True, will also save the output as a .csv file
    :return: a list of 1s and 0s indicating if each input article is reliable
    """

    if df_in is not None:
        df_in = pd.read_csv('dataset/test.csv')
    df_in = df_in.fillna('')

    # might move this outside the function to improve performance when repeatedly calling predict()
    model = RobertaForSequenceClassification.from_pretrained('saved-models/roBERTa-base/')
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
        df_out.to_csv('dataset/predictions.csv')

    return class_predictions


if __name__ == '__main__':
    predict()
