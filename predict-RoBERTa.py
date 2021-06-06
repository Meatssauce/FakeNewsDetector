import pandas as pd
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from torch.nn.functional import softmax
import numpy as np


def predict(df_in=None, should_save_csv=False):
    if df_in is not None:
        df_in = pd.read_csv('dataset/test.csv')
    df_in = df_in.fillna('')

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

# upon trainer.predict(dataset)
# UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 due to no predicted samples.
# Use `zero_division` parameter to control this behavior.
#   _warn_prf(average, modifier, msg_start, len(result))
