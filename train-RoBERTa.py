import pandas as pd
import datasets as ds
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def train():
    df = pd.read_csv('dataset/train.csv')
    df = df.fillna('')
    df = df.rename(columns={'label': 'labels'})
    df['text'] = df['title'] + '\n\n' + df['text']
    df = df.drop(columns=['id', 'title'])

    dataset = ds.Dataset.from_pandas(df)
    dataset.features['labels'] = ds.ClassLabel(num_classes=2, names=['unreliable', 'reliable'])
    # dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    train_data, test_data = dataset.train_test_split(test_size=0.25).values()

    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length=512)

    # define a function that will tokenize the model, and will return the relevant inputs for the model
    def tokenization(batched_text):
        return tokenizer(batched_text['text'], padding=True, truncation=True)

    train_data = train_data.map(tokenization, batched=True, batch_size=len(train_data))
    test_data = test_data.map(tokenization, batched=True, batch_size=len(test_data))

    # train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    # test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    train_data.set_format('torch')
    test_data.set_format('torch')

    # define the training arguments
    training_args = TrainingArguments(
        output_dir='trainer/results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        disable_tqdm=False,
        load_best_model_at_end=True,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=8,
        fp16=True,
        logging_dir='trainer/logs',
        dataloader_num_workers=4,
        run_name='roberta-classification'
    )

    # define accuracy metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # instantiate the trainer class and check for available devices
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=test_data
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer.train()
    trainer.evaluate()

    trainer.model.save_pretrained('saved-models/roBERTa-base/')


if __name__ == '__main__':
    train()

# {'eval_loss': 0.006559914909303188, 'eval_accuracy': 0.9990384615384615, 'eval_f1': 0.9990393852065321,
# 'eval_precision': 0.9996155324875048, 'eval_recall': 0.9984639016897081, 'eval_runtime': 106.7908,
# 'eval_samples_per_second': 48.693, 'epoch': 3.0}
# {'train_runtime': 1875.6339, 'train_samples_per_second': 0.389, 'epoch': 3.0}
