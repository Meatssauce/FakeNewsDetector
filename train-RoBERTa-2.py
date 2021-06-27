import pandas as pd
import datasets as ds
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def train():
    df_true = pd.read_csv('dataset2/True.csv')
    df_fake = pd.read_csv('dataset2/Fake.csv')

    df_true['labels'] = 1
    df_fake['labels'] = 0

    df = df_true.append(df_fake)
    df['text'] = df['title'] + '\n\n' + df['text']
    df = df.drop(columns=['title', 'subject', 'date'])

    dataset = ds.Dataset.from_pandas(df)
    dataset.features['labels'] = ds.ClassLabel(num_classes=2, names=['unreliable', 'reliable'])
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
        num_train_epochs=1,
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

    trainer.model.save_pretrained('saved-models/roBERTa-base-2/')


if __name__ == '__main__':
    train()

# {'eval_loss': 9.170903649646789e-05, 'eval_accuracy': 1.0, 'eval_f1': 1.0, 'eval_precision': 1.0,
# 'eval_recall': 1.0, 'eval_runtime': 234.5631, 'eval_samples_per_second': 47.855, 'epoch': 1.0}


