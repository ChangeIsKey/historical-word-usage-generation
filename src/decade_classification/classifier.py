from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import mean_squared_error
from transformers import DataCollatorWithPadding
import json 
from sklearn.model_selection import train_test_split
import torch
import os
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

LLMs_CACHE_DIR = os.environ["TMPDIR"]

base_model = "roberta-large"
output_dir = ""
path_to_dataset = ""
start_year = 1700
end_year = 2021
batch_size = 32
num_labels = len(range(start_year,end_year,10))
tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=LLMs_CACHE_DIR)
model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_labels, cache_dir=LLMs_CACHE_DIR)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


label2id = {y:j for j,y in enumerate(range(start_year,end_year,10))}

def load_dataset():
    texts = []
    labels = []
    with open(path_to_dataset) as f:
        for line in f:
            line = json.loads(line)
            text = line['text']
            start, end = line['start'], line['end']
            text = text[:start] + text[start:end] +  text[end:]
            label = line['year']
            texts.append(text)
            label = int(label/10)*10
            label = label2id[label]
            labels.append(label)
    return texts, labels


texts, labels = load_dataset()
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1)

with open('train.pkl','wb+') as f:
    pickle.dump((train_texts,train_labels),f)

with open('val.pkl','wb+') as f:
    pickle.dump((val_texts,val_labels),f)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)


class TimeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = TimeDataset(train_encodings, train_labels)
val_dataset = TimeDataset(val_encodings, val_labels)



args = TrainingArguments(
    learning_rate=1e-6,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    evaluation_strategy='steps',
    eval_steps=0.1,
    do_eval=True,
    weight_decay=0.01,
    output_dir=output_dir,
    save_strategy='steps',
    save_steps=0.1
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'f1': f1_score(labels, predictions, average='weighted')}

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
