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
import random
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import DataLoader, IterableDataset
from sklearn.metrics import root_mean_squared_error


LLMs_CACHE_DIR = os.environ["TMPDIR"]
base_model = "roberta-large"
path_to_dataset = ""
output_dir = ""
batch_size = 32

tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=LLMs_CACHE_DIR)
model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1, cache_dir=LLMs_CACHE_DIR)
num_added_toks = tokenizer.add_tokens(['<t>','</t>'], special_tokens=True) 
model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def load_dataset():
    texts = []
    labels = []
    with open(path_to_dataset) as f:
        for line in f:
            line = json.loads(line)
            text = line['usage']
            start, end = line['offsets']
            lemma = line['lemma']
            label = float(line['label'])
            text = text[:start] + '<t>' + text[start:end]  + '</t>' +  text[end:]
            definition = line['definition']
            if not definition == None:
                text = text + '</s></s>' + line['definition']
                texts.append(text)
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


class WSDDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = WSDDataset(train_encodings, train_labels)
val_dataset = WSDDataset(val_encodings, val_labels)

args = TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    evaluation_strategy='steps',
    eval_steps=0.1,
    logging_dir='logs',
    do_eval=True,
    weight_decay=0.01,
    output_dir=output_dir,
    save_strategy='steps',
    save_steps=0.1
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = root_mean_squared_error(labels, predictions)
    return {"rmse": rmse}

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
