import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

from modeling_toxbert import BertToxicityRegressor

import torch
import torch.nn as nn
from datasets import Dataset
import numpy as np
import evaluate

MODEL_OUTPUT_DIR = "bert-toxic-signals-probab"
DATA_DIR = "jigsaw-data/train.csv"

def preprocess_function(examples):
    return tokenizer(examples["comment_text"], padding = True, truncation = True, return_tensors = "pt")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.sigmoid(torch.from_numpy(predictions))
    rmse_error = torch.sqrt(torch.mean((predictions - labels)**2)).item()
    return {"rmse_error": rmse_error}

df_train = pd.read_csv(DATA_DIR)
df_train = df_train[["comment_text", "target", "severe_toxicity", "obscene", "identity_attack", "insult", "threat"]]
labels = df_train[["target", "severe_toxicity", "obscene", "identity_attack", "insult", "threat"]].values.tolist()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertToxicityRegressor.from_pretrained("bert-base-uncased", num_labels = 6, problem_type = "multi_label_classification")

train_ds = Dataset.from_pandas(df_train)
tokenized_train_ds = train_ds.map(preprocess_function, batched = True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_ds = tokenized_train_ds.remove_columns(["target", "severe_toxicity", "obscene", "identity_attack", "insult", "threat"])
tokenized_ds = tokenized_ds.add_column("labels", labels)
tokenized_ds = tokenized_ds.train_test_split(test_size = 0.2)

accuracy = evaluate.load("accuracy")

model.cuda()

training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to = "wandb"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
