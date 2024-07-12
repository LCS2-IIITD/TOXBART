import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    Trainer,
    DataCollatorWithPadding
)

from modeling_toxbert import BertToxicityRegressor

import torch, os
import torch.nn as nn
from datasets import Dataset
import numpy as np
import evaluate
from sklearn.metrics import classification_report

df_test_1 = pd.read_csv("jigsaw-data/test_private_expanded.csv")
df_test_2 = pd.read_csv("jigsaw-data/test_public_expanded.csv")

df_test = pd.concat([df_test_1, df_test_2], axis = 0)
df_test = df_test[["comment_text", "toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat"]]

labels = df_test[["toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat"]].values.tolist()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BertToxicityRegressor.from_pretrained("bert-toxic-signals-probab/checkpoint-90244")

test_ds = Dataset.from_pandas(df_test)

def preprocess_function(examples):
    return tokenizer(examples["comment_text"], padding = True, truncation = True, return_tensors = "pt")

tokenized_test_ds = test_ds.map(preprocess_function, batched = True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_ds = tokenized_test_ds.remove_columns(["toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat"])
tokenized_ds = tokenized_ds.add_column("labels", labels)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.sigmoid(torch.from_numpy(predictions))
    rmse_error = torch.sqrt(torch.mean((predictions - labels)**2)).item()
    return {"rmse_error": rmse_error}

model.cuda()

trainer = Trainer(
    model=model,
    eval_dataset=tokenized_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

preds = trainer.predict(tokenized_ds)
print(preds)
