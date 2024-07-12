import sys
sys.path.append("..")

from tox_bert.modeling_toxbert import BertToxicityRegressor
from transformers import AutoTokenizer, BertTokenizer, BartTokenizer, BartForConditionalGeneration, get_scheduler, BartConfig, Trainer, TrainingArguments, trainer_utils, DataCollatorWithPadding
import pandas as pd
import pickle, torch, os, math, copy, argparse
from datasets import DatasetDict, Dataset, concatenate_datasets
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.trainer_utils import set_seed
import numpy as np
from datasets import load_metric
# from detoxify import Detoxify
from tqdm import tqdm

"""
For baseline on SBIC dataset use this:
```python train.py --dataset_type sbic --threshold 0.5 --output_dir sbic_config1_model
--tox_model_dir ../tox_bert/bert-toxic-signals-probab/<tox-bert-checkpoint-folder>
--data_file ../../data/SBIC.v2.trn.csv```

For baseline on LatentHatred dataset use this:
```python train.py --dataset_type latent --threshold 0.5 --output_dir latent_config1_model
--tox_model_dir ../tox_bert/bert-toxic-signals-probab/<tox-bert-checkpoint-folder>
--data_file ../../data/latenthatred_posts_train.tsv```
"""

MODEL_NAME = "facebook/bart-base"
# MODEL_NAME = "facebook/bart-large"

id2tox = {
    1: "<TOXIC>",
    0: "<NOT_TOXIC>"
}

id2sevtox = {
    1: "<SEVERE_TOXIC>",
    0: "<NOT_SEVERE_TOXIC>"
}

id2obs = {
    1: "<OBSCENE>",
    0: "<NOT_OBSCENE>"
}

id2ide = {
    1: "<IDENTITY_ATTACK>",
    0: "<NOT_IDENTITY_ATTACK>"
}

id2ins = {
    1: "<INSULT>",
    0: "<NOT_INSULT>"
}

id2thr = {
    1: "<THREAT>",
    0: "<NOT_THREAT>"
}

# id2tox = {
#     1: "toxic",
#     0: "not toxic"
# }

# id2sevtox = {
#     1: "severely toxic",
#     0: "not severely toxic"
# }

# id2obs = {
#     1: "obscene",
#     0: "not obscene"
# }

# id2ide = {
#     1: "identity attack",
#     0: "no identity attack"
# }

# id2ins = {
#     1: "insulting",
#     0: "not insulting"
# }

# id2thr = {
#     1: "threatful",
#     0: "not threatful"
# }


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset_type', type=str, default='sbic', choices=['sbic', 'latent'], help='Pass in a dataset type.')
    parser.add_argument('--output_dir', type=str, default='model', help='Pass in a model output directory.')
    parser.add_argument('--tox_model_dir', type=str, help='Pass in the toxicity regressor model output directory.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Pass in a lambda/threshold value.')
    parser.add_argument('--toxic_bert', type=bool, default=False, help='Pass true if detoxify API is to be used')
    parser.add_argument('--ablate', type=str, default=None, help='Pass ablation type.')
   
    parser.add_argument('--seed', type=int, default=685, help='Pass in a seed value.')
    parser.add_argument('--batch_size', type=int, default=8, help='Pass in a batch size.')
    parser.add_argument('--num_epochs', type=float, default=3.0, help='Pass in the number of training epochs.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Pass in the learning rate for training.')
    parser.add_argument('--data_file', type=str, default='../data/SBIC.v2.trn.csv', help='Data File to load.')
    # parser.add_argument('--dev_file', type=str, help='Dev File to load in case data split isn\'t used.') # we do not use the dev file here, but ../data/SBIC.v2.dev.csv can be used

    return parser.parse_args()


def toxic_bert_inference(df):
    for batch in tqdm(range(math.ceil(len(df)/1000))):
        detoxify = Detoxify("unbiased", device="cuda")

        start = batch*1000
        end = min((batch+1)*1000, len(df))

        results = detoxify.predict(df['post'][start:end].values.tolist())

        toxic_preds = torch.tensor([[
            results['toxicity'][i],
            results['severe_toxicity'][i],
            results['obscene'][i],
            results['threat'][i],
            results['insult'][i],
            results['identity_attack'][i]
        ] for i in range(len(results['toxicity']))])

        if batch == 0:
            input_signals = toxic_preds
        else:
            input_signals = torch.cat((input_signals, toxic_preds), dim=0)
        
        input_signals = input_signals.to(device = "cpu")

        torch.cuda.empty_cache()

    return input_signals


def inference(ds, tox_tokenizer, tox_model_dir):
    tox_model = BertToxicityRegressor.from_pretrained(tox_model_dir, num_labels = 6, problem_type = "multi_label_classification")
    data_collator = DataCollatorWithPadding(tokenizer=tox_tokenizer)
    model.eval()
    trainer = Trainer(
        model=tox_model,
        eval_dataset=ds,
        tokenizer=tox_tokenizer,
        data_collator=data_collator
    )
    input_preds = trainer.predict(Dataset.from_dict({"input_ids": ds["input_ids"]})).predictions

    input_signals = torch.sigmoid(torch.from_numpy(input_preds))

    return input_signals

def get_toxicity_probabilities(df, tox_model_dir, toxic_bert = False, ablate = None):
    if ablate is not None:
        if ablate == "ones":
            input_signals = torch.from_numpy(np.ones((len(df), 6)))
        elif ablate == "zero":
            input_signals = torch.from_numpy(np.zeros((len(df), 6)))
        elif ablate == "rand":
            input_signals = torch.from_numpy(np.random.random_sample((len(df), 6)))
        else:
            raise ValueError('ablate can only take values ["ones", "zero", "rand"] or Nonetype.')
    else:
        print('getting toxicity probabilities')
        def tox_tokenize_data(
            dataset,
            tokenizer,
            padding=True,
            max_length=512,
        ):
            def process_labels(target_tokenized):
                target_tokenized['label_input_ids'] = target_tokenized['input_ids']
                del target_tokenized['input_ids']
                del target_tokenized['attention_mask']

            def tokenize(examples):
                pad_examples = "max_length" if padding else False

                seq2seq_tokenized = tokenizer(
                    examples['post'],
                    padding=pad_examples,
                    truncation=True,
                    max_length=max_length,
                    return_tensors = "pt"
                )

                with tokenizer.as_target_tokenizer():
                    target_tokenized = tokenizer(
                        examples['target'],
                        padding = pad_examples,
                        truncation = True,
                        max_length = max_length,
                        return_tensors = "pt"
                    )

                process_labels(target_tokenized)

                return {**seq2seq_tokenized, **target_tokenized}

            if 'HITId' in dataset:
                remove_cols = ['HITId', 'post', 'target']
            else:
                remove_cols = ['post', 'target']
            tokenized = dataset.map(
                tokenize, batched=True,
                batch_size=1000,
                num_proc=1,
                remove_columns=remove_cols
            )
            return tokenized

        train_tox_ds = Dataset.from_pandas(df)

        if toxic_bert:
            input_signals = toxic_bert_inference(df)
        else:
            tox_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            tokenized_tox_ds = tox_tokenize_data(train_tox_ds, tox_tokenizer)

            input_signals = inference(tokenized_tox_ds, tox_tokenizer, tox_model_dir)

    return input_signals.cpu()

def get_toxicity_labels(signals, sep_token, lambda_):
    labels = []
    for signal in signals:
        label = [
            id2tox[int(signal[0]>=lambda_)],
            id2sevtox[int(signal[1]>=lambda_)],
            id2obs[int(signal[2]>=lambda_)],
            id2ide[int(signal[3]>=lambda_)],
            id2ins[int(signal[4]>=lambda_)],
            id2thr[int(signal[5]>=lambda_)]
        ]
        labels.append(str(sep_token).join(label))

    return labels

def tokenize_data(
    dataset,
    tokenizer,
    dataset_type = "sbic",
    padding=True,
    max_length=1024,
):
    def process_labels(target_tokenized):
        target_tokenized['labels'] = torch.stack([
            torch.tensor([(l if l != tokenizer.pad_token_id else -100) for l in label]).to(device = "cuda")
            for label in target_tokenized['input_ids']
        ])
        del target_tokenized['input_ids']
        del target_tokenized['attention_mask']

    def tokenize(examples):
        pad_examples = "max_length" if padding else False
        
        seq2seq_tokenized = tokenizer(
            examples['post'],
            padding=pad_examples,
            truncation=True,
            max_length=max_length,
            return_tensors = "pt"
        )
        seq2seq_tokenized["input_ids"] = seq2seq_tokenized.pop("input_ids")
        seq2seq_tokenized["attention_mask"] = seq2seq_tokenized.pop("attention_mask")
        with tokenizer.as_target_tokenizer():
            target_tokenized = tokenizer(
                examples['target'],
                padding=pad_examples,
                truncation=True,
                max_length=max_length,
                return_tensors = "pt"
            )
        process_labels(target_tokenized)
        return {**seq2seq_tokenized, **target_tokenized}

    if dataset_type == "sbic":
        remove_cols = ['HITId', 'post', 'target']
    else:
        remove_cols = ['post', 'target']
    tokenized = dataset.map(
        tokenize, batched=True,
        num_proc=1,
        remove_columns=remove_cols
    )
    return tokenized, tokenizer

def train(
    model,
    datasets,
    model_name = "model",
    num_epochs=3.0,
    learning_rate=5e-6,
    batch_size=16,
    data_collator=None,
):

    training_args = TrainingArguments(
        output_dir = model_name,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit = 1,
        learning_rate = learning_rate,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        num_train_epochs = num_epochs,
        fp16 = True,
        report_to = "none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        data_collator=data_collator,
    )

    trainer.train()

def clean_post(df):
    df.post = df.post.str.replace(r'\bRT\b', ' ', regex=True)
    df.post = df.post.str.replace('(@[^\s]*\s|\s?@[^\s]*$)', ' ', regex=True)
    df.post = df.post.str.replace('https?://[^\s]*(\s|$)',' ',regex=True)
    df.post = df.post.str.strip()
    return df

def clean_target(df, target_col='targetStereotype', train=True, d_type = "sbic"):
    df[target_col] = df[target_col].replace(np.nan, '', regex=True)

    if not train:
        df[target_col] = df[target_col].str.lower()
        if target_col == 'targetStereotype':
            df = df.groupby(['HITId', 'post'], as_index=False).agg({target_col:set})
        if d_type == "sbic":
            df[target_col] = df[target_col].apply(lambda x: list(x))
        else:
            df[target_col] = df[target_col].apply(lambda x: [x])

    df.rename(columns={target_col: 'target'}, inplace=True)
    if target_col == 'targetStereotype':
        return df[['HITId','post','target']]
    return df[['post', 'target']]

if __name__ == "__main__":
    args = parse_args()
    set_seed(685)
    
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    model.cuda()
    print(model.device)

    if args.dataset_type == 'sbic':
        df = pd.read_csv(args.data_file, sep=",", engine='python')
        df = clean_post(df)
        df = clean_target(df, target_col = "targetStereotype", d_type=args.dataset_type)
        df_post = df[['HITId', 'post']]
        df_post = df_post.drop_duplicates().reset_index(drop=True)
        df = df_post.merge(df.drop(columns='post'), on='HITId', validate='one_to_many')
    else:
        df = pd.read_csv(args.data_file, sep="\\t", engine='python')[['post', 'implied_statement']]
        df = clean_post(df)
        df = clean_target(df, target_col = "implied_statement", d_type=args.dataset_type)

    input_signals = get_toxicity_probabilities(df, args.tox_model_dir, toxic_bert=args.toxic_bert, ablate=args.ablate)

    toxic_labels = get_toxicity_labels(input_signals, sep_token=tokenizer.sep_token, lambda_ = args.threshold)

    df['post'] = df['post'].apply(lambda x: x + tokenizer.sep_token)
    df['post'] = df['post'] + toxic_labels

    train_ds = Dataset.from_pandas(df)

    print("tokenizing")
    datasets = train_ds.train_test_split(test_size=0.2, shuffle=True)
    tokenized, tokenizer = tokenize_data(datasets, tokenizer, dataset_type=args.dataset_type, padding=True, max_length=1024)
    
    train(
        model,
        tokenized,
        model_name=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
    )