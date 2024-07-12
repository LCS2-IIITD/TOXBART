from transformers import BartTokenizer, Trainer, TrainingArguments, BartForConditionalGeneration
import pandas as pd
import torch, argparse
from datasets import Dataset
from transformers.trainer_utils import set_seed
import numpy as np

"""
For meta-data experiment on SBIC dataset use this:
```python train.py --dataset_type sbic --output_dir sbic_config2_model --data_file ../../data/SBIC.v2.trn.csv```

For meta-data experiment on LatentHatred dataset use this:
```python train.py --dataset_type latent --output_dir latent_config2_model --data_file ../../data/latenthatred_raw_train.tsv```
"""

LEWDY = '[lewdY]'
LEWDN = '[lewdN]'
OFFY = '[offY]'
OFFN = '[offN]'
INTY = '[intY]'
INTN = '[intN]'
GRPY = '[grpY]'
GRPN = '[grpN]'
INGY = '[ingY]'
INGN = '[ingN]'

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset_type', type=str, default='sbic', choices=['sbic', 'latent'], help='Pass in a dataset type.')
    parser.add_argument('--output_dir', type=str, default='model', help='Pass in a model output directory.')
   
    parser.add_argument('--seed', type=int, default=685, help='Pass in a seed value.')
    parser.add_argument('--batch_size', type=int, default=16, help='Pass in a batch size.')
    parser.add_argument('--num_epochs', type=float, default=3.0, help='Pass in the number of training epochs.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Pass in the learning rate for training.')
    parser.add_argument('--data_file', type=str, default='../../data/SBIC.v2.trn.csv', help='Data File to load.')
    # parser.add_argument('--dev_file', type=str, help='Dev File to load in case data split isn\'t used.') # we do not use the dev file here, but ../data/SBIC.v2.dev.csv can be used

    return parser.parse_args()

def categorize_var(df):
    df.sexYN = np.where(df.sexYN >= 0.5, LEWDY, LEWDN)
    df.offensiveYN = np.where(df.offensiveYN >= 0.5, OFFY, OFFN)
    df.intentYN = np.where(df.intentYN >= 0.5, INTY, INTN)
    df.whoTarget = np.where(df.whoTarget >= 0.5, GRPY, GRPN)
    df.speakerMinorityYN = np.where(df.speakerMinorityYN >= 0.5, INGY, INGN)
    
    return df

def create_sbic_text_column(df, tokenizer):
    df.targetMinority = df.targetMinority.replace(np.nan, '', regex=True)
    df['text'] = tokenizer.bos_token + df.post + tokenizer.sep_token + df.sexYN + ' ' + df.offensiveYN + ' ' + \
                  df.intentYN + ' ' + df.whoTarget + tokenizer.sep_token + df.targetMinority + \
                  tokenizer.sep_token + df.speakerMinorityYN + tokenizer.eos_token
    return df

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
        if 'HITId' in dataset:
            seq2seq_tokenized = tokenizer(
                examples['text'],
                padding=pad_examples,
                truncation=True,
                max_length=max_length,
                return_tensors = "pt"
            )
        else:
            seq2seq_tokenized = tokenizer(
                examples['post'],
                padding=pad_examples,
                truncation=True,
                max_length=max_length,
                return_tensors = "pt"
            )

        with tokenizer.as_target_tokenizer():
            target_tokenized = tokenizer(
                examples['targetStereotype'],
                padding=pad_examples,
                truncation=True,
                max_length=max_length,
                return_tensors = "pt"
            )
        process_labels(target_tokenized)
        return {**seq2seq_tokenized, **target_tokenized}

    if dataset_type == 'sbic':
        remove_cols = ['HITId', 'text', 'targetStereotype']
    else:
        remove_cols = ['ID', 'post', 'targetStereotype']
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

def clean_target(df, target_col='targetStereotype', train=True):
    df[target_col] = df[target_col].replace(np.nan, '', regex=True)
    
    if not train:
        # lower case for testing. for training; doesn't matter.
        df[target_col] = df[target_col].str.lower()
        if 'HITId' in df.columns:
            df = df.groupby(['HITId', 'post'], as_index=False).agg({target_col:set})
        df[target_col] = df[target_col].apply(lambda x: list(x))
    
    df.rename(columns={target_col:'targetStereotype'}, inplace=True)

    if 'HITId' in df.columns:
        return df[['HITId','post','text','targetStereotype']]
    return df[['ID','post','targetStereotype']]

def combine_attrs_post(df, sep_token):
    df = df.fillna('')
    df.post = df.post + sep_token + df.attribute + sep_token + df.targetMinority
    return df

if __name__ == "__main__":
    args = parse_args()
    set_seed(685)

    if args.dataset_type == 'sbic':    
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        
        model.cuda()
        print(model.device)

        categorical_tokens = [LEWDY,LEWDN,OFFY,OFFN,INTY,INTN,GRPY,GRPN,INGY,INGN]
        special_tokens = {'additional_special_tokens': categorical_tokens}
        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))

        df_train = pd.read_csv(args.data_file)
        df_train = categorize_var(df_train)
        df_train = create_sbic_text_column(df_train, tokenizer)
        df_train = clean_post(df_train)
        df_train = clean_target(df_train)
        df_post = df_train[['HITId', 'post']]
        df_post = df_post.drop_duplicates().reset_index(drop=True)
        df_train = df_post.merge(df_train.drop(columns='post'), on='HITId', validate='one_to_many')

    else:
        df_raw = pd.read_csv(args.data_file, sep="\t")
        df_posts = pd.read_csv(args.data_file, sep="\t")
        helper_df = pd.read_csv("../data/implicit_hate_v1_stg2.tsv", sep = "\t")
        
        dict_train = {"ID": [], "post": [], "attribute": [], "targetMinority": [], "targetStereotype": []}
        
        for i in range(df_posts.shape[0]):
            if df_raw["ID"][i] in helper_df["ID"].values:
                dict_train["ID"].append(df_raw["ID"].values[i])
                dict_train["post"].append(df_posts["post"].values[i])
                dict_train["attribute"].append(helper_df["implicit_class"].loc[helper_df.ID == df_raw["ID"].values[i]].values[0])
                dict_train["targetMinority"].append(df_posts["target"].values[i])
                dict_train["targetStereotype"].append(df_posts["implied_statement"].values[i])
        
        df_train = pd.DataFrame(dict_train)
        
        all_attrs = df_train.attribute.unique()
        
        attr_to_tok = {}
        for attr in all_attrs:
            attr_to_tok[attr] = "<" + attr + ">"
        
        df_train["attribute"] = df_train.apply(lambda row: attr_to_tok[row["attribute"]], axis = 1)
        
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", additional_special_tokens = list(attr_to_tok.values()))
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        
        df_train = combine_attrs_post(df_train, tokenizer.sep_token)
        
        df_train = clean_post(df_train)
        df_train = clean_target(df_train)
        
        model.resize_token_embeddings(len(tokenizer))
        
        model.cuda()
    
    
    train_ds = Dataset.from_pandas(df_train)
    
    print(train_ds)

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