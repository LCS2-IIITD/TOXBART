import sys
sys.path.append(".")

import torch.utils

from knowledge_utils import *

from datasets import Dataset, DatasetDict
from transformers import BartForConditionalGeneration
from transformers.trainer_utils import set_seed
from transformers.data.data_collator import DataCollatorForSeq2Seq

import os
import pickle
import pandas as pd
import numpy as np
import argparse

SEQ2SEQ_MODEL_NAME = 'facebook/bart-base'

EDGE_DATA_FILE = '../data/conceptnet-assertions-5.7.0.csv'
EDGE_DICT_FILE = '../data/edge_dictionary.pickle'
EMB_DICT_FILE = '../data/embedding_dictionary.pickle'

BART_HIDDEN_SIZE = 768
EMBEDDING_SIZE = 300
MAX_LENGTH = 1024

"""
For Knowledge Models on SBIC dataset use this:
```python train.py --knowledge_type top \\
--knowledge_graph conceptnet --dataset_type sbic \\
--data_file ../data/SBIC.v2.trn.csv --output_dir sbic_conceptnet_topk_model```

For Knowledge Models on LatentHatred dataset use this:
```python train.py --knowledge_type top \\
--knowledge_graph conceptnet --dataset_type latent \\
--data_file ../data/latenthatred_posts_train.tsv --output_dir latent_conceptnet_topk_model```
"""

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--knowledge_type', type=str, default='top', choices=['top', 'bot', 'rand'], help='Pass in a knowledge type.')
    parser.add_argument('--knowledge_graph', type=str, default='conceptnet', choices=['conceptnet', 'stereokg'], help='Pass in a knowledge type.')
    parser.add_argument('--dataset_type', type=str, default='sbic', choices=['sbic', 'latent'], help='Pass in a dataset type.')
    parser.add_argument('--output_dir', type=str, default='model', help='Pass in a model output directory.')
   
    parser.add_argument('--seed', type=int, default=685, help='Pass in a seed value.')
    parser.add_argument('--batch_size', type=int, default=4, help='Pass in a batch size.')
    parser.add_argument('--k', type=int, default=20, help='Pass in a value for k.')
    parser.add_argument('--num_epochs', type=float, default=3.0, help='Pass in the number of training epochs.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Pass in the learning rate for training.')
    parser.add_argument('--warmup_ratio', type=float, default=0.5, help='Warmup ratio to be applied if training for 1 epoch.')
    parser.add_argument('--sep', type=str, default=',', help='Separator for data file.')
    parser.add_argument('--data_file', type=str, default='../data/SBIC.v2.trn.csv', help='Data File to load.')
    parser.add_argument('--dev_file', type=str, help='Dev File to load in case data split isn\'t used.') # we do not use the dev file here, but ../data/SBIC.v2.dev.csv can be used

    return parser.parse_args()

def check_args(args):
    if not(os.path.isfile(args.data_file)):
        raise ValueError('Must pass in an existing data file for training.')

def process_data(args, data_file):
    print('loading and tokenizing data ...')

    if args.dataset_type == 'sbic':
        df = pd.read_csv(data_file, sep=args.sep, engine='python')
        df = clean_post(df)
        df = clean_target(df, target_col = "targetStereotype")
        df_post = df[['HITId', 'post']]
        df_post = df_post.drop_duplicates().reset_index(drop=True)
    else:
        df = pd.read_csv(data_file, sep=args.sep, engine='python') [['post', 'implied_statement']]
        df = clean_post(df)
        df = clean_target(df, target_col = "implied_statement")
    
    print('processing edge file ...')
    if not os.path.exists(EDGE_DICT_FILE):
        edge_dict = process_edge_file(EDGE_DATA_FILE)
        pickle.dump(edge_dict, open(EDGE_DICT_FILE, 'wb'))
    else:
        edge_dict = pickle.load(open(EDGE_DICT_FILE, 'rb'))

    emb_dict = None

    print('finding top k tuples ...')
    if args.dataset_type == 'sbic':
        df_post = concat_top_k_tuples(df_post, edge_dict, knowledge_type = args.knowledge_type, knowledge_graph = args.knowledge_graph, emb_dict=emb_dict, k=args.k, ds=args.dataset_type)
        df_final = df_post.merge(df.drop(columns='post'), on='HITId', validate='one_to_many')
    else:
        df_final = concat_top_k_tuples(df, edge_dict, knowledge_type = args.knowledge_type, knowledge_graph = args.knowledge_graph, emb_dict=emb_dict, k=args.k, ds=args.dataset_type)

    dataset = Dataset.from_pandas(df_final)
    return dataset

if __name__ == '__main__':
    args = parse_args()
    check_args(args)
    print(args)
    set_seed(args.seed)
    
    dataset = process_data(args, args.data_file)
    if args.dev_file is not None:
        dev_dataset = process_data(args, args.dev_file)
        datasets = DatasetDict({"train": dataset, "test": dev_dataset})
    else:
        datasets = dataset.train_test_split(test_size=0.2, shuffle=True)
    
    print('tokenizing data ...')
    tokenizer, tokenized = tokenize_data(datasets, padding=False, max_length=MAX_LENGTH)

    print('initializing model ...')
    model = BartForConditionalGeneration.from_pretrained(SEQ2SEQ_MODEL_NAME)
    model.train()
    if torch.cuda.is_available():
        model.cuda()
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model, padding=True, max_length=MAX_LENGTH)
    train(
        model,
        tokenized,
        model_name = args.output_dir,
        data_collator=data_collator,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio
    )

