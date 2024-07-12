from datasets import Dataset
from knowledge_utils import *
from utils import *
from transformers import BartForConditionalGeneration
import os
import pickle
import pandas as pd
import torch
import argparse

SEQ2SEQ_MODEL_NAME = 'facebook/bart-base'

EDGE_DATA_FILE = '../data/conceptnet-assertions-5.7.0.csv'
EDGE_DICT_FILE = '../data/edge_dictionary.pickle'
EMB_DICT_FILE = '../data/embedding_dictionary.pickle'

BART_HIDDEN_SIZE = 768
EMBEDDING_SIZE = 300
MAX_LENGTH = 1024

PRED_DIRECTORY = 'pred/'
RESULTS_DIRECTORY = 'results/'

"""
For inference on the SBIC dataset use this:
```python test.py --model sbic_conceptnet_topk_model/<checkpoint-directory>
--output sbic_conceptnet_topk --knowledge_type top --knowledge_graph conceptnet
--dataset_type sbic  --data_file ../data/SBIC.v2.tst.csv --generate_scores```

For inference on the LatentHatred dataset use this:
```python test.py --model latent_conceptnet_topk_model/<checkpoint-directory>
--output latent_conceptnet_topk --knowledge_type top --knowledge_graph conceptnet
--dataset_type latent --data_file ../data/latenthatred_posts_test.tsv --generate_scores```

Tune the `knowledge_graph` and `knowledge_type` arguments appropriately for your experiments.
"""

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-m',
        '--model',
        required=False,
        help='The path for the checkpoint folder',
    )

    parser.add_argument(
        "-o",
        "--output",
        required=False,
        help="The name of the pickle file which will contain all predictions",
    )
    
    parser.add_argument('--knowledge_type', type=str, default='top', choices=['top', 'bot', 'rand'], help='Pass in a knowledge type.')
    parser.add_argument('--knowledge_graph', type=str, default='conceptnet', choices=['conceptnet', 'stereokg'], help='Pass in a knowledge type.')
    parser.add_argument('--dataset_type', type=str, default='sbic', choices=['sbic', 'latent'], help='Pass in a dataset type.')
    parser.add_argument("--get_ranking_scores", action='store_true', help = 'Pass true if you want to dump the cosine scores as well')

    parser.add_argument('--k', type=int, default=20, help='Pass in a value for k.')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size for testing. Use a smaller batch size with more classifiers.')
    parser.add_argument('--sep', type=str, default=',', help='Separator for data file.')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA for testing')
    parser.add_argument('--generate_scores', action='store_true', help='If True, will generate scores')
    parser.add_argument('--save_results_to_csv', action='store_true', help='If true, will save the generation results to csv.')
    parser.add_argument('--num_results', type=int, default=200, help='If saving results to csv, then this variable saves \'num_results\' samples.')
    parser.add_argument('--data_file', type=str, default='../data/SBIC.v2.tst.csv', help='Data File to load.')

    return parser.parse_args()

def check_args(args):
    model_path = args.model
    model_name = get_file_name(model_path)
    data_source = get_file_name(args.data_file)
    
    print(data_source, model_name, args.model, args.data_file)

    pickle_file = os.path.join(PRED_DIRECTORY, args.output + "_" + model_name + '_' + data_source + '.pickle')
    results_file = os.path.join(RESULTS_DIRECTORY, args.output + "_" + model_name + '_' + data_source + '.csv')
    
    print(pickle_file)
    
    if not(os.path.isfile(args.data_file)):
        raise ValueError('Must pass in an existing data file for training.')
    
    return model_path, pickle_file, results_file

if __name__ == '__main__':
    args = parse_args()
    
    model_path, pickle_file, results_file = None, None, None
    
    if not args.get_ranking_scores:
        model_path, pickle_file, results_file = check_args(args)
    
    if args.dataset_type == 'sbic':
        df = pd.read_csv(args.data_file, sep=args.sep, engine='python')
        df = clean_post(df)
        df = clean_target(df, train=False, target_col = "targetStereotype")
        df_post = df[['HITId', 'post']]
        df_post = df_post.drop_duplicates().reset_index(drop=True)
    else:
        df = pd.read_csv(args.data_file, sep=args.sep, engine='python') [['post', 'implied_statement']]
        df = clean_post(df)
        df = clean_target(df, train=False, target_col = "implied_statement")
    
    if not os.path.exists(pickle_file):
        print('processing edge file ...')
        if not os.path.exists(EDGE_DICT_FILE):
            edge_dict = process_edge_file(EDGE_DATA_FILE)
            pickle.dump(edge_dict, open(EDGE_DICT_FILE, 'wb'))
        else:
            edge_dict = pickle.load(open(EDGE_DICT_FILE, 'rb'))

        emb_dict = None

        print('finding top k tuples ...')
        if args.dataset_type == 'sbic':
            df_post = concat_top_k_tuples(df_post, edge_dict, knowledge_type = args.knowledge_type, knowledge_graph = args.knowledge_graph, emb_dict=emb_dict, k=args.k, get_scores = args.get_ranking_scores, ds=args.dataset_type)
            df_final = df_post.merge(df.drop(columns='post'), on='HITId', validate='one_to_many')

            results_cols=['HITId', 'post', 'target']
        else:
            df_final = concat_top_k_tuples(df, edge_dict, knowledge_type = args.knowledge_type, knowledge_graph = args.knowledge_graph, emb_dict=emb_dict, k=args.k, get_scores = args.get_ranking_scores, ds=args.dataset_type)

            results_cols=['post', 'target']

        if args.get_ranking_scores:
            exit(-1)

        dataset = Dataset.from_pandas(df_final)
        seq2seq_tok, tokenized = tokenize_data(dataset, train=False, padding=False, max_length=MAX_LENGTH)

        print('initializing model ...')
        model = BartForConditionalGeneration.from_pretrained(model_path)
        forward_method = model.get_encoder().forward
        torch_cols = ['input_ids', 'attention_mask']
        model.eval()
        if args.use_cuda and torch.cuda.is_available():
            model.cuda()

        print('running model tests ...')
        batch_iter = MinibatchIterator(tokenized, seq2seq_tok, batch_size=args.batch_size, torch_cols=torch_cols, use_cuda=args.use_cuda)
        generate_stereotypes(
            batch_iter,
            seq2seq_tok,
            model,
            forward_method,
            results_cols=results_cols,
            pickle_file=pickle_file,
        )

        if args.generate_scores or args.save_results_to_csv:
            print("generating base model scores ...")
            generate_scores(
                pickle_file,
                save_results_to_csv=args.save_results_to_csv,
                num_results=args.num_results,
                dataset_type=args.dataset_type,
            )
    else:
        if args.generate_scores or args.save_results_to_csv:
            print("generating base model scores ...")
            generate_scores(
                pickle_file,
                save_results_to_csv=args.save_results_to_csv,
                num_results=args.num_results,
                dataset_type=args.dataset_type,
            )