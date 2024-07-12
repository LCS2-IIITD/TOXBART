from transformers import BartTokenizer, BartForConditionalGeneration
import pandas as pd
import pickle, torch, os, random, csv, argparse
from tqdm import tqdm
from datasets import Dataset
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge import Rouge
from datasets import load_metric
import torch.nn as nn

"""
For inference on the SBIC dataset use this:
```python test.py --model sbic_config2_model/<checkpoint-directory> --output sbic_config2 --dataset_type sbic \\
--data_file ../../data/SBIC.v2.tst.csv --generate_scores```

For inference on the LatentHatred dataset use this:
```python test.py --model latent_config2_model/<checkpoint-directory> --output latent_config2 --dataset_type latent \\
--data_file ../../data/latenthatred_raw_test.tsv --generate_scores```
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

    parser.add_argument(
        '-m',
        '--model',
        required=True,
        help='Path to the checkpoint folder',
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Name of the pickle file which will contain all predictions",
    )

    parser.add_argument('--dataset_type', type=str, default='sbic', choices=['sbic', 'latent'], help='Pass in a dataset type.')

    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size for testing. Use a smaller batch size with more classifiers.')
    parser.add_argument('--save_results_to_csv', action='store_true', help='If true, will save the generation results to csv.')
    parser.add_argument('--num_results', type=int, default=200, help='If saving results to csv, then this variable saves \'num_results\' samples.')
    parser.add_argument('--data_file', type=str, default='../../data/SBIC.v2.tst.csv', help='Data File to load.')

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
    
def setup_tokenizer(model_name):
    tokenizer = BartTokenizer.from_pretrained(model_name, use_fast=True)
    categorical_tokens = [LEWDY,LEWDN,OFFY,OFFN,INTY,INTN,GRPY,GRPN,INGY,INGN]
    special_tokens = {'additional_special_tokens': categorical_tokens}
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer

# Language Generation Utils
def get_bleu_score(references, hypotheses, return_all_scores=False):
    bleu = np.empty((len(hypotheses), 2))
    for i, hyp in enumerate(hypotheses):
        bleu_ref = np.empty(len(references[i]))
        for j,ref in enumerate(references[i]):
            if len(ref) == 0 and len(hyp) == 0:
                bleu_ref[j] = 1.0
            elif len(ref) == 0 and len(hyp) != 0:
                bleu_ref[j] = 0.0
            elif len(ref) != 0 and len(hyp) == 0:
                bleu_ref[j] = 0.0
            else:
                bleu_ref[j] = sentence_bleu([ref], hyp, weights=(0.5, 0.5))
        bleu[i] = [np.max(bleu_ref), np.average(bleu_ref)]
    
    if return_all_scores:
        return bleu
    else:
        return np.average(bleu, axis=0)

def get_rouge_scores(references, hypotheses, return_all_scores=False):
    rouge_scores = np.empty((len(hypotheses), 2, 3))
    rouge = Rouge(metrics=['rouge-l'])

    for i, hyp in enumerate(hypotheses):
        ref_scores = np.empty((len(references[i]), 3))
        for j, ref in enumerate(references[i]):
            if len(ref) == 0 and len(hyp) == 0:
                scores = [{'rouge-l': {'f': 1.0, 'p': 1.0, 'r': 1.0}}]
            elif len(ref) == 0 and len(hyp) != 0:
                scores = [{'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}]
            elif len(ref) != 0 and len(hyp) == 0:
                scores = [{'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}]
            else:
                scores = [rouge.get_scores(hyp, ref)]
            ref_scores[j, 0] = scores[0]['rouge-l']['p']
            ref_scores[j, 1] = scores[0]['rouge-l']['r']

            if ref_scores[j, 0] + ref_scores[j, 1] == 0.0:
                ref_scores[j, 2] = 0.0
            elif np.isnan(ref_scores[j, 0]):
                ref_scores[j, 2] = np.nan
            else:
                ref_scores[j, 2] = 2 * ((ref_scores[j, 0] * ref_scores[j, 1]) / \
                                  (ref_scores[j, 0] + ref_scores[j, 1]))

        max_j = np.argmax(ref_scores, axis=0)[2]
        rouge_scores[i,0,:] = ref_scores[max_j]
        rouge_scores[i,1,:] = np.average(ref_scores, axis=0)

    if return_all_scores:
        return rouge_scores
    else:
        return np.average(rouge_scores, axis=0)

def get_bert_score(bert_scores, hypotheses, references, return_all_scores=False):
    for i, _ in enumerate(hypotheses):
        if len(hypotheses[i]) == 0:
            if len(references[i]) == 1:
                if len(references[i][0]) == 0:
                    bert_scores['precision'][i] = 1.0
                    bert_scores['recall'][i] = 1.0
                    bert_scores['f1'][i] = 1.0
                else:
                    bert_scores['precision'][i] = 0.0
                    bert_scores['recall'][i] = 0.0
                    bert_scores['f1'][i] = 0.0
            else:
                bert_scores['precision'][i] = 0.0
                bert_scores['recall'][i] = 0.0
                bert_scores['f1'][i] = 0.0
        elif len(references[i]) == 1:
            if len(references[i][0]) == 0:
                bert_scores['precision'][i] = 0.0
                bert_scores['recall'][i] = 0.0
                bert_scores['f1'][i] = 0.0
    
    precision = np.average(bert_scores['precision'])
    recall = np.average(bert_scores['recall'])
    f1 = np.average(bert_scores['f1'])
    #f1 = 2 * (precision * recall) / (precision + recall)
    if return_all_scores:
        return bert_scores
    else:
        return precision, recall, f1

def generate_scores(pickle_file, save_results_to_csv=False, generation_seed=756, num_results=200, save_file='results.csv', dataset_type = "sbic"):
    results = pickle.load(open(pickle_file, 'rb+'))
    references = results['targetStereotype'].tolist()
    if dataset_type == "latent":
       references = [[''.join(i)] for i in references]
    hypotheses = results['prediction'].tolist()
    
    if not(save_results_to_csv):
        print(references[:10], hypotheses[:10])
        bleu_score_max, bleu_score_avg = get_bleu_score(references, hypotheses)
        rouge_scores_max, rouge_scores_avg = get_rouge_scores(references, hypotheses)
        
        metric = load_metric('bertscore')
        bert_scores = metric.compute(predictions=hypotheses, references=references, lang='en')
        bert_score = get_bert_score(bert_scores, hypotheses, references)

        print("Bleu Score (Avg): ", round(bleu_score_avg, 4))
        print("Bleu Score (Max): ", round(bleu_score_max, 4))
        print("Rouge Score (Avg) (Precision, Recall, F1): ", [round(i, 4) for i in rouge_scores_avg])
        print("Rouge Score (Max) (Precision, Recall, F1): ", [round(i, 4) for i in rouge_scores_max])
        print('BERT Score (Max) (Precision, Recall, F1): ', [round(i, 4) for i in bert_score])
    
    else:
        indices = list(range(len(references)))
        random.seed(generation_seed)
        random.shuffle(indices)

        results_csv = []

        if dataset_type == "sbic":
            col_names = ['HITId', 'text', 'targetStereotype', 'prediction']
        else:
            col_names = ['ID', 'post', 'targetStereotype', 'prediction']
        
        for idx in indices[:num_results]:
            results_csv.append([
                results[col_names[0]][idx],
                results[col_names[1]][idx].replace('\n', ' '),
                ', '.join(results[col_names[2]][idx]),
                results[col_names[3]][idx],
            ])

        with open(save_file, 'w') as f:
            csv_writer = csv.writer(f, delimiter='|')
            csv_writer.writerow(col_names)
            csv_writer.writerows(results_csv)

class MinibatchIterator:
    # torch_cols must be in the same order that the model accepts the arguments.
    def __init__(self, data, tokenizer, batch_size=16, max_length=1024, use_cuda=True, torch_cols=['input_ids', 'attention_mask']):
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_cuda = use_cuda
        self.torch_cols = torch_cols
        self.current_pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_pos == self.data.num_rows:
                raise StopIteration

        self.next_pos = min(self.data.num_rows, self.current_pos + self.batch_size)
        next_batch = self.data[self.current_pos:self.next_pos]
        # next_batch = self.tokenizer.pad(next_batch, padding=True, max_length=self.max_length)
        
        for col in self.torch_cols:
            next_batch[col] = torch.tensor(next_batch[col])
            if self.use_cuda:
                next_batch[col] = next_batch[col].cuda()
        
        self.current_pos = self.next_pos
        return next_batch

def tokenize_textgen_df(
    dataset,
    tokenizer,
    dataset_type = "sbic",
    padding=True,
    max_length=1024,
):
    def tokenize(examples):
        pad_examples = "max_length" if padding else False

        if dataset_type == "sbic":
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
        return seq2seq_tokenized

    tokenized = dataset.map(
        tokenize, batched=True,
        num_proc=1,
    )
    return tokenized

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

def generate_stereotype(
    batch,
    tokenizer,
    model,
    model_enc_func,
    encoder_input_cols=['input_ids', 'attention_mask'],
):
    num_beams = 10
    enc_input = []

    for col in encoder_input_cols:
        enc_input.append(batch[col])

    encoder_outputs = model_enc_func(*enc_input, return_dict=True)
    model_kwargs = {'encoder_outputs': encoder_outputs}
    output_ids = model.generate(batch['input_ids'], num_beams=num_beams, length_penalty = 5.0, **model_kwargs)

    input_strs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
    output_strs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    return input_strs, output_strs

def generate_stereotypes(
    batch_iter,
    tokenizer,
    model,
    model_enc_func,
    results_cols=['HITId', 'post'],
    pickle_file=None,
    num_views=6,
    view_model=False
):
    if os.path.isfile(pickle_file):
      results = pickle.load(open(pickle_file, 'rb'))
      return results

    results = [[] for _ in range(len(results_cols) + 1)]
    
    for batch in tqdm(batch_iter):
        _, output_strs = generate_stereotype(
            batch,
            tokenizer,
            model,
            model_enc_func,
            batch_iter.torch_cols,
            num_views,
            view_model,
        )
        for i,col in enumerate(results_cols):
            results[i].extend(batch[col])
        results[i + 1].extend([output_str.lower() for output_str in output_strs])

    df_dict = {}
    for i,col in enumerate(results_cols):
      df_dict[col] = results[i]
    df_dict['prediction'] = results[-1]
    
    results = pd.DataFrame(df_dict)
    if pickle_file is not None:
      pickle.dump(results, open(pickle_file, 'wb'))
    
    return results

def combine_attrs_post(df, sep_token):
    df = df.fillna('')
    df.post = df.post + sep_token + df.attribute + sep_token + df.targetMinority
    return df

if __name__ == "__main__":
    args = parse_args()
    data_source = "test"
    pickle_file = 'pred/' + args.output + '_' + data_source + '.pickle'
    results_file = 'results/' + args.output + '_' + data_source + '.csv'

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = BartForConditionalGeneration.from_pretrained(args.model)
    model.to(device)

    if args.dataset_type == "sbic":
        tokenizer = setup_tokenizer("facebook/bart-base")

        df = pd.read_csv(args.data_file, engine = 'python')

        df = categorize_var(df)
        df = create_sbic_text_column(df, tokenizer)
        df = clean_post(df)
        df = clean_target(df, train=False)
        df_post = df[['HITId', 'post']]
        df_post = df_post.drop_duplicates().reset_index(drop=True)
        df = df_post.merge(df.drop(columns='post'), on='HITId', validate='one_to_many')

        results_cols=['HITId','text','targetStereotype']
    
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
        
        df = pd.DataFrame(dict_train)

        all_attrs = df.attribute.unique()
        attr_to_tok = {}
        for attr in all_attrs:
            attr_to_tok[attr] = "<" + attr + ">"

        df["attribute"] = df.apply(lambda row: attr_to_tok[row["attribute"]], axis = 1)
        
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", additional_special_tokens = list(attr_to_tok.values()))
        
        df = combine_attrs_post(df, tokenizer.sep_token)
        df = clean_post(df)
        df = clean_target(df)
        
        model.resize_token_embeddings(len(tokenizer))

        results_cols=['ID', 'post', 'targetStereotype']

    test_ds = Dataset.from_pandas(df)

    tokenized_ds = tokenize_textgen_df(test_ds, tokenizer, dataset_type=args.dataset_type, padding=True, max_length=1024)

    torch_cols = ['input_ids', 'attention_mask']
    batch_iter = MinibatchIterator(tokenized_ds, tokenizer, batch_size=args.batch_size, torch_cols=torch_cols, use_cuda=True)

    model.eval()

    generate_stereotypes(
        batch_iter,
        tokenizer,
        model,
        model.get_encoder().forward,
        results_cols=results_cols,
        pickle_file=pickle_file,
    )
    
    generate_scores(
        pickle_file,
        save_results_to_csv=False,
        num_results=200,
        save_file=results_file
    )