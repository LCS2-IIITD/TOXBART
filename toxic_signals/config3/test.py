import sys
sys.path.append("..")

from modeling_toxbart import ToxBARTC1
from tox_bert.modeling_toxbert import BertToxicityRegressor
from transformers import BertTokenizer, BartTokenizer, Trainer, DataCollatorWithPadding, BartForConditionalGeneration
import pandas as pd
import pickle, torch, os, random, csv, argparse
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge import Rouge
from datasets import load_metric

"""
For inference on the SBIC dataset use this:
```python test.py --model sbic_config3_model/<checkpoint-directory> \\
--output sbic_config3 --dataset_type sbic --tox_model_dir ../tox_bert/bert-toxic-signals-probab/<tox-bert-checkpoint-folder> \\
--data_file ../../data/SBIC.v2.tst.csv --generate_scores```

For inference on the LatentHatred dataset use this:
```python test.py --model latent_config3_model/<checkpoint-directory> \\
--output latent_config3 --dataset_type latent --tox_model_dir ../tox_bert/bert-toxic-signals-probab/<tox-bert-checkpoint-folder> \\
--data_file ../../data/latenthatred_posts_test.tsv --generate_scores```
"""

NUM_BEAMS = 10
PRED_DIR = 'pred/'
RESULTS_DIR = 'results/'

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
    parser.add_argument('--tox_model_dir', type=str, help='Pass in the toxicity regressor model output directory.')

    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size for testing. Use a smaller batch size with more classifiers.')
    parser.add_argument('--save_results_to_csv', action='store_true', help='If true, will save the generation results to csv.')
    parser.add_argument('--num_results', type=int, default=200, help='If saving results to csv, then this variable saves \'num_results\' samples.')
    parser.add_argument('--data_file', type=str, default='../../data/SBIC.v2.tst.csv', help='Data File to load.')

    return parser.parse_args()

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
    input_preds = trainer.predict(Dataset.from_dict({"input_ids": ds["input_ids"]}))
    
    return torch.sigmoid(torch.from_numpy(input_preds.predictions))

def get_toxicity_probabilities(df, tox_model_dir):
    print('getting toxicity probabilities')
    def tox_tokenize_data(
        dataset,
        tokenizer,
        padding=True,
        max_length=512,
    ):
        def tokenize(examples):
            pad_examples = "max_length" if padding else False
            
            seq2seq_tokenized = tokenizer(
                examples['post'],
                padding=pad_examples,
                truncation=True,
                max_length=max_length,
                return_tensors = "pt"
            )
            
            return seq2seq_tokenized

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

    tox_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_tox_ds = Dataset.from_pandas(df)
    tokenized_tox_ds = tox_tokenize_data(train_tox_ds, tox_tokenizer)

    input_signals = inference(tokenized_tox_ds, tox_tokenizer, tox_model_dir)

    input_signals = input_signals.unsqueeze(1)
    print(input_signals.shape)
    input_signals = input_signals.repeat(1, NUM_BEAMS, 1)
    print(input_signals.shape)

    return input_signals

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
            col_names = ['HITId', 'post', 'target', 'prediction']
            for idx in indices[:num_results]:
                results_csv.append([
                    results[col_names[0]][idx],
                    results[col_names[1]][idx].replace('\n', ' '),
                    ', '.join(results[col_names[2]][idx]),
                    results[col_names[3]][idx],
                ])
        else:
            col_names = ['post', 'target', 'prediction']
            for idx in indices[:num_results]:
                results_csv.append([
                    results[col_names[0]][idx].replace('\n', ' '),
                    ''.join(results[col_names[1]][idx]),
                    results[col_names[2]][idx],
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

def tokenize_data(
    dataset,
    tokenizer,
    padding=True,
    max_length=1024,
):
    def tokenize(examples):
        pad_examples = "max_length" if padding else False
        
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
    return tokenized, tokenizer


def clean_post(df):
    df.post = df.post.str.replace(r'\bRT\b', ' ', regex=True)
    df.post = df.post.str.replace('(@[^\s]*\s|\s?@[^\s]*$)', ' ', regex=True)
    df.post = df.post.str.replace('https?://[^\s]*(\s|$)',' ',regex=True)
    df.post = df.post.str.strip()
    return df

def clean_target(df, target_col='targetStereotype', train=True):
    df[target_col] = df[target_col].replace(np.nan, '', regex=True)

    if not train:
        df[target_col] = df[target_col].str.lower()
        if target_col == 'targetStereotype':
            df = df.groupby(['HITId', 'post'], as_index=False).agg({target_col:set})
        df[target_col] = df[target_col].apply(lambda x: list(x))

    df.rename(columns={target_col: 'target'}, inplace=True)
    if target_col == 'targetStereotype':
        return df[['HITId','post','target']]
    return df[['post', 'target']]

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
    model_kwargs = {'encoder_outputs': encoder_outputs, "toxic_signals": batch["toxic_signals"]}
    output_ids = model.generate(batch['input_ids'], num_beams=num_beams, length_penalty=5.0, **model_kwargs)

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

if __name__ == "__main__":
    args = parse_args()
    data_source = "test"
    pickle_file = os.path.join(PRED_DIR, args.output + '_' + data_source + '.pickle')
    results_file = os.path.join(RESULTS_DIR, args.output + '_' + data_source + '.csv')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = ToxBARTC1.from_pretrained(args.model)

    model.to(device)

    if args.dataset_type == 'sbic':
        df = pd.read_csv(args.data_file, sep=",", engine='python')
        df = clean_post(df)
        df = clean_target(df, target_col = "targetStereotype", train = False)
        df_post = df[['HITId', 'post']]
        df_post = df_post.drop_duplicates().reset_index(drop=True)
        df = df_post.merge(df.drop(columns='post'), on='HITId', validate='one_to_many')
        results_cols = ['HITId','post','target']
    else:
        df = pd.read_csv(args.data_file, sep="\\t", engine='python')[['post', 'implied_statement']]
        df = clean_post(df)
        df = clean_target(df, target_col = "implied_statement", train=False)
        results_cols = ['post','target']
    
    test_ds = Dataset.from_pandas(df)

    input_signals = get_toxicity_probabilities(df)
    signals_ds = Dataset.from_dict({"toxic_signals": input_signals})
    
    test_ds = concatenate_datasets([test_ds, signals_ds], axis = 1)

    tokenized_ds, tokenizer = tokenize_data(test_ds, tokenizer, padding=True, max_length=1024)
    
    print(tokenized_ds)

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
        save_results_to_csv=args.save_results_to_csv,
        num_results=200,
        save_file=results_file,
        dataset_type=args.dataset_type,
    )