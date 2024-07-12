import math
import torch
import os
import pickle
import random
import csv
import numpy as np
import pandas as pd
from datasets import load_metric
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, GPT2Tokenizer
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge import Rouge
from tqdm import tqdm
import torch.nn as nn

# Language Generation Utils
def get_bleu_score(references, hypotheses, return_all_scores=False):
    #tokenized_hypotheses = list(map(str.split, hypotheses))
    #tokenized_references = list(map(lambda s: list(map(str.split, s)), references))
    
    bleu = np.empty((len(hypotheses), 2))
    for i,hyp in enumerate(hypotheses):
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
          scores = rouge.get_scores(hyp, ref)
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
    for i,_ in enumerate(hypotheses):
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

# Testing Utils
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

def generate_stereotype(
    batch,
    tokenizer,
    model,
    model_enc_func,
    encoder_input_cols=['input_ids','attention_mask'],
):
    num_beams = 10
    enc_input = []
    
    for col in encoder_input_cols:
      enc_input.append(batch[col])
    
    encoder_outputs = model_enc_func(*enc_input, return_dict=True)
    model_kwargs = {'encoder_outputs': encoder_outputs}
    output_ids = model.generate(batch['input_ids'], num_beams=num_beams, length_penalty=5.0, **model_kwargs)

    input_strs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
    output_strs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    return input_strs, output_strs

def generate_scores(pickle_file, save_results_to_csv=False, generation_seed=756, num_results=200, save_file='results.csv', dataset_type = "sbic"):
    results = pickle.load(open(pickle_file, 'rb+'))
    
    references = results['target'].tolist()
    if dataset_type == "latent":
        references = [[''.join(i).lower()] for i in references]
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


# Minibatch Iterator Class
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
        next_batch = self.tokenizer.pad(next_batch, padding=True, max_length=self.max_length)
        
        for col in self.torch_cols:
            next_batch[col] = torch.tensor(next_batch[col])
            if self.use_cuda:
                next_batch[col] = next_batch[col].cuda()
        
        self.current_pos = self.next_pos
        return next_batch

def get_file_name(file_path, remove_extension=True):
    file_name = os.path.basename(os.path.normpath(file_path))
    if remove_extension:
        return os.path.splitext(file_name)[0]
    return file_name

def get_step_variables(
    num_rows,
    num_epochs,
    batch_size,
    eval_percent=5.0,
    warmup_ratio=0.5
):
    if num_epochs == 1:
        one_epoch_steps = math.ceil(num_rows / batch_size)
        save_steps = one_epoch_steps // 2
        eval_steps = (save_steps * eval_percent) // 100
    else:
        one_epoch_steps = math.ceil(num_rows / batch_size)
        save_steps = (one_epoch_steps * num_epochs) // 2
        eval_steps = (one_epoch_steps * num_epochs * 5.0) // 100
    
    if num_epochs == 1:
        warmup_steps = one_epoch_steps * warmup_ratio
    else:
        warmup_steps = one_epoch_steps
    return warmup_steps, save_steps, eval_steps

def train(
    model,
    datasets,
    model_name = "model",
    num_epochs=3.0,
    learning_rate=5e-6,
    batch_size=4,
    eval_percent=10.0,
    data_collator=None,
    warmup_ratio=0.5 # As of now this only applies to the single epoch case.
):
    num_rows = datasets['train'].num_rows
    warmup_steps, save_steps, eval_steps = get_step_variables(
        num_rows,
        num_epochs,
        batch_size,
        eval_percent=eval_percent,
        warmup_ratio=warmup_ratio
    )
    
    eval_steps = 5000
    print("Linear Warm Up: ", warmup_steps)
    print("Save Steps: ", save_steps)
    print("Eval Steps: ", eval_steps)

    training_args = TrainingArguments(
        output_dir = model_name,
        evaluation_strategy = 'steps',
        eval_steps = eval_steps,
        logging_steps = eval_steps,
        save_steps = save_steps,
        save_total_limit = 1,
        warmup_steps = warmup_steps,
        learning_rate = learning_rate,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        num_train_epochs = num_epochs,
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

# Warning Overwrite
def custom_warning(msg, *args, **kwargs):
    return 'WARNING: ' + str(msg) + '\n'

# Dataset Utils

# Data Cleaning/Preprocessing

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

# Tokenization Utils
def tokenize_data(
    dataset,
    train=True,
    padding=True,
    max_length=128,
    special_tokens=[],
):
    def process_labels(target_tokenized):
        target_tokenized['labels'] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in target_tokenized['input_ids']
        ]
      
        del target_tokenized['input_ids']
        del target_tokenized['attention_mask']

    def tokenize(examples):
        pad_examples = "max_length" if padding else False
        
        seq2seq_tokenized = tokenizer(
            examples['post'],
            padding=pad_examples,
            truncation=True,
            max_length=max_length,
        )

        if train:
            with tokenizer.as_target_tokenizer():
                target_tokenized = tokenizer(
                    examples['target'],
                    padding=pad_examples,
                    truncation=True,
                    max_length=max_length,
                )
            process_labels(target_tokenized)
            return {**seq2seq_tokenized, **target_tokenized}
        return seq2seq_tokenized

    #### get_tokenized_data function body
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", additional_special_tokens=special_tokens)
 
    if train:
        if 'HITId' in dataset:
            remove_cols = ['HITId', 'post', 'target']
        else:
            remove_cols = ['post', 'target']
        tokenized = dataset.map(
            tokenize, batched=True,
            num_proc=1,
            remove_columns=remove_cols
        )
    else:
        tokenized = dataset.map(
            tokenize, batched=True,
            num_proc=1,
        )

    return tokenizer, tokenized

# Dataset Statistics Helpers
def generate_length_distribution(dataset, partition=20):
    input_tokens = dataset['input_ids']
    lengths = list(map(len, input_tokens))
    lengths.sort()

    ntiles = []
    for i in range(partition):
      ntiles.append(float(i / partition) * 100)
    
    print('Percentile: ', np.percentile(lengths, ntiles))

