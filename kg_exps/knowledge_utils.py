import sys
sys.path.append(".")

from utils import *
from stereokg_utils import *

from tqdm import tqdm
from transformers import BartForConditionalGeneration,AutoModelForCausalLM,AutoTokenizer
from datasets import Dataset, load_metric
from collections import defaultdict
from nltk.corpus import stopwords
from nltk import word_tokenize,tokenize
from nltk.stem import WordNetLemmatizer
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
import string
import torch
import pickle
import os
import random
import numpy as np
import pandas as pd

SEQ2SEQ_MODEL_NAME = 'facebook/bart-base'

EDGE_DATA_FILE = '../data/conceptnet-assertions-5.7.0.csv'

BART_HIDDEN_SIZE = 768
EMBEDDING_SIZE = 300
MAX_LENGTH = 1024

ADJECTIVE_TAGS = {'JJ','JJR','JJS'}
NOUN_TAGS = {'NN','NNS','NNP','NNPS'}
VERB_TAGS = {'VB','VBG','VBN','VBZ'}
STOPWORDS = stopwords.words('english')

SEP_TOKEN = '</s>'

REL_DICT = {
    'RelatedTo': ' is related to ',
    'FormOf': ' is a form of ',
    'IsA': ' is a ',
    'PartOf': ' is a part of ',
    'HasA': ' has a ',
    'UsedFor': ' is used for ',
    'CapableOf': ' is capable of ',
    'AtLocation': ' is at ',
    'Causes': ' causes ',
    'HasSubevent': ' happens with ',
    'HasFirstSubevent': ' begins with ',
    'HasLastSubevent': ' concludes with ',
    'HasPrerequisite': ' requires ',
    'HasProperty': ' has property ',
    'MotivatedByGoal': ' is motivated by the goal ',
    'ObstructedBy': ' is obstructed by ',
    'Desires': ' desires ',
    'CreatedBy': ' is created by ',
    'Synonym': ' is similar in meaning to ',
    'Antonym': ' is opposite in meaning to ',
    'DistinctFrom': ' is distinct from ',
    'DerivedFrom': ' is derived from ',
    'SymbolOf': ' symbolically represents ',
    'DefinedAs': ' is said to be ',
    'MannerOf': ' is a type of ',
    'LocatedNear': ' is typically near ',
    'HasContext': ' is used in the context of ',
    'SimilarTo': ' is similar to ',
    'EtymologicallyRelatedTo': ' has a similar origin to ',
    'EtymologicallyDerivedFrom': ' is derived in meaning from ',
    'CausesDesire': ' causes someone to desire ',
    'MadeOf': ' is made of ',
    'ReceivesAction': ' can receive ',
    'ExternalURL': ' can be understood further at ',
    'InstanceOf': ' is an example of ',
    'Entails': ' entails that ',
    'NotDesires': ' does not desire ',
    'NotUsedFor': ' is not used for ',
    'NotCapableOf': ' is not capable of ',
    'NotHasProperty': ' does not have property ',
}

PROMPTS = [
  'were known for', 'were described as', 'were regarded as',
  'were thought of as', 'were well-known for', 'were seen as a',
  'have a reputation for', 'would frequently', 'worked as',
  'have a job as', 'have a part-time job as', 'earned money by',
  'started working as', 'have various hobbies such as', 'would regularly engage in',
  'frequently talked about', 'behaved as though', 'liked to',
]


# Explicit Knowledge Utils

class LemmaTokenizer:
    def __init__(self):
      self.ignore = {',', '.', ';', ':', '"', '``', "''", '`'}.union(STOPWORDS)
      self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
      return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore]

def add_queries(query_lists, query_tokens, post, lemmatizer):
    post_tok = word_tokenize(post)
    pos_tags = nltk.pos_tag(post_tok)
    query_lists.append([])
    
    for tag_tuple in pos_tags:
      tag = tag_tuple[1]
      word = tag_tuple[0]
      
      pos = ''
      if tag in ADJECTIVE_TAGS:
        pos = 'a'
      elif tag in NOUN_TAGS:
        pos = 'n'
      elif tag in VERB_TAGS:
        pos = 'v'

      if pos != '':
        word = lemmatizer.lemmatize(word, pos)
        if tag != 'NNP' and tag != 'NNPS':
          word = word.lower()
        
        if word not in STOPWORDS:
          query_lists[-1].append(word)
          query_tokens.add(word)
    
def process_edge_file(edge_data_file):
    edge_dict = defaultdict(list)
    with open(edge_data_file, 'r') as edge_data:
      for line in tqdm(edge_data):
        line = line.strip().split('\t')
        start = line[2].split('/')
        end = line[3].split('/')
        
        if start[2] == 'en' and end[2] == 'en':
          node_info = literal_eval(line[4])
          relation = line[1].split('/')
          
          if relation[2] == 'dbpedia':
            continue
          
          edge_dict[start[3]].append((relation[2], node_info['weight'], end[3]))
    return edge_dict

def process_embedding_file(emb_data_file):
    emb_dict = defaultdict(list)
    with open(emb_data_file, 'r') as emb_file:
      txt = emb_file.readline()
      for line in tqdm(emb_file):
        line = line.split(' ')
        line[-1] = line[-1].strip()
        
        key = line[0]
        weights = list(map(float, line[1:]))
        emb_dict[key] = weights
     
    return emb_dict

def get_idf(posts, vocabulary):
    vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), vocabulary=vocabulary)
    vectorizer.fit_transform(posts)
    return dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

def collect_edges(query_tokens, idf, edge_dict, emb_dict=None):
    query_dict = defaultdict(list)

    for query in tqdm(list(query_tokens)):
      idf_score = idf[query]
      
      for edge_info in edge_dict[query]:
        score = idf_score * edge_info[1]
        
        if emb_dict is not None:
          if query not in emb_dict or edge_info[2] not in emb_dict:
            continue
        
        query_dict[query].append((score, query, edge_info[0], edge_info[2]))
    
    return query_dict

def collect_ordered_k_tuples(query_lists, query_dict, all_tuples, k=20, k_type = "top"):
    ordered_k_tuples = []
    
    for query_list in tqdm(query_lists):
        ordered_k_tuple = set()
        
        for query in query_list:
            ordered_k_tuple = ordered_k_tuple.union(query_dict[query])
      
        ordered_k_tuple = list(ordered_k_tuple)
        if k_type == "top":
            ordered_k_tuple.sort()
            ordered_k_tuples.append(ordered_k_tuple[:k])
        elif k_type == "bot":
            ordered_k_tuple.sort(reverse = True)
            ordered_k_tuples.append(ordered_k_tuple[:k])
        elif k_type == "rand":
            ordered_k_tuples.append(random.sample(all_tuples, k))

    return ordered_k_tuples

def add_ordered_k_tuples(posts, ordered_k_tuples, sep_token=SEP_TOKEN, kg = "conceptnet", knowledge_type = "top", k = 20, get_scores=False, ds = "sbic"):
    print("kg type: ", kg)
    if kg=="conceptnet":
        print('conceptnet')
        post_score_triple_mapping = {}
        for i, post in enumerate(tqdm(posts)):
            ordered_k_tuple = ordered_k_tuples[i]
            edge_string = ''
            scores = []
            for edge in ordered_k_tuple:
                start = edge[1]
                relation = REL_DICT[edge[2]]
                end = edge[3].replace('_', ' ')

                edge_string += sep_token + start + relation + end
                
                scores.append(edge[0])
                
            posts[i] += edge_string
            
            if get_scores:
                post_score_triple_mapping[i] = (post, scores, edge_string)
        
        if get_scores:
            with open(f"cosinescore_dumps/{ds}_{kg}_{knowledge_type}_k_{k}_dumps.pkl", "wb") as file:
                pickle.dump(post_score_triple_mapping, file)
            file.close()
    elif kg == "stereokg":
        print('stereokg')
        df_stereokg = pd.read_csv("../data/stereoKG.tsv", delimiter = ",")
        df_stereokg = df_stereokg.drop(columns = ["Unnamed: 0"])
        stereokg_vals = df_stereokg["linearised"].values
        post_cos_triple_mapping = {}
        for i, post in enumerate(tqdm(posts)):
            post_embeddings = get_post_embedding(post)
            stereokg_embeddings = get_stereokg_embeddings(stereokg_vals)
            cosine_scores = get_cosine_scores(post_embeddings, stereokg_embeddings)
            stereokg_k_triples = get_cosine_k_triples(k = k, cosine_scores = cosine_scores, kg_vals = stereokg_vals, typ = knowledge_type)
            # cosine_top_k_scores = cosine_scores[cosine_scores[:, 0].cpu().sort().indices.numpy()[::-1][:k].copy()]
            edge_string = ''
            for each in stereokg_k_triples:
                edge_string += sep_token + each
            posts[i]+=edge_string

            if get_scores:
                post_cos_triple_mapping[i] = (post, cosine_top_k_scores, stereokg_k_triples)

        if get_scores:
            with open(f"cosinescore_dumps/{ds}_{kg}_{knowledge_type}_k_{k}_dumps.pkl", "wb") as file:
                pickle.dump(post_cos_triple_mapping, file)
            file.close()
#         if not os.path.exists(f"cosinescore_dumps/{ds}_{kg}_{knowledge_type}_k_20_dumps.pkl"):
#             df_stereokg = pd.read_csv("../data/stereoKG.tsv", delimiter = ",")
#             df_stereokg = df_stereokg.drop(columns = ["Unnamed: 0"])
#             stereokg_vals = df_stereokg["linearised"].values
#             post_cos_triple_mapping = {}
#             for i, post in enumerate(tqdm(posts)):
#                 post_embeddings = get_post_embedding(post)
#                 stereokg_embeddings = get_stereokg_embeddings(stereokg_vals)
#                 cosine_scores = get_cosine_scores(post_embeddings, stereokg_embeddings)
#                 stereokg_k_triples = get_cosine_k_triples(k = k, cosine_scores = cosine_scores, kg_vals = stereokg_vals, typ = knowledge_type)
#                 # cosine_top_k_scores = cosine_scores[cosine_scores[:, 0].cpu().sort().indices.numpy()[::-1][:k].copy()]
#                 edge_string = ''
#                 for each in stereokg_k_triples:
#                     edge_string += sep_token + each
#                 posts[i]+=edge_string

#                 if get_scores:
#                     post_cos_triple_mapping[i] = (post, cosine_top_k_scores, stereokg_k_triples)
            
#             if get_scores:
#                 with open(f"cosinescore_dumps/{ds}_{kg}_{knowledge_type}_k_{k}_dumps.pkl", "wb") as file:
#                     pickle.dump(post_cos_triple_mapping, file)
#                 file.close()
#         else:
#             with open(f"cosinescore_dumps/{ds}_{kg}_{knowledge_type}_k_20_dumps.pkl", "rb") as file:
#                 post_cos_triple_mapping = pickle.load(file)
#             file.close()
            
#             for i, post in enumerate(tqdm(posts)):
#                 _, cosine_top_k_scores, stereokg_top_k_triples = post_cos_triple_mapping[i]
    else:
        raise ValueError("kg is supposed to be either 'conceptnet' or 'stereokg'.")
    return posts


def concat_top_k_tuples(
    df_post,
    edge_dict,
    knowledge_type,
    knowledge_graph,
    sep_token=SEP_TOKEN,
    emb_dict=None,
    k=20,
    get_scores = False,
    ds = "sbic"
):
    posts = df_post['post'].tolist()
    lemmatizer = WordNetLemmatizer()
    query_lists = []
    query_tokens = set()
    ordered_k_tuples = []
    
    if knowledge_graph == 'conceptnet':
        for post in tqdm(posts):
            add_queries(query_lists, query_tokens, post, lemmatizer)

        idf = get_idf(posts, query_tokens)
        query_dict = collect_edges(query_tokens, idf, edge_dict, emb_dict)

        all_tuples = set()
        for key in tqdm(query_dict):
            all_tuples = all_tuples.union(query_dict[key])
        all_tuples = sorted(list(all_tuples), key = lambda x: x[0])

        ordered_k_tuples = collect_ordered_k_tuples(query_lists, query_dict, all_tuples, k=k, k_type = knowledge_type)

    # print(ordered_k_tuples)
    posts = add_ordered_k_tuples(posts, ordered_k_tuples, sep_token, knowledge_graph, knowledge_type, k=k, get_scores=get_scores, ds=ds)
    
    new_df = df_post.copy()
    
    new_df['post'] = posts

    return new_df
