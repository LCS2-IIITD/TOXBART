from multiprocessing.sharedctypes import Value
import pickle
import numpy as np
import pandas as pd
import os, random
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from ast import literal_eval
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

STEREOKG_EMB_FILE_NAME = "../data/stereo-kg-embeddings.pickle"
STEREO_EDGE_DICT_FILE_NAME = "../data/stereokg-edge-dict.pickle"
MODEL_NAME = "all-MiniLM-L6-v2"

# Utils for StereoKG
def get_post_embedding(post):
    model = SentenceTransformer(MODEL_NAME)
    model.cuda()
    post_embeddings = model.encode(post, convert_to_tensor=True)
    return post_embeddings

def get_stereokg_embeddings(kg_vals):
    if not os.path.exists(f"{STEREOKG_EMB_FILE_NAME}"):
        model = SentenceTransformer(MODEL_NAME)
        model.cuda()
        kg_embeddings = model.encode(kg_vals, convert_to_tensor=True)
        with open(f'{STEREOKG_EMB_FILE_NAME}', 'wb') as file:
            pickle.dump(kg_embeddings, file)
        file.close()
    else:
        with open(f'{STEREOKG_EMB_FILE_NAME}', 'rb') as file:
            kg_embeddings = pickle.load(file)
        file.close()
    return kg_embeddings

def get_cosine_scores(post_embeddings, kg_embeddings):
    cosine_scores = util.cos_sim(kg_embeddings, post_embeddings)
    return cosine_scores

def get_cosine_k_triples(k, cosine_scores, kg_vals, typ):
    if typ == "top":
        return kg_vals[cosine_scores[:, 0].cpu().sort().indices.numpy()[::-1][:k]].tolist()
    elif typ == "bot":
        return kg_vals[cosine_scores[:, 0].cpu().sort().indices.numpy()[:k]].tolist()
    else:
        return kg_vals[random.sample(cosine_scores[:, 0].cpu().sort().indices.numpy().tolist(), k)].tolist()

def get_stereokg_edge_dict():
    edges = {}
    if not os.path.exists(f"{STEREO_EDGE_DICT_FILE_NAME}"):
        df = pd.read_csv("../data/stereoKG.tsv", delimiter = "\t")
        for i in range(df.shape[0]):
            start = df.iloc[i, 0]
            rel = df.iloc[i, 1]
            end = df.iloc[i, 2]
            if start not in edges:
                edges[start] = []
            edges[start].append([rel, end])
        with open(f'{STEREO_EDGE_DICT_FILE_NAME}', 'wb') as edge_file:
            pickle.dump(edges, edge_file)
        edge_file.close()
    else:
        with open(f'{STEREO_EDGE_DICT_FILE_NAME}', 'rb') as edge_file:
            edges = pickle.load(edge_file)
        edge_file.close()
    return edges
