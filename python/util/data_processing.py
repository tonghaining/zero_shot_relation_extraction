import numpy as np
import re
import random
import json
import collections
from . import parameters as params
import pickle

FIXED_PARAMETERS = params.load_parameters()

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 1,
#    "contradiction": 2,
    "hidden": 0
}

PADDING = "<PAD>"
UNKNOWN = "<UNK>"

def load_uwre_data(path):
    """
    Load UWRE data.
    "uwre" is set to "genre". 
    """
    data = []
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            loaded_example["genre"] = "uwre"
            data.append(loaded_example)
        random.seed(1)
        random.shuffle(data)
    return data

def tokenize(string):
    string = re.sub(r'\(|\)', '', string)
    return string.split()

def build_uwre_dictionary(training_datasets, relation_descriptions):
    """
    Extract vocabulary and build dictionary.
    """
    word_counter = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        for example in dataset:
            word_counter.update(tokenize(example['sentence']))
    
    for relation in relation_descriptions:
        for description in relation_descriptions[relation]:
            word_counter.update(tokenize(description))
    
    vocabulary = set([word for word in word_counter])
    vocabulary = list(vocabulary)
    vocabulary = [PADDING, UNKNOWN] + vocabulary

    word_indices = dict(zip(vocabulary, range(len(vocabulary))))

    return word_indices

def uwre_sentences_to_padded_index_sequences(word_indices, datasets):
    """
    Annotate datasets with feature vectors. Adding right-sided padding.
    """
    for i, dataset in enumerate(datasets):
        for example in dataset:
            example['sentence' + '_index_sequence'] = np.zeros((FIXED_PARAMETERS["seq_length"]), dtype=np.int32)

            token_sequence = tokenize(example['sentence'])
            padding = FIXED_PARAMETERS["seq_length"] - len(token_sequence)

            for i in range(FIXED_PARAMETERS["seq_length"]):
                if i >= len(token_sequence):
                    index = word_indices[PADDING]
                else:
                    if token_sequence[i] in word_indices:
                        index = word_indices[token_sequence[i]]
                    else:
                        index = word_indices[UNKNOWN]
                example['sentence' + '_index_sequence'][i] = index

def descriptions_to_padded_index_sequences(word_indices, relation_descriptions):
    padded_relation_descriptions = {}

    for relation in relation_descriptions:
        descriptions = np.zeros((len(relation_descriptions[relation]), FIXED_PARAMETERS["seq_length"]), dtype=np.int32)
        for j,description in enumerate(relation_descriptions[relation]):
            token_sequence = tokenize(description)
            padding = FIXED_PARAMETERS["seq_length"] - len(token_sequence)

            for i in range(FIXED_PARAMETERS["seq_length"]):
                if i >= len(token_sequence):
                    index = word_indices[PADDING]
                else:
                    if token_sequence[i] in word_indices:
                        index = word_indices[token_sequence[i]]
                    else:
                        index = word_indices[unknown]
                descriptions[j][i] = index
        padded_relation_descriptions[relation] = descriptions
    return padded_relation_descriptions


def loadEmbedding_zeros(path, word_indices):
    """
    Load GloVe embeddings. Initializng OOV words to vector of zeros.
    """
    emb = np.zeros((len(word_indices), FIXED_PARAMETERS["word_embedding_dim"]), dtype='float32')
    
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if FIXED_PARAMETERS["embeddings_to_load"] != None:
                if i >= FIXED_PARAMETERS["embeddings_to_load"]:
                    break
            
            s = line.split()
            if s[0] in word_indices: # s[0] is the word
                if len(s) > 301: # there is a exception for "." in glove.text
                    tail = s[len(s) - 300:]
                    head = [s[0]]
                    s = head + tail
                    # print(head)
                emb[word_indices[s[0]], :] = np.asarray(s[1:])

    return emb


def loadEmbedding_rand(path, word_indices):
    """
    Load GloVe embeddings. Doing a random normal initialization for OOV words.
    """
    n = len(word_indices)
    m = FIXED_PARAMETERS["word_embedding_dim"]
    emb = np.empty((n, m), dtype=np.float32)

    emb[:,:] = np.random.normal(size=(n,m))

    # Explicitly assign embedding of <PAD> to be zeros.
    emb[0:2, :] = np.zeros((1,m), dtype="float32")
    
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if FIXED_PARAMETERS["embeddings_to_load"] != None:
                if i >= FIXED_PARAMETERS["embeddings_to_load"]:
                    break
            
            s = line.split()
            if s[0] in word_indices:
                if len(s) > 301:
                    tail = s[len(s)-300:]                
                    head = [s[0]]
                    s = head + tail
                    # print(head)
                emb[word_indices[s[0]], :] = np.asarray(s[1:]) 
    return emb

