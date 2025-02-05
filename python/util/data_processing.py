import numpy as np
import re
import random
import json
import collections
from . import parameters as params
import pickle

FIXED_PARAMETERS = params.load_parameters()

NLI_LABEL_MAP = {
    "entailment": 0,
    "contradiction": 1,
}
UWRE_LABEL_MAP = {
    "ENTAIL": 0,
    "NOT_ENTAIL": 1,
}

PADDING = "<PAD>"
UNKNOWN = "<UNK>"


def tokenize(string):
    string = re.sub(r'\(|\)', '', string)
    return string.split()

# multiNLI data functions, for pretrain

def load_nli_data(path, snli=False):
    """
    Load MultiNLI or SNLI data.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. 
    """
    data = []
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in NLI_LABEL_MAP:
                continue
            loaded_example["label"] = NLI_LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            data.append(loaded_example)
        random.seed(1)
        random.shuffle(data)
    return data

def load_nli_data_genre(path, genre, snli=True):
    """
    Load a specific genre's examples from MultiNLI, or load SNLI data and assign a "snli" genre to the examples.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. If set to true, it will overwrite the genre label for MultiNLI data.
    """
    data = []
    j = 0
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in NLI_LABEL_MAP:
                continue
            loaded_example["label"] = NLI_LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            if loaded_example["genre"] == genre:
                data.append(loaded_example)
        random.seed(1)
        random.shuffle(data)
    return data

def build_nli_dictionary(training_datasets):
    """
    Extract vocabulary and build dictionary.
    """  
    word_counter = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        for example in dataset:
            word_counter.update(tokenize(example['sentence1_binary_parse']))
            word_counter.update(tokenize(example['sentence2_binary_parse']))
        
    vocabulary = set([word for word in word_counter])
    vocabulary = list(vocabulary)
    vocabulary = [PADDING, UNKNOWN] + vocabulary
        
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))

    return word_indices

def build_dictionary(multi_nli_datasets, uwre_training_datasets, relation_descriptions):
    """
    Extract vocabulary and build dictionary.
    """
    word_counter = collections.Counter()
    for i, dataset in enumerate(multi_nli_datasets):
        for example in dataset:
            word_counter.update(tokenize(example['sentence1_binary_parse']))
            word_counter.update(tokenize(example['sentence2_binary_parse']))
            
    for i, dataset in enumerate(uwre_training_datasets):
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

def sentences_to_padded_index_sequences(word_indices, datasets):
    """
    Annotate datasets with feature vectors. Adding right-sided padding. 
    """
    for i, dataset in enumerate(datasets):
        for example in dataset:
            for sentence in ['sentence1_binary_parse', 'sentence2_binary_parse']:
                example[sentence + '_index_sequence'] = np.zeros((FIXED_PARAMETERS["seq_length"]), dtype=np.int32)

                token_sequence = tokenize(example[sentence])
                padding = FIXED_PARAMETERS["seq_length"] - len(token_sequence)

                for i in range(FIXED_PARAMETERS["seq_length"]):
                    if i >= len(token_sequence):
                        index = word_indices[PADDING]
                    else:
                        if token_sequence[i] in word_indices:
                            index = word_indices[token_sequence[i]]
                        else:
                            index = word_indices[UNKNOWN]
                    example[sentence + '_index_sequence'][i] = index

# UW-RE related functions

def load_uwre_data(path):
    """
    Load UWRE data.
    "uwre" is set to "genre". 
    """
    data = []
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in UWRE_LABEL_MAP:
                continue
            loaded_example["label"] = UWRE_LABEL_MAP[loaded_example["gold_label"]]
            loaded_example["genre"] = "uwre"
            data.append(loaded_example)
        random.seed(1)
        random.shuffle(data)
    return data

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
                        index = word_indices[UNKNOWN]
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

