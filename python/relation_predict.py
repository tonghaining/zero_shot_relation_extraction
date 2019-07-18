"""
Script to generate a CSV file of predictions on the test data.
"""

import tensorflow as tf
import os
import importlib
import random
from util import logger
import util.parameters as params
from util.data_processing import *
from util.evaluate import *
import pickle

FIXED_PARAMETERS = params.load_parameters()
modname = FIXED_PARAMETERS["model_name"]
logpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".log"
logger = logger.Logger(logpath)

model = FIXED_PARAMETERS["model_type"]

module = importlib.import_module(".".join(['models', model]))
MyModel = getattr(module, 'MyModel')

# Logging parameter settings at each launch of training script
# This will help ensure nothing goes awry in reloading a model and we consistenyl use the same hyperparameter settings.
logger.Log("FIXED_PARAMETERS\n %s" % FIXED_PARAMETERS)


######################### LOAD DATA #############################

logger.Log("Loading data")

test_uwre = load_uwre_data(FIXED_PARAMETERS["test_uwre"])

relation_description_path = FIXED_PARAMETERS['relation_description']
with open(relation_description_path, 'r') as file:
    relation_descriptions = json.load(file)

all_relation = list(relation_descriptions.keys())

dictpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".p"

if not os.path.isfile(dictpath):
    print("No dictionary found!")
    exit(1)

else:
    logger.Log("Loading dictionary from %s" % (dictpath))
    word_indices = pickle.load(open(dictpath, "rb"))
    logger.Log("Padding and indexifying sentences")
    uwre_sentences_to_padded_index_sequences(word_indices, [test_uwre])

loaded_embeddings = loadEmbedding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices)
padded_relation_descriptions = descriptions_to_padded_index_sequences(word_indices, relation_descriptions)

class modelClassifier:
    def __init__(self, seq_length):
        ## Define hyperparameters
        self.learning_rate =  FIXED_PARAMETERS["learning_rate"]
        self.display_epoch_freq = 1
        self.display_step_freq = 50
        self.embedding_dim = FIXED_PARAMETERS["word_embedding_dim"]
        self.dim = FIXED_PARAMETERS["hidden_embedding_dim"]
        self.batch_size = FIXED_PARAMETERS["batch_size"]
        self.emb_train = FIXED_PARAMETERS["emb_train"]
        self.keep_rate = FIXED_PARAMETERS["keep_rate"]
        self.sequence_length = FIXED_PARAMETERS["seq_length"]
        self.alpha = FIXED_PARAMETERS["alpha"]
        self.description_num = int(FIXED_PARAMETERS["description_num"])

        logger.Log("Building model from %s.py" %(model))
        self.model = MyModel(seq_length=self.sequence_length, emb_dim=self.embedding_dim,  hidden_dim=self.dim, embeddings=loaded_embeddings, emb_train=self.emb_train, description_num=self.description_num)

        # Perform gradient descent with Adam
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999).minimize(self.model.total_cost)

        # tf things: initialize variables and create placeholder for session
        logger.Log("Initializing variables")
        self.init = tf.global_variables_initializer()
        self.sess = None
        self.saver = tf.train.Saver()
        
        self.sess = tf.Session()
        self.sess.run(self.init)

    def get_padded_descriptions(self, instance, relations, description_num):
        premise_list = []
        hypothesis_list = []

        for relation in relations:
            premise_instance = instance['sentence_index_sequence']
            premise = [premise_instance] * description_num
            premise_list.append(premise)

            hypothesis_len = len(padded_relation_descriptions[relation])
            hypothesis_ind = np.random.choice(hypothesis_len, description_num, replace=False)
            hypothesis = padded_relation_descriptions[relation][hypothesis_ind]
            hypothesis_list.append(hypothesis)

        premise_vectors = np.vstack(premise_list)
        hypothesis_vectors = np.vstack(hypothesis_list)

        return premise_vectors, hypothesis_vectors

    def classify(self, instance, relation_scope = all_relation):
      
      minibatch_premise_vectors, minibatch_hypothesis_vectors = self.get_padded_descriptions(instance, relation_scope, description_num=self.description_num)
      
      feed_dict = {self.model.premise_x: minibatch_premise_vectors, 
             self.model.hypothesis_x: minibatch_hypothesis_vectors, 
             self.model.keep_rate_ph: 1.0}
      logit = self.sess.run(self.model.logits, feed_dict)

      return np.argmax(logit[:,0], axis=0)

def extract_relation(classify, eval_set, name):
    """
    Get comma-separated CSV of predictions.
    Output file has two columns: pairID, prediction
    """
    RELATION_MAP = dict(enumerate(all_relation))
        
    predictions = []
    
    for i in range(len(eval_set)):
        instance = eval_set[i]
        true_label = instance['relation']
        true_ind = all_relation.index(true_label)
        
        hypothesis = classify(instance)
        prediction = RELATION_MAP[hypothesis]
        pairID = eval_set[i]["pairID"]  
        predictions.append((pairID, hypothesis, prediction,true_ind, true_label))

        logger.Log("predict for %i -th instance:\n \t predict index: %i,\t predict relation name: %s,\n \t real index: %i,\t real relation name: %s" % (i, hypothesis, prediction,true_ind, true_label))

    with open(name + '_predictions.csv', 'w') as f:
        w = csv.writer(f, delimiter=',')
        w.writerow(['pairID', 'predict_index', 'predict_relation', 'real_index', 'real_relation'])
        for example in predictions:
            w.writerow(example)

classifier = modelClassifier(FIXED_PARAMETERS["seq_length"])

"""
Get CSVs of predictions.
"""

logger.Log("Creating CSV of predicitons on matched test set: %s" %(modname+"_predictions.csv"))
extract_relation(classifier.classify, test_uwre, modname)

