import tensorflow as tf
import os
import importlib
import random
from util import logger
import util.parameters as params
from util.data_processing import *
from util.evaluate import *

FIXED_PARAMETERS = params.load_parameters()
modname = FIXED_PARAMETERS["model_name"]
logpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".log"
logger = logger.Logger(logpath)

model = FIXED_PARAMETERS["model_type"]

module = importlib.import_module(".".join(['models', model])) 
MyModel = getattr(module, 'MyModel')

# Logging parameter settings at each launch of training script
# This will help ensure nothing goes awry in reloading a model and we consistently use the same hyperparameter settings. 
logger.Log("FIXED_PARAMETERS\n %s" % FIXED_PARAMETERS)


######################### LOAD DATA #############################

logger.Log("Loading data")

training_mnli = load_nli_data(FIXED_PARAMETERS["training_mnli"])
dev_matched = load_nli_data(FIXED_PARAMETERS["dev_matched"])
dev_mismatched = load_nli_data(FIXED_PARAMETERS["dev_mismatched"])
test_matched = load_nli_data(FIXED_PARAMETERS["test_matched"])
test_mismatched = load_nli_data(FIXED_PARAMETERS["test_mismatched"])

training_uwre = load_uwre_data(FIXED_PARAMETERS["training_uwre"])
dev_uwre = load_uwre_data(FIXED_PARAMETERS["dev_uwre"])

relation_description_path = FIXED_PARAMETERS['relation_description']
with open(relation_description_path, 'r') as file:
    relation_descriptions = json.load(file)

if 'temp.jsonl' in FIXED_PARAMETERS["test_matched"]:
    # Removing temporary empty file that was created in parameters.py
    os.remove(FIXED_PARAMETERS["test_matched"])
    logger.Log("Created and removed empty file called temp.jsonl since test set is not available.")

dictpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".p"

if not os.path.isfile(dictpath): 
    logger.Log("Building dictionary")
    word_indices = build_dictionary([training_mnli], [training_uwre], relation_descriptions)
    
    logger.Log("Padding and indexifying sentences")
    sentences_to_padded_index_sequences(word_indices, [training_mnli, 
                                        dev_matched, dev_mismatched, 
                                        test_matched, test_mismatched])
    uwre_sentences_to_padded_index_sequences(word_indices, [dev_uwre])
    padded_relation_descriptions = descriptions_to_padded_index_sequences(word_indices, relation_descriptions)
    pickle.dump(word_indices, open(dictpath, "wb"))

else:
    logger.Log("Loading dictionary from %s" % (dictpath))
    word_indices = pickle.load(open(dictpath, "rb"))
    logger.Log("Padding and indexifying sentences")
    sentences_to_padded_index_sequences(word_indices, [training_mnli, 
                                        dev_matched, dev_mismatched, 
                                        test_matched, test_mismatched])
    uwre_sentences_to_padded_index_sequences(word_indices, [dev_uwre])
    padded_relation_descriptions = descriptions_to_padded_index_sequences(word_indices, relation_descriptions)

logger.Log("Loading embeddings")
loaded_embeddings = loadEmbedding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices)


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
        self.model = MyModel(seq_length=self.sequence_length, emb_dim=self.embedding_dim, 
                                hidden_dim=self.dim, embeddings=loaded_embeddings, 
                                emb_train=self.emb_train, description_num=self.description_num)

        # Perform gradient descent with Adam
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999).minimize(self.model.total_cost)

        # Boolean stating that training has not been completed, 
        self.completed = False 

        # tf things: initialize variables and create placeholder for session
        logger.Log("Initializing variables")
        self.init = tf.global_variables_initializer()
        self.sess = None
        self.saver = tf.train.Saver()
    
    def get_uwre_minibatch(self, dataset, start_index, end_index):
        indices = range(start_index, end_index)
        premise_list = []
        hypothesis_list = []

        for i in indices:
            relation = dataset[i]['relation']

            premise_instance = dataset[i]['sentence_index_sequence']
            premise = [premise_instance] * self.description_num
            premise_list.append(premise)

            hypothesis_len = len(padded_relation_descriptions[relation])
            hypothesis_ind = np.random.choice(hypothesis_len, self.description_num, replace=False)
            hypothesis = padded_relation_descriptions[relation][hypothesis_ind]
            hypothesis_list.append(hypothesis)

        premise_vectors = np.vstack(premise_list)
        hypothesis_vectors = np.vstack(hypothesis_list)
        genres = [dataset[i]['genre'] for i in indices]
        labels = [dataset[i]['label'] for i in indices]

        return premise_vectors, hypothesis_vectors, labels, genres

    def get_minibatch(self, dataset, start_index, end_index):
        indices = range(start_index, end_index)
        premise_vectors = np.vstack([[dataset[i]['sentence1_binary_parse_index_sequence']] * self.description_num for i in indices])
        hypothesis_vectors = np.vstack([[dataset[i]['sentence2_binary_parse_index_sequence']] * self.description_num for i in indices])
        genres = [dataset[i]['genre'] for i in indices]
        labels = [dataset[i]['label'] for i in indices]
        return premise_vectors, hypothesis_vectors, labels, genres


    def train(self, train_mnli, dev_mat, dev_mismat, dev_uw):        
        self.sess = tf.Session()
        self.sess.run(self.init)

        self.step = 0
        self.epoch = 0
        self.best_dev_mat = 0.
        self.best_mtrain_acc = 0.
        self.last_train_acc = [.001, .001, .001, .001, .001]
        self.best_step = 0

        # Restore most recent checkpoint if it exists. 
        # Also restore values for best dev-set accuracy and best training-set accuracy
        ckpt_file = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt"
        if os.path.isfile(ckpt_file + ".meta"):
            if os.path.isfile(ckpt_file + "_best.meta"):
                self.saver.restore(self.sess, (ckpt_file + "_best"))
                self.best_dev_mat, dev_cost_mat = evaluate_classifier(self.classify, dev_mat, self.batch_size)
                best_dev_mismat, dev_cost_mismat = evaluate_classifier(self.classify, dev_mismat, self.batch_size)
                self.best_mtrain_acc, mtrain_cost = evaluate_classifier(self.classify, train_mnli[0:5000], self.batch_size)
                logger.Log("Restored best matched-dev acc: %f\n Restored best mismatched-dev acc: %f\n \
                            Restored best MulitNLI train acc: %f" %(self.best_dev_mat, best_dev_mismat, self.best_mtrain_acc))
                

            self.saver.restore(self.sess, ckpt_file)
            logger.Log("Model restored from file: %s" % ckpt_file)

        # Combine MultiNLI and SNLI data. Alpha has a default value of 0, if we want to use SNLI data, it must be passed as an argument.

        ### Training cycle
        logger.Log("Training...")

        while True:
            training_data = train_mnli
            random.shuffle(training_data)
            avg_cost = 0.
            total_batch = int(len(training_data) / self.batch_size)

            # Loop over all batches in epoch
            for i in range(total_batch):
                # Assemble a minibatch of the next B examples
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres = self.get_minibatch(
                    training_data, self.batch_size * i, self.batch_size * (i + 1))
                
                # Run the optimizer to take a gradient step, and also fetch the value of the 
                # cost function for logging
                feed_dict = {self.model.premise_x: minibatch_premise_vectors,
                                self.model.hypothesis_x: minibatch_hypothesis_vectors,
                                self.model.y: minibatch_labels, 
                                self.model.keep_rate_ph: self.keep_rate}
                _, c = self.sess.run([self.optimizer, self.model.total_cost], feed_dict)

                if self.step % self.display_step_freq == 0:
                    dev_acc_mat, dev_cost_mat = evaluate_classifier(self.classify, dev_mat, self.batch_size)
                    dev_acc_mismat, dev_cost_mismat = evaluate_classifier(self.classify, dev_mismat, self.batch_size)

                    dev_acc_uwre, dev_cost_uwre = evaluate_classifier(self.uwre_classify, dev_uw, self.batch_size)

                    mtrain_acc, mtrain_cost = evaluate_classifier(self.classify, train_mnli[0:5000], self.batch_size)

                    logger.Log("Step: %i\t uwre dev acc: %f\t uwre dev cost %f\t" \
                            %(self.step, dev_acc_uwre, dev_cost_uwre))
                    logger.Log("Step: %i\t Dev-matched acc: %f\t Dev-mismatched acc: %f\t \
                            MultiNLI train acc: %f" %(self.step, dev_acc_mat, 
                                dev_acc_mismat, mtrain_acc))
                    logger.Log("Step: %i\t Dev-matched cost: %f\t Dev-mismatched cost: %f\t \
                            MultiNLI train cost: %f" %(self.step, dev_cost_mat, 
                                dev_cost_mismat, mtrain_cost))

                if self.step % 500 == 0:
                    self.saver.save(self.sess, ckpt_file)
                    best_test = 100 * (1 - self.best_dev_mat / dev_acc_mat)
                    if best_test > 0.04:
                        self.saver.save(self.sess, ckpt_file + "_best")
                        self.best_dev_mat = dev_acc_mat
                        self.best_mtrain_acc = mtrain_acc
                        self.best_step = self.step
                        logger.Log("Checkpointing with new best matched-dev accuracy: %f" %(self.best_dev_mat))

                self.step += 1

                # Compute average loss
                avg_cost += c / (total_batch * self.batch_size)
                                
            # Display some statistics about the epoch
            if self.epoch % self.display_epoch_freq == 0:
                logger.Log("Epoch: %i\t Avg. Cost: %f" %(self.epoch+1, avg_cost))
            
            self.epoch += 1 
            self.last_train_acc[(self.epoch % 5) - 1] = mtrain_acc

            # Early stopping
            progress = 1000 * (sum(self.last_train_acc)/(5 * min(self.last_train_acc)) - 1) 

            if (progress < 0.1) or (self.step > self.best_step + 30000):
                logger.Log("Best matched-dev accuracy: %s" %(self.best_dev_mat))
                logger.Log("MultiNLI Train accuracy: %s" %(self.best_mtrain_acc))
                self.completed = True
                break


    def restore(self, best=True):
        if True:
            path = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best"
        else:
            path = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt"
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.saver.restore(self.sess, path)
        logger.Log("Model restored from file: %s" % path)
   
    def uwre_classify(self, examples):
        # This classifies a list of examples
        total_batch = int(len(examples) / self.batch_size)
        logits = np.empty(2)
        genres = []
        for i in range(total_batch):
            minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres = self.get_uwre_minibatch(examples, self.batch_size * i, self.batch_size * (i + 1))
            feed_dict = {self.model.premise_x: minibatch_premise_vectors,
                                self.model.hypothesis_x: minibatch_hypothesis_vectors,
                                self.model.y: minibatch_labels,
                                self.model.keep_rate_ph: 1.0}
            genres += minibatch_genres
            logit, cost = self.sess.run([self.model.logits, self.model.total_cost], feed_dict)
            logits = np.vstack([logits, logit])
        return genres, np.argmax(logits[1:], axis=1), cost

    def classify(self, examples):
        # This classifies a list of examples
        total_batch = int(len(examples) / self.batch_size)
        logits = np.empty(2)
        genres = []
        for i in range(total_batch):
            minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres = self.get_minibatch(examples, 
                                    self.batch_size * i, self.batch_size * (i + 1))
            feed_dict = {self.model.premise_x: minibatch_premise_vectors, 
                                self.model.hypothesis_x: minibatch_hypothesis_vectors,
                                self.model.y: minibatch_labels, 
                                self.model.keep_rate_ph: 1.0}
            genres += minibatch_genres
            logit, cost = self.sess.run([self.model.logits, self.model.total_cost], feed_dict)
            logits = np.vstack([logits, logit])

        return genres, np.argmax(logits[1:], axis=1), cost



classifier = modelClassifier(FIXED_PARAMETERS["seq_length"])

"""
Either train the model and then run it on the test-sets or 
load the best checkpoint and get accuracy on the test set. Default setting is to train the model.
"""

test = params.train_or_test()

print("ALL RESULTS ON TEST")

if test == False:
    classifier.train(training_mnli, dev_matched, dev_mismatched, dev_uwre)
    logger.Log("Acc on matched multiNLI dev-set: %s" 
        % (evaluate_classifier(classifier.classify, test_matched, FIXED_PARAMETERS["batch_size"]))[0])
    logger.Log("Acc on mismatched multiNLI dev-set: %s" 
        % (evaluate_classifier(classifier.classify, test_mismatched, FIXED_PARAMETERS["batch_size"]))[0])
else: 
    results, bylength = evaluate_final(classifier.restore, classifier.classify, 
        [test_matched, test_mismatched, ], FIXED_PARAMETERS["batch_size"])
    logger.Log("Acc on multiNLI matched dev-set: %s" %(results[0]))
    logger.Log("Acc on multiNLI mismatched dev-set: %s" %(results[1]))

