import tensorflow as tf
import os
import importlib
import random
import json
from util import logger
import util.parameters as params
from util.data_processing import *
from util.evaluate import *
from sklearn.metrics import f1_score
from itertools import repeat


FIXED_PARAMETERS = params.load_parameters()
modname = FIXED_PARAMETERS["model_name"]
logpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".log"
logger = logger.Logger(logpath)

model = FIXED_PARAMETERS["model_type"]

module = importlib.import_module(".".join(['models', model]))
MyModel = getattr(module, 'MyModel')

relation_description_path = FIXED_PARAMETERS['relation_description']

# Logging parameter settings at each launch of training script
# This will help ensure nothing goes awry in reloading a model and we consistenyl use the same hyperparameter settings.
logger.Log("FIXED_PARAMETERS\n %s" % FIXED_PARAMETERS)
logger.Log("Loading data")

training_uwre = load_uwre_data(FIXED_PARAMETERS["training_uwre"])
dev_uwre = load_uwre_data(FIXED_PARAMETERS["dev_uwre"])
test_uwre = load_uwre_data(FIXED_PARAMETERS["test_uwre"])

with open(relation_description_path, 'r') as file:
    relation_descriptions = json.load(file)

if 'temp.jsonl' in FIXED_PARAMETERS["test_matched"]:
    # Removing temporary empty file that was created in parameters.py
    os.remove(FIXED_PARAMETERS["test_matched"])
    logger.Log("Created and removed empty file called temp.jsonl since test set is not available.")

dictpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".p"

if not os.path.isfile(dictpath):
    logger.Log("Building dictionary")
    word_indices = build_uwre_dictionary([training_uwre], relation_descriptions)
    logger.Log("Padding and indexifying sentences")
    uwre_sentences_to_padded_index_sequences(word_indices, [training_uwre, dev_uwre, test_uwre])
    padded_relation_descriptions = descriptions_to_padded_index_sequences(word_indices, relation_descriptions)
    pickle.dump(word_indices, open(dictpath, "wb"))

else:
    logger.Log("Loading dictionary from %s" % (dictpath))
    word_indices = pickle.load(open(dictpath, "rb"))
    logger.Log("Padding and indexifying sentences")
    uwre_sentences_to_padded_index_sequences(word_indices, [training_uwre, dev_uwre, test_uwre])
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
        self.model = MyModel(seq_length=self.sequence_length, emb_dim=self.embedding_dim,  hidden_dim=self.dim, embeddings=loaded_embeddings, emb_train=self.emb_train, description_num=self.description_num)

        # Perform gradient descent with Adam
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999).minimize(self.model.total_cost)

        # Boolean stating that training has not been completed,
        self.completed = False

        # tf things: initialize variables and create placeholder for session
        logger.Log("Initializing variables")
        self.init = tf.global_variables_initializer()
        self.sess = None
        self.saver = tf.train.Saver()

    def get_minibatch(self, dataset, start_index, end_index):
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


    def train(self, train_uwre, dev_uwre, test_uwre):
        self.sess = tf.Session()
        self.sess.run(self.init)

        self.step = 1
        self.epoch = 0
        self.best_dev_uwre = 0.
        self.best_strain_acc = 0.
        self.last_train_acc = [.001, .001, .001, .001, .001]
        self.best_step = 0

        # Restore most recent checkpoint if it exists.
        # Also restore values for best dev-set accuracy and best training-set accuracy.
        ckpt_file = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt"
        if os.path.isfile(ckpt_file + ".meta"):
            if os.path.isfile(ckpt_file + "_best.meta"):
                self.saver.restore(self.sess, (ckpt_file + "_best"))
                test_f1 = evaluate_f1(self.classify, test_uwre, self.batch_size)
                test_acc, test_cost = evaluate_classifier(self.classify, test_uwre, self.batch_size)
                dev_f1 = evaluate_f1(self.classify, dev_uwre, self.batch_size)
                self.best_dev_uwre, dev_cost_uwre = evaluate_classifier(self.classify, dev_uwre, self.batch_size)
                self.best_strain_acc, strain_cost = evaluate_classifier(self.classify, train_uwre[0:5000], self.batch_size)
                logger.Log("Restored best UWRE test f1: %f\n Restored best UWRE test acc: %f\n Restored best UWRE-dev f1: %f\n Restored best UWRE-dev acc: %f\n Restored best UWRE train acc: %f" %(test_f1, test_acc, dev_f1, self.best_dev_uwre,  self.best_strain_acc))

            self.saver.restore(self.sess, ckpt_file)
            logger.Log("Model restored from file: %s" % ckpt_file)

        training_data = train_uwre

        ### Training cycle
        logger.Log("Training...")

        while True:
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
                    dev_f1 = evaluate_f1(self.classify, dev_uwre, self.batch_size)
                    dev_acc_uwre, dev_cost_uwre = evaluate_classifier(self.classify, dev_uwre, self.batch_size)
                    strain_acc, strain_cost = evaluate_classifier(self.classify, train_uwre[0:5000], self.batch_size)

                    logger.Log("Step: %i\t Dev f1-score: %f" %(self.step, dev_f1))
                    logger.Log("Step: %i\t Dev-UWRE acc: %f\t UWRE train acc: %f" %(self.step, dev_acc_uwre, strain_acc))
                    logger.Log("Step: %i\t Dev-UWRE cost: %f\t UWRE train cost: %f" %(self.step, dev_cost_uwre, strain_cost))

                if self.step % 500 == 0:
                    self.saver.save(self.sess, ckpt_file)
                    best_test = 100 * (1 - self.best_dev_uwre / dev_acc_uwre)
                    if best_test > 0.04:
                        self.saver.save(self.sess, ckpt_file + "_best")
                        self.best_dev_uwre = dev_acc_uwre
                        self.best_strain_acc = strain_acc
                        self.best_step = self.step
                        logger.Log("Checkpointing with new best UWRE-dev accuracy: %f" %(self.best_dev_uwre))

                self.step += 1

                # Compute average loss
                avg_cost += c / (total_batch * self.batch_size)

            # Display some statistics about the epoch
            if self.epoch % self.display_epoch_freq == 0:
                logger.Log("Epoch: %i\t Avg. Cost: %f" %(self.epoch+1, avg_cost))

            self.epoch += 1
            self.last_train_acc[(self.epoch % 5) - 1] = strain_acc

            # Early stopping
            progress = 1000 * (sum(self.last_train_acc)/(5 * min(self.last_train_acc)) - 1)

            if (progress < 0.1) or (self.step > self.best_step + 10000):# pre 30000
                logger.Log("Best uwre-dev accuracy: %s" %(self.best_dev_uwre))
                logger.Log("MultiNLI Train accuracy: %s" %(self.best_strain_acc))
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

    def classify(self, examples):
        # This classifies a list of examples
        total_batch = int(len(examples) / self.batch_size)
        logits = np.empty(3)
        genres = []
        for i in range(total_batch):
            minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres = self.get_minibatch(
                examples, self.batch_size * i, self.batch_size * (i + 1))
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

# While test-set isn't released, use dev-sets for testing
test_matched = test_uwre
test_mismatched = dev_uwre


if test == False:
    classifier.train(training_uwre, dev_uwre, test_uwre)
    logger.Log("Acc on UWRE dev: %s" %(evaluate_classifier(classifier.classify, test_matched, FIXED_PARAMETERS["batch_size"]))[0])
    logger.Log("F1-score on UWRE dev: %s" %(evaluate_f1(classifier.classify, test_mismatched, FIXED_PARAMETERS["batch_size"])))
    logger.Log("Acc on UWRE test: %s" %(evaluate_classifier(classifier.classify, test_matched, FIXED_PARAMETERS["batch_size"]))[0])
    logger.Log("F1-score on UWRE test: %s" %(evaluate_f1(classifier.classify, test_mismatched, FIXED_PARAMETERS["batch_size"])))
else:
    results = evaluate_uwre_final(classifier.restore, classifier.classify, [test_matched, test_mismatched], FIXED_PARAMETERS["batch_size"])
    logger.Log("Acc on UWRE test: %s" %(results[0]))
    logger.Log("F1-score on UWRE test: %s" %(results[1]))
