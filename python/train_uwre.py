"""
Training script to train a model on only SNLI data. MultiNLI data is loaded into the embeddings enabling us to test the model on MultiNLI data.
"""

import tensorflow as tf
import os
import importlib
import random
from util import logger
import util.parameters as params
from util.data_processing import *
from util.evaluate import *
from sklearn.metrics import f1_score

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

#################### OVER WRITEN FUNCTIONS ######################

def load_uwre_data(path):
    """
    Load MultiNLI or SNLI data.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. 
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

def build_uwre_dictionary(training_datasets):
    """
    Extract vocabulary and build dictionary.
    """  
    word_counter = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        for example in dataset:
            word_counter.update(tokenize(example['sentence1']))
            word_counter.update(tokenize(example['sentence2']))
        
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
            for sentence in ['sentence1', 'sentence2']:
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

def evaluate_f1(classifier, eval_set, batch_size):
    """
    Function to get accuracy and cost of the model, evaluated on a chosen dataset.
    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    genres, hypotheses, _cost = classifier(eval_set)
    full_batch = int(len(eval_set) / batch_size) * batch_size
    y_true = np.zeros((len(eval_set)))
    y_pred = np.zeros((len(eval_set)))# len(eval_set) != len(hypotheses), one is 446, the other 447  ??!
    for i in range(full_batch):
        y_true[i] = eval_set[i]['label']
        y_pred[i] = hypotheses[i]
    f1 = f1_score(y_true, y_pred, labels=[0,1,2], average='macro')
    return f1

def evaluate_uwre_final(restore, classifier, eval_sets, batch_size):
    """
    Function to get percentage accuracy of the model, evaluated on a set of chosen datasets.
    
    restore: a function to restore a stored checkpoint
    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    restore(best=True)
    percentages = []
    length_results = []
    for eval_set in eval_sets:
        bylength_prem = {}
        bylength_hyp = {}
        genres, hypotheses, cost = classifier(eval_set)
        correct = 0
        cost = cost / batch_size
        full_batch = int(len(eval_set) / batch_size) * batch_size
        y_true = np.zeros((full_batch))
        y_predict = np.zeros((full_batch))

        for i in range(full_batch):
            hypothesis = hypotheses[i]
            y_predict[i] = hypothesis
            y_true[i] = eval_set[i]['label']
            
            length_1 = len(eval_set[i]['sentence1'].split())
            length_2 = len(eval_set[i]['sentence2'].split())
            if length_1 not in bylength_prem.keys():
                bylength_prem[length_1] = [0,0]
            if length_2 not in bylength_hyp.keys():
                bylength_hyp[length_2] = [0,0]

            bylength_prem[length_1][1] += 1
            bylength_hyp[length_2][1] += 1

            if hypothesis == eval_set[i]['label']:
                correct += 1  
                bylength_prem[length_1][0] += 1
                bylength_hyp[length_2][0] += 1    
        percentages.append(correct / float(len(eval_set)))  
        length_results.append((bylength_prem, bylength_hyp))
        f1 = f1_score(y_true, y_predict, labels=[0,1,2], average='macro')
    return percentages, f1

######################### LOAD DATA #############################

logger.Log("Loading data")
training_uwre = load_uwre_data(FIXED_PARAMETERS["training_uwre"])
dev_uwre = load_uwre_data(FIXED_PARAMETERS["dev_uwre"])
test_uwre = load_uwre_data(FIXED_PARAMETERS["test_uwre"])

if 'temp.jsonl' in FIXED_PARAMETERS["test_matched"]:
    # Removing temporary empty file that was created in parameters.py
    os.remove(FIXED_PARAMETERS["test_matched"])
    logger.Log("Created and removed empty file called temp.jsonl since test set is not available.")

dictpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".p"

if not os.path.isfile(dictpath): 
    logger.Log("Building dictionary")
    word_indices = build_uwre_dictionary([training_uwre])
    logger.Log("Padding and indexifying sentences")
    uwre_sentences_to_padded_index_sequences(word_indices, [training_uwre, dev_uwre, test_uwre])
    pickle.dump(word_indices, open(dictpath, "wb"))

else:
    logger.Log("Loading dictionary from %s" % (dictpath))
    word_indices = pickle.load(open(dictpath, "rb"))
    logger.Log("Padding and indexifying sentences")
    uwre_sentences_to_padded_index_sequences(word_indices, [training_uwre, dev_uwre, test_uwre])

logger.Log("Loading embeddings")
loaded_embeddings = loadEmbedding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices)

class modelClassifier:
    def __init__(self, seq_length):
        ## Define hyperparameters
        self.learning_rate =  FIXED_PARAMETERS["learning_rate"]
        self.display_epoch_freq = 1
        self.display_step_freq = 2
        self.embedding_dim = FIXED_PARAMETERS["word_embedding_dim"]
        self.dim = FIXED_PARAMETERS["hidden_embedding_dim"]
        self.batch_size = FIXED_PARAMETERS["batch_size"]
        self.emb_train = FIXED_PARAMETERS["emb_train"]
        self.keep_rate = FIXED_PARAMETERS["keep_rate"]
        self.sequence_length = FIXED_PARAMETERS["seq_length"] 
        self.alpha = FIXED_PARAMETERS["alpha"]

        logger.Log("Building model from %s.py" %(model))
        self.model = MyModel(seq_length=self.sequence_length, emb_dim=self.embedding_dim,  hidden_dim=self.dim, embeddings=loaded_embeddings, emb_train=self.emb_train)

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
        #print(dataset[0])
        premise_vectors = np.vstack([dataset[i]['sentence1_index_sequence'] for i in indices])
        hypothesis_vectors = np.vstack([dataset[i]['sentence2_index_sequence'] for i in indices])
        genres = [dataset[i]['genre'] for i in indices]
        labels = [dataset[i]['label'] for i in indices]
        return premise_vectors, hypothesis_vectors, labels, genres


    def train(self, train_uwre, dev_uwre, test_uwre):
        # def train(self, train_uwre, dev_mat, dev_mismat, dev_uwre):
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

                # Since a single epoch can take a  ages for larger models (ESIM),
                #  we'll print accuracy every 50 steps
                if self.step % self.display_step_freq == 0:
                    test_f1 = evaluate_f1(self.classify, test_uwre, self.batch_size)
                    test_acc, test_cost = evaluate_classifier(self.classify, test_uwre, self.batch_size)
                    dev_f1 = evaluate_f1(self.classify, dev_uwre, self.batch_size)
                    dev_acc_uwre, dev_cost_uwre = evaluate_classifier(self.classify, dev_uwre, self.batch_size)
                    strain_acc, strain_cost = evaluate_classifier(self.classify, train_uwre[0:5000], self.batch_size)

                    logger.Log("Step: %i\t Test f1-score: %f\t Dev f1-score: %f" %(self.step, test_f1, dev_f1))
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

            if (progress < 0.1) or (self.step > self.best_step + 30000):
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
    logger.Log("F1-score on UWRE dev: %s" %(evaluate_f1(classifier.classify, test_mismatched, FIXED_PARAMETERS["batch_size"]))[0])
    logger.Log("Acc on UWRE test: %s" %(evaluate_classifier(classifier.classify, test_matched, FIXED_PARAMETERS["batch_size"]))[0])
    logger.Log("F1-score on UWRE test: %s" %(evaluate_f1(classifier.classify, test_mismatched, FIXED_PARAMETERS["batch_size"]))[0])
else: 
    results = evaluate_uwre_final(classifier.restore, classifier.classify, [test_matched, test_mismatched], FIXED_PARAMETERS["batch_size"])
    logger.Log("Acc on UWRE test: %s" %(results[0]))
    logger.Log("F1-score on UWRE test: %s" %(results[1]))
