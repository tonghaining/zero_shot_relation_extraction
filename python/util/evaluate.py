import csv
import sys
import numpy as np
from sklearn.metrics import f1_score

def evaluate_classifier(classifier, eval_set, batch_size):
    """
    Function to get accuracy and cost of the model, evaluated on a chosen dataset.

    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    correct = 0
    genres, hypotheses, cost = classifier(eval_set)
    cost = cost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size
    for i in range(full_batch):
        hypothesis = hypotheses[i]
        if hypothesis == eval_set[i]['label']:
            correct += 1        
    return correct / float(len(eval_set)), cost

def evaluate_classifier_genre(classifier, eval_set, batch_size):
    """
    Function to get accuracy and cost of the model by genre, evaluated on a chosen dataset. It returns a dictionary of accuracies by genre and cost for the full evaluation dataset.
    
    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    genres, hypotheses, cost = classifier(eval_set)
    correct = dict((genre,0) for genre in set(genres))
    count = dict((genre,0) for genre in set(genres))
    cost = cost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size

    for i in range(full_batch):
        hypothesis = hypotheses[i]
        genre = genres[i]
        if hypothesis == eval_set[i]['label']:
            correct[genre] += 1.
        count[genre] += 1.

        if genre != eval_set[i]['genre']:
            print('welp!')

    accuracy = {k: correct[k]/count[k] for k in correct}

    return accuracy, cost

def evaluate_classifier_bylength(classifier, eval_set, batch_size):
    """
    Function to get accuracy and cost of the model by genre, evaluated on a chosen dataset. It returns a dictionary of accuracies by genre and cost for the full evaluation dataset.
    
    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    genres, hypotheses, cost = classifier(eval_set)
    correct = dict((genre,0) for genre in set(genres))
    count = dict((genre,0) for genre in set(genres))
    cost = cost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size

    for i in range(full_batch):
        hypothesis = hypotheses[i]
        genre = genres[i]
        if hypothesis == eval_set[i]['label']:
            correct[genre] += 1.
        count[genre] += 1.

        if genre != eval_set[i]['genre']:
            print('welp!')

    accuracy = {k: correct[k]/count[k] for k in correct}

    return accuracy, cost

def evaluate_f1(classifier, eval_set, batch_size):
    """
    Function to get f1 score of the model, evaluated on a chosen dataset.
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
    f1 = f1_score(y_true, y_pred, labels=[0,1], average='macro')
    return f1

def evaluate_uwre_final(restore, classifier, eval_sets, batch_size):
    """
    Function to get percentage accuracy and f1 score of the model, evaluated on a set of chosen datasets.
    
    restore: a function to restore a stored checkpoint
    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    restore(best=True)
    percentages = []
    length_results = []
    for eval_set in eval_sets:
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
            
            if hypothesis == eval_set[i]['label']:
                correct += 1  
        percentages.append(correct / float(len(eval_set)))  
        f1 = f1_score(y_true, y_predict, labels=[0,1], average='macro')
    return percentages, f1

def evaluate_final(restore, classifier, eval_sets, batch_size, name):
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

    predictions = []

    for eval_set in eval_sets:
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
            
            if hypothesis == eval_set[i]['label']:
                correct += 1  

            prediction = (eval_set[i]["pairID"], eval_set[i]["relation"], int(y_true[i]), int(hypothesis))
            predictions.append(prediction)

        percentages.append(correct / float(len(eval_set)))  
        f1 = f1_score(y_true, y_predict, labels=[0,1], average='macro')

    with open(name + '_predictions.csv', 'w') as f:
        w = csv.writer(f, delimiter=',')
        w.writerow(['pairID', 'relation', 'true_label', 'predict_label'])
        for example in predictions:
            w.writerow(example)

    return percentages, f1
