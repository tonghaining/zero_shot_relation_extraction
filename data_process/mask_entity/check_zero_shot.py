import json

dev = 'relation_splits/dev.0'
train = 'relation_splits/train.0'
test = 'relation_splits/test.0'

dev_relation = {}
train_relation = {}
test_relation = {}


with open(dev, 'r') as f:
    for line in f:
        relation = line.split('\t')[0]
        if relation not in dev_relation:
            dev_relation[relation] = 1
        else:
            dev_relation[relation] += 1

dev_relation

with open(train, 'r') as f:
    for line in f:
        relation = line.split('\t')[0]
        if relation not in train_relation:
            train_relation[relation] = 1
        else:
            train_relation[relation] += 1

train_relation

with open(test, 'r') as f:
    for line in f:
        relation = line.split('\t')[0]
        if relation not in test_relation:
            test_relation[relation] = 1
        else:
            test_relation[relation] += 1

test_relation
