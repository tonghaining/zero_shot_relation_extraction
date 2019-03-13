import random
import json

random.seed(0)

relation_description_file = "./relation_descriptions.txt"
relation_description_json = "./relation_descriptions.json"
test_relation_description_json = "./test_relation_descriptions.json"
dev_relation_description_json = "./dev_relation_descriptions.json"
train_relation_description_json = "./train_relation_descriptions.json"

lines = open(relation_description_file, 'r').readlines()

relation_descriptions = {}

for line in lines:
        rd = line.split("\t")
        if len(rd) == 1:
            continue
        relation, description = rd[0],rd[1]
        if relation in relation_descriptions: 
            relation_descriptions[relation].append(description)
            continue
        else:
            relation_descriptions[relation] = [ description ]

#print(relation_descriptions.keys())
#print(len(relation_descriptions))

all_label = list(relation_descriptions.keys())
random.shuffle(all_label)
test_label = all_label[0:12]
dev_label = all_label[12:24]
train_label = all_label[24:]

test_relations = {}
dev_relations = {}
train_relations = {}

for k in relation_descriptions.keys():
    if k in test_label:
        test_relations[k] = relation_descriptions[k]
    elif k in dev_label:
        dev_relations[k] = relation_descriptions[k]
    elif k in train_label:
        train_relations[k] = relation_descriptions[k]
    else:
        pass

with open(relation_description_json, 'w') as outfile:  
    json.dump(relation_descriptions, outfile)
with open(test_relation_description_json, 'w') as outfile:  
    json.dump(test_relations, outfile)
with open(dev_relation_description_json, 'w') as outfile:  
    json.dump(dev_relations, outfile)
with open(train_relation_description_json, 'w') as outfile:  
    json.dump(train_relations, outfile)



