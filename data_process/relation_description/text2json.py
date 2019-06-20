import random
import json

random.seed(0)

relation_description_file = "./relation_descriptions.txt"
relation_description_json = "./relation_descriptions.json"
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

print(relation_descriptions.keys())
print(len(relation_descriptions))

with open(relation_description_json, 'w') as outfile:  
    json.dump(relation_descriptions, outfile)

