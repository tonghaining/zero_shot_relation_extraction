import json
import os
import random
from stanfordcorenlp import StanfordCoreNLP
random.seed(0)

in_data = './dev'
dev_relations = '../relation_description/dev_relation_descriptions.json'
out_data = './dev.json'

nlp = StanfordCoreNLP(r'/home/haining/Documents/master thesis/program/stanford/stanford-corenlp-full-2018-10-05')

NUM_OF_LINES=5000

def ner_random_pick(sentence):
    lst = nlp.ner(sentence)
    #print('ner result:',lst)
    container_list = []
    container_word = ''
    last_ter = 'O'
    for tup in lst:
        current_ter = tup[1]
        if current_ter == 'O':
            if container_word == '':
                last_ter = 'O'
            else:
                container_list.append(container_word)
                container_word = ''
                last_ter = 'O'
        else:
            if current_ter == last_ter:
                container_word = container_word + ' ' + tup[0]
            else:
                if container_word == '':
                    container_word = tup[0]
                else:
                    container_list.append(container_word)
                    container_word = tup[0]
                last_ter = current_ter
    if container_word != '':
        container_list.append(container_word)
    #print('result list:', container_list)
    if len(container_list) == 0:
        to_return = max(sentence.split(), key=len)
    else:
        to_return = random.choice(container_list)
    print("chosen nn:", to_return)
    return to_return

json_container = []
sentence_container = ""
circle = 0
counter = 1

with open(dev_relations) as f:
    relation_descriptions = json.load(f)
print(relation_descriptions.keys())

with open(in_data, encoding='utf-8') as fin:
    for line in fin:
        content = line.split('\t')
        length = len(content)
        relation = content[0]
        if relation in relation_descriptions.keys(): # dev relations
            if length == 5: # positive
                counter += 1
                current_subject = content[2]
                current_object = content[4].replace('\n','')
                current_sentence = content[3]
                current_description = random.choice(relation_descriptions[relation])
                current_sentence = current_sentence.replace(current_subject, 'SUBJECT_ENTITY').replace(current_object, 'OBJECT_ENTITY')
                json_container.append({
                    'gold_label': 'entailment',
                    'relation': relation,
                    'sentence1': current_sentence,
                    'sentence2': current_description,
                    'pairID': str(circle) + current_subject[0:2] + current_object[0:2] + str(counter)
                    })
            elif length == 4: # negative
                counter += 1
                current_subject = content[2]
                current_sentence = content[3]
                current_description = random.choice(relation_descriptions[relation])
                random_obj = ner_random_pick(current_sentence) 
                current_sentence = current_sentence.replace(current_subject, 'SUBJECT_ENTITY').replace(random_obj, 'OBJECT_ENTITY')
                json_container.append({
                    'gold_label': 'neutral',
                    'relation': relation,
                    'sentence1': current_sentence,
                    'sentence2': current_description,
                    'pairID': str(circle) + current_subject[0:2] + random.choice(current_sentence) + str(counter)
                    })
            else:
                pass
            print('circle:',circle,' counter:', counter)
        else:
            pass
        if counter % NUM_OF_LINES == 0:
            circle += 1
            with open(out_data, 'a') as outfile:  
                    for d in json_container:
                        json.dump(d, outfile, ensure_ascii=False)
                        outfile.write("\n")
            json_container = []
        else:
            pass
if len(json_container) != 0:
    with open(out_data, 'a') as outfile:  
            for d in json_container:
                json.dump(d, outfile, ensure_ascii=False)
                outfile.write("\n")
    json_container = {}
