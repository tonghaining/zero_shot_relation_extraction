import json
import os
import random
from stanfordcorenlp import StanfordCoreNLP
import logging
random.seed(0)


log = './log'
logging.basicConfig(filename=log,level=logging.DEBUG,format='%(asctime)s %(message)s', datefmt='%d/%m/%Y%H:%M:%S')
relations = '../relation_description/extended_relation_descriptions.json'


nlp = StanfordCoreNLP(r'/home/znr113/master_thesis/data/stanford-corenlp-full-2018-10-05')
# nlp = StanfordCoreNLP(r'/home/haining/Documents/master thesis/program/stanford/stanford-corenlp-full-2018-10-05')


NUM_OF_LINES=5000

def ner_random_pick(sentence, subject):
    lst = nlp.ner(sentence)
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

    if container_word != '': # take care of the last one
        container_list.append(container_word)

    if subject in container_list: # remove subject from the candidate
        container_list.remove(subject)

    if len(container_list) == 0: # if no candidate, choose the longest word as object
        to_return = max(sentence.replace(subject,'').split(), key=len)
    else:
        to_return = random.choice(container_list)

    # logging.info("chosen nn:"+str(to_return))
    return to_return


with open(relations) as f:
    relation_descriptions = json.load(f)

def mask_file(in_data, out_data):
    circle = 0
    counter = 0
    json_container = []
    sentence_container = ""

    with open(in_data, encoding='utf-8') as fin:
        for line in fin:
            content = line.split('\t')
            length = len(content)
            relation = content[0]
            if relation in relation_descriptions.keys(): # valid relations
                if length == 5: # positive
                    counter += 1
                    current_subject = content[2]
                    current_object = content[4].replace('\n','')
                    current_sentence = content[3]
                    current_sentence = current_sentence.replace(current_subject, 'SUBJECT_ENTITY').replace(current_object, 'OBJECT_ENTITY')
                    json_container.append({
                    'gold_label': 'entailment',
                    'relation': relation,
                    'sentence': current_sentence,
                    'pairID': current_subject + str(counter)
                    })
                elif length == 4: # negative
                    counter += 1
                    current_subject = content[2]
                    current_sentence = content[3]
                    random_obj = ner_random_pick(current_sentence, current_subject)
                    current_sentence = current_sentence.replace(current_subject, 'SUBJECT_ENTITY').replace(random_obj, 'OBJECT_ENTITY')
                    json_container.append({
                    'gold_label': 'neutral',
                    'relation': relation,
                    'sentence': current_sentence,
                    'pairID': current_subject + str(counter)
                    })
                else:
                    continue
                # logging.info('circle:' + str(circle) +' counter:' + str(counter))
            else:
                print('UNKNOWN RELATION:', str(relation))
                # logging.info("OPS-----------UNKNOWN RELATION:---"+str(relation))
                continue
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

for fold in range(10):
    dev_data = './relation_splits/dev.' + str(fold)
    mask_dev = './masked_splits/dev.' + str(fold)

    train_data = './relation_splits/train.' + str(fold)
    mask_train = './masked_splits/train.' + str(fold)

    test_data = './relation_splits/test.' + str(fold)
    mask_test = './masked_splits/test.' + str(fold)
    
    print(fold)
    logging.info('on fold:' + str(fold))
    
    print('masking dev...')
    logging.info('masking dev...')
    mask_file(dev_data, mask_dev)
    
    print('masking train...')
    logging.info('masking train...')
    mask_file(train_data, mask_train)
    
    print('masking test...')
    logging.info('masking test...')
    mask_file(test_data, mask_test)
