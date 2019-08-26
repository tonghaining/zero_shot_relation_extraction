import json
import os
import random
import logging
random.seed(0)


log = './log'
logging.basicConfig(filename=log,level=logging.DEBUG,format='%(asctime)s %(message)s', datefmt='%d/%m/%Y%H:%M:%S')
relations = '../relation_description/extended_relation_descriptions.json'


NUM_OF_LINES=5000

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
                    json_container.append({
                    'gold_label': 'neutral',
                    'relation': relation,
                    'sentence': current_sentence,
                    'pairID': current_subject + str(counter)
                    })
                else:
                    continue
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
    mask_dev = './json_uwre/dev.' + str(fold)

    train_data = './relation_splits/train.' + str(fold)
    mask_train = './json_uwre/train.' + str(fold)

    test_data = './relation_splits/test.' + str(fold)
    mask_test = './json_uwre/test.' + str(fold)
    
    logging.info('on fold:' + str(fold))
    
    logging.info('converting dev...')
    mask_file(dev_data, mask_dev)
    
    logging.info('converting train...')
    mask_file(train_data, mask_train)
    
    logging.info('converting test...')
    mask_file(test_data, mask_test)
