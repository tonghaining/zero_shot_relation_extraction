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
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] == 'neutral':
                
            counter += 1 
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
    dev_data = './mask_entity/dev.' + str(fold)
    mask_dev = './entailment_contadiction/dev.' + str(fold)

    train_data = './mask_entity/train.' + str(fold)
    mask_train = './entailment_contadiction/train.' + str(fold)

    test_data = './mask_entity/test.' + str(fold)
    mask_test = './entailment_contadiction/test.' + str(fold)
    
    logging.info('on fold:' + str(fold))
    
    logging.info('converting dev...')
    mask_file(dev_data, mask_dev)
    
    logging.info('converting train...')
    mask_file(train_data, mask_train)
    
    logging.info('converting test...')
    mask_file(test_data, mask_test)
