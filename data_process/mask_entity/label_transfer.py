import json
import os
import logging


NUM_OF_LINES=5000

def mask_file(in_data, out_data):
    circle = 0
    counter = 0
    json_container = []
    sentence_container = ""

    with open(in_data, encoding='utf-8') as fin:
        for line in fin:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] == 'neutral':
                loaded_example['gold_label'] = 'NOT_ENTAIL'
            elif loaded_example['gold_label'] == 'entailment':
                loaded_example['gold_label'] = 'ENTAIL'
            else:
                continue
            counter += 1
            json_container.append(loaded_example) 
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
    dev_data = './masked_splits/dev.' + str(fold)
    mask_dev = './transfer_label/dev.' + str(fold)

    train_data = './masked_splits/train.' + str(fold)
    mask_train = './transfer_label/train.' + str(fold)

    test_data = './masked_splits/test.' + str(fold)
    mask_test = './transfer_label/test.' + str(fold)
    
    logging.info('on fold:' + str(fold))
    
    logging.info('converting dev...')
    mask_file(dev_data, mask_dev)
    
    logging.info('converting train...')
    mask_file(train_data, mask_train)
    
    logging.info('converting test...')
    mask_file(test_data, mask_test)
