import json
import os
import random
import subprocess
random.seed(0)

Fold = 0

#shuffle_bash = './fold' + str(Fold) +'/shuffle.sh'
#shuffle_res = subprocess.run(["bash", shuffle_bash])
#print(shuffle_res)

in_data = './uwre.json'
relations = './relation_descriptions.json'
out_train_data = './fold'+ str(Fold) + '/train_fold' + str(Fold) +'.json'
out_dev_data = './fold'+ str(Fold) + '/dev_fold' + str(Fold) +'.json'
out_test_data = './fold'+ str(Fold) + '/test_fold' + str(Fold) +'.json'


with open(relations) as f:
    relation_descriptions = json.load(f)

relations = list(relation_descriptions.keys())
random.shuffle(relations)
duplicate_relation = relations + relations

Flag1 = Fold*12
Flag2 = (Fold*12 + 24)
Flag3 = (Fold*12 + 36)
Flag4 = (Fold*12 + 120)

test_relations = duplicate_relation[Flag1 : Flag2]
dev_relations = duplicate_relation[Flag2 : Flag3]
train_relations = duplicate_relation[Flag3 : Flag4]

# stratified sampling rate: test/dev/train = 5/1/100
#
# train = 100.000 * 84
# dev = 1.000 * 12
# test = 5.000 * 24

train_data = []
dev_data = []
test_data = []
NUM_TO_CACHE = 100000
i=0
with open(in_data) as f:
    for line in f:
        print(i)
        i += 1
        loaded_example = json.loads(line)
        loaded_relation = loaded_example['relation']
        if loaded_relation in train_relations:
            train_data.append(loaded_example)
        elif loaded_relation in test_relations:
            test_data.append(loaded_example)
        elif loaded_relation in dev_relations:
            dev_data.append(loaded_example)
        else:
            continue
        if i % NUM_TO_CACHE == 0:
            with open(out_test_data, 'a') as outfile:
                for d in test_data:
                    json.dump(d, outfile, ensure_ascii=False)
                    outfile.write("\n")
            with open(out_dev_data, 'a') as outfile:
                for d in dev_data:
                    json.dump(d, outfile, ensure_ascii=False)
                    outfile.write("\n")
            with open(out_train_data, 'a') as outfile:
                for d in train_data:
                    json.dump(d, outfile, ensure_ascii=False)
                    outfile.write("\n")
            train_data = []
            test_data = []
            dev_data = []
if len(train_data) != 0 or len(test_data) != 0 or len(dev_data) != 0:
    with open(out_test_data, 'a') as outfile:
       for d in test_data:
           json.dump(d, outfile, ensure_ascii=False)
           outfile.write("\n")
    with open(out_dev_data, 'a') as outfile:
       for d in dev_data:
           json.dump(d, outfile, ensure_ascii=False)
           outfile.write("\n")
    with open(out_train_data, 'a') as outfile:
       for d in train_data:
           json.dump(d, outfile, ensure_ascii=False)
           outfile.write("\n")

#train_data = train_data[:80000000]
#dev_data = dev_data[:800000]
#test_data = test_data[:4000000]

#random.shuffle(test_data)
#with open(out_test_data, 'w') as outfile:  
#    for d in test_data:
#        json.dump(d, outfile, ensure_ascii=False)
#        outfile.write("\n")
#test_data = []
#
#random.shuffle(dev_data)
#with open(out_dev_data, 'w') as outfile:  
#    for d in dev_data:
#        json.dump(d, outfile, ensure_ascii=False)
#        outfile.write("\n")
#dev_data = []
#
#random.shuffle(train_data)
#with open(out_train_data, 'w') as outfile:  
#    for d in train_data:
#        json.dump(d, outfile, ensure_ascii=False)
#        outfile.write("\n")
