#!/bin/bash
# normal cpu stuff: allocate cpus, memory
# we run on the gpu partition and we allocate 1 titanx gpu
#SBATCH -p gpu --gres=gpu:titanx:a00818
#We expect that our program should not run langer than 2 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=2:00:00

#your script, in this case: write the hostname and the ids of the chosen gpus.
echo $CUDA_VISIBLE_DEVICES
PYTHONPATH=$PYTHONPATH:. python relation_predict.py ccim connect_3_DeNum_5 --description_num 5 --crossfold 3 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python relation_predict.py icim inference_3_DeNum_5 --description_num 5 --crossfold 3 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python relation_predict.py pcim pooling_3_DeNum_5 --description_num 5 --crossfold 3 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python relation_predict.py mcim mlp_3_DeNum_5 --description_num 5 --crossfold 3 --genre travel --test

