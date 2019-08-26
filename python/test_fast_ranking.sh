#!/bin/bash
# normal cpu stuff: allocate cpus, memory
# we run on the gpu partition and we allocate 1 titanx gpu
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH -p gpu --gres=gpu:1
#We expect that our program should not run langer than 2 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=4:00:00

#your script, in this case: write the hostname and the ids of the chosen gpus.
echo $CUDA_VISIBLE_DEVICES
PYTHONPATH=$PYTHONPATH:. python relation_predict.py connect_esim connect_3_DeNum_5 --description_num 5 --crossfold 3 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python relation_predict.py inference_esim inference_3_DeNum_5 --description_num 5 --crossfold 3 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python relation_predict.py pooling_esim pooling_3_DeNum_5 --description_num 5 --crossfold 3 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python relation_predict.py mlp_esim mlp_3_DeNum_5 --description_num 5 --crossfold 3 --genre travel --test

