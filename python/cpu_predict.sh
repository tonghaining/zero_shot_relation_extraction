#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=10 --cpus-per-task=1 --mem=40000M
# we run on the gpu partition and we allocate 1 titanx gpu
# not use !SBATCH -p gpu --gres=gpu:titanx:1
#We expect that our program should not run langer than 2 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=1:00:00

#your script, in this case: write the hostname and the ids of the chosen gpus.
#image1
echo $CUDA_VISIBLE_DEVICES
PYTHONPATH=$PYTHONPATH:. python relation_predict.py hidden_esim pretrained_hidden_0_DeNum_1 --crossfold 0 --description_num 1 --genre travel --emb_train
