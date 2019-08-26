#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=10 --cpus-per-task=1 --mem=10000M
# we run on the gpu partition and we allocate 1 titanx gpu
# not use !SBATCH -p gpu --gres=gpu:titanx:1
#We expect that our program should not run langer than 2 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=15:00:00

#your script, in this case: write the hostname and the ids of the chosen gpus.
#image1
echo $CUDA_VISIBLE_DEVICES
#PYTHONPATH=$PYTHONPATH:. python relation_predict.py pooling_esim pooling_6_DeNum_10 --crossfold 6 --description_num 10 --genre travel --test
#PYTHONPATH=$PYTHONPATH:. python relation_predict.py mlp_esim mlp_6_DeNum_10 --crossfold 6 --description_num 10 --genre travel --test
#PYTHONPATH=$PYTHONPATH:. python relation_predict.py inference_esim inference_6_DeNum_10 --crossfold 6 --description_num 10 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python relation_predict.py connect_esim connect_6_DeNum_10 --crossfold 6 --description_num 10 --genre travel --test
