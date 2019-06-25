#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=40000M
# we run on the gpu partition and we allocate 1 titanx gpu
#SBATCH -p gpu --gres=gpu:titanx:a00756
#We expect that our program should not run langer than 2 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=30:00:00

#your script, in this case: write the hostname and the ids of the chosen gpus.
echo $CUDA_VISIBLE_DEVICES
PYTHONPATH=$PYTHONPATH:. python train_multi.py mean_esim attempt_5_DeNum_15 --crossfold 5 --description_num 15 --genre travel --emb_train 
