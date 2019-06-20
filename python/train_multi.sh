#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=40000M
# we run on the gpu partition and we allocate 1 titanx gpu
#SBATCH -p gpu --gres=gpu:titanx:a00637
#We expect that our program should not run langer than 2 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=40:00:00

#your script, in this case: write the hostname and the ids of the chosen gpus.
echo $CUDA_VISIBLE_DEVICES
PYTHONPATH=$PYTHONPATH:. python train_multi.py mean_esim attempt_mean_DeNum_15_batch32 --description_num 15 --keep_rate 0.9 --alpha 0.15 --genre travel --emb_train 
