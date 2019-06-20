#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=20000M
# we run on the gpu partition and we allocate 1 titanx gpu
#SBATCH -p gpu --gres=gpu:titanx:a00818
#We expect that our program should not run langer than 2 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=20:00:00

#your script, in this case: write the hostname and the ids of the chosen gpus.
echo $CUDA_VISIBLE_DEVICES
PYTHONPATH=$PYTHONPATH:. python train_multi.py max_esim attempt_max_1 --genre travel --emb_train --test
