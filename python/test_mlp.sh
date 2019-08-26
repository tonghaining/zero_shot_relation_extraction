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
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_5 --description_num 5 --crossfold 0 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_5 --description_num 5 --crossfold 1 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_5 --description_num 5 --crossfold 2 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_5 --description_num 5 --crossfold 3 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_5 --description_num 5 --crossfold 4 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_5 --description_num 5 --crossfold 5 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_5 --description_num 5 --crossfold 6 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_5 --description_num 5 --crossfold 7 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_5 --description_num 5 --crossfold 8 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_5 --description_num 5 --crossfold 9 --genre travel --test

PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_10 --description_num 10 --crossfold 0 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_10 --description_num 10 --crossfold 1 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_10 --description_num 10 --crossfold 2 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_10 --description_num 10 --crossfold 3 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_10 --description_num 10 --crossfold 4 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_10 --description_num 10 --crossfold 5 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_10 --description_num 10 --crossfold 6 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_10 --description_num 10 --crossfold 7 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_10 --description_num 10 --crossfold 8 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_10 --description_num 10 --crossfold 9 --genre travel --test

PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_15 --description_num 15 --crossfold 0 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_15 --description_num 15 --crossfold 1 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_15 --description_num 15 --crossfold 2 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_15 --description_num 15 --crossfold 3 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_15 --description_num 15 --crossfold 4 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_15 --description_num 15 --crossfold 5 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_15 --description_num 15 --crossfold 6 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_15 --description_num 15 --crossfold 7 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_15 --description_num 15 --crossfold 8 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python train_multi.py mlp_esim only_pretrained_mlp_DeNum_15 --description_num 15 --crossfold 9 --genre travel --test

