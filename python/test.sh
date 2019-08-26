#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=20000M
# we run on the gpu partition and we allocate 1 titanx gpu
#SBATCH -p gpu --gres=gpu:titanx:a00818
#We expect that our program should not run langer than 2 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=10:00:00

#your script, in this case: write the hostname and the ids of the chosen gpus.
echo $CUDA_VISIBLE_DEVICES

PYTHONPATH=$PYTHONPATH:. python test_multi.py connect_esim connect_7_DeNum_5 --description_num 5 --crossfold 7 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py connect_esim connect_7_DeNum_10 --description_num 10 --crossfold 7 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py connect_esim connect_7_DeNum_15 --description_num 15 --crossfold 7 --genre travel --test

PYTHONPATH=$PYTHONPATH:. python test_multi.py connect_esim connect_8_DeNum_5 --description_num 5 --crossfold 8 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py connect_esim connect_8_DeNum_10 --description_num 10 --crossfold 8 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py connect_esim connect_8_DeNum_15 --description_num 15 --crossfold 8 --genre travel --test

PYTHONPATH=$PYTHONPATH:. python test_multi.py connect_esim connect_9_DeNum_5 --description_num 5 --crossfold 9 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py connect_esim connect_9_DeNum_10 --description_num 10 --crossfold 9 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py connect_esim connect_9_DeNum_15 --description_num 15 --crossfold 9 --genre travel --test


PYTHONPATH=$PYTHONPATH:. python test_multi.py inference_esim inference_7_DeNum_5 --description_num 5 --crossfold 7 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py inference_esim inference_7_DeNum_10 --description_num 10 --crossfold 7 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py inference_esim inference_7_DeNum_15 --description_num 15 --crossfold 7 --genre travel --test

PYTHONPATH=$PYTHONPATH:. python test_multi.py inference_esim inference_8_DeNum_5 --description_num 5 --crossfold 8 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py inference_esim inference_8_DeNum_10 --description_num 10 --crossfold 8 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py inference_esim inference_8_DeNum_15 --description_num 15 --crossfold 8 --genre travel --test

PYTHONPATH=$PYTHONPATH:. python test_multi.py inference_esim inference_9_DeNum_5 --description_num 5 --crossfold 9 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py inference_esim inference_9_DeNum_10 --description_num 10 --crossfold 9 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py inference_esim inference_9_DeNum_15 --description_num 15 --crossfold 9 --genre travel --test

PYTHONPATH=$PYTHONPATH:. python test_multi.py pooling_esim pooling_7_DeNum_5 --description_num 5 --crossfold 7 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py pooling_esim pooling_7_DeNum_10 --description_num 10 --crossfold 7 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py pooling_esim pooling_7_DeNum_15 --description_num 15 --crossfold 7 --genre travel --test

PYTHONPATH=$PYTHONPATH:. python test_multi.py pooling_esim pooling_8_DeNum_5 --description_num 5 --crossfold 8 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py pooling_esim pooling_8_DeNum_10 --description_num 10 --crossfol  8 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py pooling_esim pooling_8_DeNum_15 --description_num 15 --crossfold 8 --genre travel --test

PYTHONPATH=$PYTHONPATH:. python test_multi.py pooling_esim pooling_9_DeNum_5 --description_num 5 --crossfold 9 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py pooling_esim pooling_9_DeNum_10 --description_num 10 --crossfold 9 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py pooling_esim pooling_9_DeNum_15 --description_num 15 --crossfold 9 --genre travel --test

PYTHONPATH=$PYTHONPATH:. python test_multi.py mlp_esim mlp_7_DeNum_5 --description_num 5 --crossfold 7 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py mlp_esim mlp_7_DeNum_10 --description_num 10 --crossfold 7 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py mlp_esim mlp_7_DeNum_15 --description_num 15 --crossfold 7 --genre travel --test

PYTHONPATH=$PYTHONPATH:. python test_multi.py mlp_esim mlp_8_DeNum_5 --description_num 5 --crossfold 8 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py mlp_esim mlp_8_DeNum_10 --description_num 10 --crossfold 8 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py mlp_esim mlp_8_DeNum_15 --description_num 15 --crossfold 8 --genre travel --test

PYTHONPATH=$PYTHONPATH:. python test_multi.py mlp_esim mlp_9_DeNum_5 --description_num 5 --crossfold 9 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py mlp_esim mlp_9_DeNum_10 --description_num 10 --crossfold 9 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py mlp_esim mlp_9_DeNum_15 --description_num 15 --crossfold 9 --genre travel --test







