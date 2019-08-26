#!/bin/bash
# normal cpu stuff: allocate cpus, memory
# we run on the gpu partition and we allocate 1 titanx gpu
#SBATCH -p gpu --gres=gpu:titanx:a00701
#We expect that our program should not run langer than 2 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=1:00:00

#your script, in this case: write the hostname and the ids of the chosen gpus.
echo $CUDA_VISIBLE_DEVICES
PYTHONPATH=$PYTHONPATH:. python test_multi.py inference_esim inference_5_DeNum_5 --description_num 5 --crossfold 5 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py inference_esim inference_5_DeNum_10 --description_num 10 --crossfold 5 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py inference_esim inference_5_DeNum_15 --description_num 15 --crossfold 5 --genre travel --test

PYTHONPATH=$PYTHONPATH:. python test_multi.py inference_esim inference_6_DeNum_5 --description_num 5 --crossfold 6 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py inference_esim inference_6_DeNum_10 --description_num 10 --crossfold 6 --genre travel --test
PYTHONPATH=$PYTHONPATH:. python test_multi.py inference_esim inference_6_DeNum_15 --description_num 15 --crossfold 6 --genre travel --test



