# Master Thesis work of Haining Tong

This is the code for my master thesis "Zero-Shot Relation Extraction via Description Learning". This repository is based on the ESIM implementation of [Baseline Models for MultiNLI Corpus](https://github.com/nyu-mll/multiNLI).

## Data
The [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/) and [UW-RE](http://nlp.cs.washington.edu/zeroshot/) data are utilized in my approach. The UW-RE data should be processed with ['entity_masker.py'](https://github.com/tonghaining/zero_shot_relation_extraction/blob/master/data_process/mask_entity/entity_masker.py) for masking out the entities before feeding into the models.

## Models
I proposed four neural network models based on the Conditional Inference Model (CIM). They could take multiple descriptions to pair with instance sentences. The information would be merged in different layers in the four models.

- Conditional Inference Model (CIM) : I convert the ESIM to CIM by replace the BiLSTM with cBiLSTM. Main code for this model is in ['cim.py'](https://github.com/tonghaining/zero_shot_relation_extraction/blob/master/python/models/cim.py)
- Connected-Merging CIM (CCIM) : in this model, the knowledge of descriptions would be allocated by connecting the cell states of BiLSTM in the Inference Composition layer. Main code for this model is in ['mcim.py'](https://github.com/tonghaining/zero_shot_relation_extraction/blob/master/python/models/mcim.py)
- Inference-Merging CIM (ICIM) : in this model, the merging work takes place in the Local inference Layer. Main code for this model is in ['icim.py'](https://github.com/tonghaining/zero_shot_relation_extraction/blob/master/python/models/icim.py)
- Pooling-Merging CIM (PCIM) : in this model, descriptions are merged in the Pooling Layer. Main code for this model is in ['pcim.py'](https://github.com/tonghaining/zero_shot_relation_extraction/blob/master/python/models/pcim.py)
- MLP-Merging CIM (MCIM) : in this model, descriptions are merged by averaging in the MLP Layer. Main code for this model is in ['mcim.py'](https://github.com/tonghaining/zero_shot_relation_extraction/blob/master/python/models/mcim.py)

## Training and Testing

### Training settings

The models can be pretrained with MultiNLI or train with UW-RE. Each setting has its own training script.

- To train a model only on MultiNLI data, I create both python code and jupyter notebook.
	- [`pre_train_mnli.py`](https://github.com/tonghaining/zero_shot_relation_extraction/blob/master/python/pre_train_mnli.py). 
	- ['Pre_Train.ipynb'](https://github.com/tonghaining/zero_shot_relation_extraction/blob/master/python/Pre_Train.ipynb)

- To train a model only on UW-RE data.
	- [`train_mnli.py`](https://github.com/tonghaining/zero_shot_relation_extraction/blob/master/python/train_mnli.py)

- To train the baseline model CIM with single description utilized.
	- [`train_single.py`](https://github.com/tonghaining/zero_shot_relation_extraction/blob/master/python/train_single.py)

### Command line flags

To start training with any of the training scripts, there are a couple of required command-line flags and an array of optional flags. The code concerning all flags can be found in [`parameters.py`](https://github.com/tonghaining/zero_shot_relation_extraction/blob/master/python/util/parameters.py). All the parameters set in `parameters.py` are printed to the log file everytime the training script is launched.

Required flags,

- `model_type`: there are three model types in this repository, `cim`, `ccim`,`icim`, `pcim`, and `mcim`. You must state which model you want to use.
- `model_name`: this is your experiment name. This name will be used the prefix the log and checkpoint files.

Optional flags,

- `description_num`: how many descriptions do you want to use for each prediction.
- `crossfold`: fold setting for cross validationi.
- `relation_description` : path to your directory with relation description
- `datapath`: path to your directory with MultiNLI, and UWRE data. Default is set to "../data"
- `ckptpath`: path to your directory where you wish to store checkpoint files. Default is set to "../logs"
- `logpath`: path to your directory where you wish to store log files. Default is set to "../logs"
- `emb_to_load`: path to your directory with GloVe data. Default is set to "../data"
- `learning_rate`: the learning rate you wish to use during training. Default value is set to 0.0004
- `keep_rate`: the hyper-parameter for dropout-rate. `keep_rate` = 1 - dropout-rate. The default value is set to 0.9.
- `seq_length`: the maximum sequence length you wish to use. Default value is set to 50. Sentences shorter than `seq_length` are padded to the right. Sentences longer than `seq-length` are truncated.
- `emb_train`: boolean flag that determines if the model updates word embeddings during training. If called, the word embeddings are updated.
- `alpha`: only used during `train_mnli` scheme. Determines what percentage of SNLI training data to use in each epoch of training. Default value set to 0.0 (which makes the model train on MultiNLI only).
- `genre`: only used during `train_genre` scheme. Use this flag to set which single genre you wish to train on. Valid genres are `travel`, `fiction`, `slate`, `telephone`, `government`, or `snli`.
- `test`: boolean used to test a trained model. Call this flag if you wish to load a trained model and test it on MultiNLI dev-sets* and SNLI test-set. When called, the best checkpoint will be used (see section on checkpoints for more details).


### Other parameters

Remaining parameters like the size of hidden layers, word embeddings, and minibatch can be changed directly in `parameters.py`. The default hidden embedding and word embedding size is set to 300, the minibatch size (`batch_size` in the code) is set to 16.

### Sample commands
You can run the following bash directly, or following the python command inside them,

- ['pre_train.sh'](https://github.com/tonghaining/zero_shot_relation_extraction/blob/master/python/pre_train.sh)
- ['train_single.sh'](https://github.com/tonghaining/zero_shot_relation_extraction/blob/master/python/train_single.sh)
- ['train_multi.sh'](https://github.com/tonghaining/zero_shot_relation_extraction/blob/master/python/train_multi.sh)

### Testing models
To test a trained model, simply add the `test` flag to the command used for training. The best checkpoint will be loaded and used to evaluate the model's performance on the UW-RE.

- ['test.sh'](https://github.com/tonghaining/zero_shot_relation_extraction/blob/master/python/test.sh) would run ['test_multi.py'](https://github.com/tonghaining/zero_shot_relation_extraction/blob/master/python/test_multi.py). It would also generate a CSV file with the true label and predicted label.
- ['test_ranking.sh'](https://github.com/tonghaining/zero_shot_relation_extraction/blob/master/python/test_ranking.sh) would run ['relation_preict.py'](https://github.com/tonghaining/zero_shot_relation_extraction/blob/master/python/relation_preict.py). It would selecting the relation with highest entailment score among all the 120 relations.

### Checkpoints

I maintain two checkpoints: the most recent checkpoint and the best checkpoint. Every 500 steps, the most recent checkpoint is updated, and we test to see if the dev-set accuracy has improved by at least 0.04%. If the accuracy has gone up by at least 0.04%, then the best checkpoint is updated.

## License

Copyright 2019, University of Copenhagen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
