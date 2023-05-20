# Enhancing Dialogue Generation via Dynamic Graph Knowledge
This repository is the code and resources for the paper [Enhancing Dialogue Generation via Dynamic Graph Knowledge]() 

Here, our approach implements [BART](https://arxiv.org/abs/1910.13461) as our base language model, so for the convenience we call it **SaBART** (Subgraph-Aggregation BART).

## Instructions

This project is mainly implemented with following tools:
- **Pytorch** 
- **DGL** 
- [pytorch-lightning](https://www.pytorchlightning.ai/) framework
- The initial checkpoints of pretrained models come from [Hugginface](https://huggingface.co).

So if you want to run this code, you must have following preliminaries:
- Python 3 or Anaconda (mine is 3.8)
- [Pytorch](https://pytorch.org/) (mine is 1.11.0)
- transformers (a package for [huggingface](https://huggingface.co/facebook/bart-base)) (mine is 4.21.3)
- [pytorch-lightning (a package)](https://www.pytorchlightning.ai/) (mine is 1.8.2)
- **DGL** (mine is 0.9) 

**It is worth mentioning that the installation of previous DGL is not so easy. I found out an interesting fact that if DGL is not compatible to Pytorch, when running this pytorch code with cuda it may give some interesting errors. I have no solution to this issue, as it depends on the individual environment.**

## Datasets and Resources

### Directly Download Dataset and Resources
To reproduce our work you need to download following files:

- Processed data (unzip them to be `datasets/cadge` directory): [dropbox](https://www.dropbox.com/s/ydtdqef2344p9m0/cadge.zip?dl=0)

- The raw data come from the paper [CCM](https://github.com/thu-coai/ccm), and you can click this [data link](https://cloud.tsinghua.edu.cn/f/d367736aaec64d399b1b/?dl=1) to directly download it.  (put it to `resources/commonsense_conversation_dataset`)

You need to download the raw data only if you want to reproduce the dataset by yourself.

### Preprocess Dataset From Scratch

Make sure you have `resources/commonsense_conversation_dataset` ready.

Download [rel2words.txt](https://www.dropbox.com/s/0wetcr2o1wa7z5f/rel2words.txt?dl=0) from Dropbox, and put it to `resources/rel2words.txt`.

Run `python tasks/chatbot/preprocess.py --model_name_or_path=facebook/bart-base` to get the dataset at `datasets/cadge`.

### The introduction of the dataset
The structure of `datasets`should be like this:
```markdown
├── datasets/cadge
      └── `id2triple.txt`    
      └── `id2word.txt`     
      └── `rel2word.txt` 
      └── `testset.txt` 
      └── `trainset.txt` 
      └── `triple2id.txt`
      └── `valset.txt`
      └── `word2id.txt`
```

## Quick Start

### 1. Install packages
```shell
pip install -r requirements.txt
```
And you have to install **Pytorch** from their [homepage](https://pytorch.org/get-started/locally/).

### 2. Collect Datasets and Resources

As mentioned above.

### 3. Run the code for training or testing

Please refer to the command examples listed in `python_commands.sh`:

For example, for our model:
```shell
python tasks/chatbot/train.py --data_dir=datasets/cadge\
 --learning_rate=1e-4 --train_batch_size=218 --eval_batch_size=24 --model_name_or_path=resources/external_models/facebook/bart-base \
 --output_dir=output/chatbot --model_name chatbot_onehop --experiment_name=chatbot_onehop-cadge\
 --max_src_len 512 --max_tgt_len 512\
 --val_check_interval=0.1 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=1 --num_sanity_val_steps=1
```

```shell
python tasks/chatbot/test.py --data_dir=datasets/cadge\
 --eval_batch_size=256 --model_name_or_path=resources/external_models/facebook/bart-base \
 --output_dir=output/chatbot --model_name chatbot_onehop --experiment_name=chatbot_onehop-cadge
```

Revise the parameters according to your demand.

## Notation
We also tried to upgrade the code to Pytorch 2.0 with corresponding packages of
pytorch-lightning, DGL, transformers, and we successfully ran it on CPUs.
However, when running on GPUs (cuda), it will have interesting errors. 
Some other people also reported similar errors online, but no practical 
solution to our case, so finally we have to change back to the latest 
version of Pytorch 1. Hope this can be addressed in the future. :-)


## Citation
If you found this repository or paper is helpful to you, please cite our paper. 
Currently we only have arxiv citation listed as follows:

This is the arxiv citation:
```angular2

```



