# COPO: Count-based Online Preference Optimization

Code for *Online Preference Alignment for Language Models via Count-based Exploration*. Paper [link](https://openreview.net/forum?id=cfKZ5VrhXt).


## Prerequisites

To run the code in this project, first, create a Python virtual environment using e.g. Conda:

```shell
conda create -n rlhf python=3.10 && conda activate rlhf
```

You can then install the remaining package dependencies as follows:

```shell
 python -m pip install .
```

You will also need Flash Attention 2 installed, which can be done by running:

```shell
python -m pip install flash-attn==2.3.6 --no-build-isolation
```

Next, log into your Hugging Face account as follows:

```shell
huggingface-cli login
```

Finally, install Git LFS so that you can push models to the Hugging Face Hub:

```shell
sudo apt-get install git-lfs
```

## Run the code 

To train COPO on [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), you need to first apply for the access. 

You should find Huggingface key and Wandb key in your account, and then replace the `xxxx` in `HUGGINGFACE_API_KEY` and `WANDB_API_KEY` in  `recipes/launch_llama3_copo.sh`. Also, you should replace the path in `BASE_DIR` with your own path to `COPO` root directory.

After the above preparation, train COPO on Meta-Llama-3-8B-Instruct:
```shell
cd COPO/recipes
sh launch_llama3_copo.sh
```


After training, the collected data will be saved in `dataset/`, and the learned model will be saved in `data/`. For each iteration, we use an independent dictionary to save it, the final model will be saved in `data/COPO-Llama-3-8B-Instruct-iter-3/merge`.


## File organization

The important implementation is saved at:

- `scripts/online_feedback.py`: generate responses based on a prompt set of preference datasets, and adopt PairRM to rank responses to obtain the online preference dataset.

- `scripts/count_trainer.py`: train the coin-flipping network based on the generated data, which is used to perform count-based exploration.

- `scripts/copo_trainer.py`: the main process to perform updates with COPO algorithm, which combines online DPO and CFN objective.

- `scripts/merge_model.py`: merge the Lora layer and the main network to a new model


## Acknowledgement
This repo is built upon [The Alignment Handbook](https://github.com/huggingface/alignment-handbook). We thank the authors for their great work. 
