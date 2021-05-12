## The CrossFit Challenge :weight_lifting: and The NLP Few-shot Gym :sweat_drops:

This repository contains code accompanying our preprint paper "CrossFit :weight_lifting:: A Few-shot Learning Challenge for Cross-task Generalization in NLP" ([arXiv](https://arxiv.org/abs/2104.08835)).

CrossFit :weight_lifting: is a task setup which aims at building few-shot learners that generalize across diverse NLP tasks. For example, we explore whether models trained with _non-classification_ tasks becomes good few-shot learner for _classfication_ tasks; whether models trained with _non-MRC QA_ tasks becomes good few-shot learners for _MRC QA_ tasks.

NLP Few-shot Gym :sweat_drops: is a repository of 160 different NLP tasks that we gather from existing open-access datasets. We manually create a two-level task ontology to analyze cross-task generalization in different settings.

### Quick Links
- [Configure Environment](#configure-environment)
- [Building the NLP Few-shot Gym](#building-the-nlp-few-shot-gym) :sweat_drops:
- [Baselines for the CrossFit Challenge](#baseline-methods) :weight_lifting:
  - [Direct Fine-tuning](#fine-tune-a-single-few-shot-task)
  - [Upstream Learning (Multi-task and MAML)](#upstream-learning)
- [Download Our Checkpoints](#download-our-checkpoints)
- [Contact Us](#contact-us)

***
### Configure Environment

```bash
# Create a new conda environment (optional)
conda create -n crossfit python=3.6.9
conda activate crossfit
# For building the NLP Few-shot Gym
pip install datasets==1.4.0 py7zr wget
# For reproducing the baseline methods
pip install torch==1.1.0 higher==0.2.1 scikit-learn==0.24.1 scipy==1.4.1 rouge==1.0.0
pip install git+https://github.com/huggingface/transformers.git@7b75aa9fa55bee577e2c7403301ed31103125a35
```
***
### Building the NLP Few-shot Gym

The following code will automatically prepare the data using :hugs: [huggingface datasets](https://github.com/huggingface/datasets), reconstruct the few-shot train/dev sets we sampled, and verify the files with MD5Sum. The processing will take roughly 3 hours.

```bash
cd tasks
# Construct the gym
# --n_proc=10 means the tasks will be prosessed in parallel with 10 subprocesses. 
python _build_gym.py --build --n_proc=10
# Verify with MD5Sum
python _build_gym.py --verify
```

If the processing is successful, the verification script will output `[Success] All files are consistent.`

If the processing for any individual task goes wrong (e.g., some datasets are hosted on google drive and there is daily quota issue), you can re-try later by running individual scripts.

```bash
# For example, if you want to construct glue_sst2
cd tasks
python glue_sst2.py
```

__Disclaimer:__ 
We use publicly-available datasets from :hugs: huggingface datasets to construct the few-shot gym. 
We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use them. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.
If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included, please [contact us](#contact-us)!

***
### Baseline Methods

:smiley: Please check `./example_scripts` for more examples!

#### Fine-tune a single few-shot task
Here we take BoolQ as an example. There are five different samples of train/dev sets for BoolQ in the directory `data/boolq/`. For _each_ sample, we do a grid search over learning rate (1e-5, 2e-5, 5e-5) and batch size (2, 4, 8). 
This script will not save the final model, however the results will be logged in a csv file in `--output_dir`.

```bash
python tune_hps_singletask.py \
--task_dir data/boolq/ \
--do_train \
--do_predict \
--learning_rate_list 1e-5 2e-5 5e-5 \
--bsz_list 2 4 8 \
--total_steps 1000 \
--eval_period 100 \
--warmup_steps 100 \
--model facebook/bart-base \
--output_dir models/singletask-boolq \
--predict_batch_size 32 \
```

Notes:
- The script will load the original bart-base weights by default. If you want to fine-tune pre-trained weights from a file, please specify `--checkpoint $CHECKPOINT`.
- If you want to fine-tune with your own task, please process your data in the same format as in `data/boolq/`.
- We provide a script that does not tune hyperparameters and saves the final model in `./example_scripts`.

#### Upstream Learning

Upstream learning refers to the stage between general pre-training and down-stream fine-tuning. In this stage we allow access to a set of training tasks, and we test the few-shot learning ability on a set of test tasks after this stage. Please check Table 1 in our preprint paper for more details about task partitions.

We include two upstream learning methods: multi-task learning and MAML (model-agnostic meta-learning). We are currently working on first-order meta-learning algorithms (First-order MAML and Reptile)!

<details>
<summary>Multi-task Learning</summary>

```bash
TASK_SPLIT=dataloader/custom_task_splits/random.json
python cli_multitask.py \
--do_train \
--train_dir data \
--custom_tasks_splits ${TASK_SPLIT} \
--total_steps 16980 \
--warmup_steps 1018 \
--model facebook/bart-base \
--output_dir models/upstream-multitask \
--train_batch_size 32 \
--num_train_epochs 10;
```
</details>

<details>
<summary>MAML</summary>

```bash
TASK_SPLIT=dataloader/custom_task_splits/random.json
python cli_maml.py \
--do_train \
--learning_rate 1e-5 \
--output_dir models/upstream-maml \
--custom_tasks_splits ${TASK_SPLIT} \
--total_steps 6000 \
--warmup_steps 360 \
--train_batch_size 1 \
--gradient_accumulation_steps 4 \
--num_train_epochs 40;
```

MAML is memory intensive. The experiment above is done with a Quadro RTX 8000 GPU (48GB). If you want to reduce memory usage, please reduce `--inner_bsz`.

</details>
<br>

Notes:
- You can specify what tasks to use during upstream learning by creating a json file and passing it to `--custom_tasks_splits`.
- The `--total_steps` in the scripts above are pre-computed so that the learning rate decreases to zero linearly during learning. We also pre-compute `--warmup_steps` to be 6% of the total steps.

***

### Download Our Checkpoints

Here we provide the checkpoints after upstream learning.

| Task Partition | Multi-task | Meta-learn |
| ----------- | ----------- | ----------- |
| 1. Random     | [multi-task-random-bart-base.pt](https://drive.google.com/file/d/1jz-hg5hvygeBSDpORw2Vq-a_a0KWfT4y/view?usp=sharing)       | [meta-learn-random-bart-base.pt](https://drive.google.com/file/d/1dPNaScWO3iktB5EZneDWr8ZSNq0DAuvT/view?usp=sharing)


:smiley: Please stay tuned for more checkpoints!

***

### Useful Tools
- `./example_scripts/finetune_a_list_of_tasks.sh` will help you fine-tune a list of tasks sequentially, given a certain model initialization.
- `./example_scripts/collect_results.py` will read each `results.csv` files in a given directory, then compute mean and standard deviation of dev/test performance.

***

### Acknowledgment
We thank authors and crowd-workers of all resources used in our study! This work would not have been possible without your efforts. We thank :hugs: [huggingface datasets](https://github.com/huggingface/datasets) team for making datasets more accessible. Our code is modified from [shmsw25/bart-closed-book-qa](https://github.com/shmsw25/bart-closed-book-qa), thanks to the authors!

***

### Contact Us
If you find bugs in our code, encounter problems when running the code, or have suggestions for the CrossFit project, please submit an issue, or reach out to Qinyuan (qinyuany@usc.edu) and Bill (yuchen.lin@usc.edu)!

If you used our code in your study, or find our paper useful, please cite us:
```
@article{ye2021crossfit,
  title={CrossFit: A Few-shot Learning Challenge for Cross-task Generalization in NLP},
  author={Ye, Qinyuan and Lin, Bill Yuchen and Ren, Xiang},
  journal={arXiv preprint arXiv:2104.08835},
  year={2021}
}
```

