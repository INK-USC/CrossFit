## The CrossFit Challenge :weight_lifting: and The NLP Few-shot Gym :sweat_drops:

This repository contains code accomapnying our preprint paper "CrossFit :weight_lifting:: A Few-shot Learning Challenge for Cross-task Generalization in NLP" ([arXiv](https://arxiv.org/abs/2104.08835)).

### Quick Links
- [Configure Environment](#configure-environment)
- [Building the NLP Few-shot Gym](#building-the-nlp-few-shot-gym) :sweat_drops:
- [Baselines for the CrossFit Challenge](#baseline-methods) :weight_lifting:
  - [Direct Fine-tuning](#fine-tune-a-single-few-shot-task)
- [Contact Us](#contact-us)

### Configure Environment

```bash
# Create a new conda environment (optional)
conda create -n crossfit python=3.6.9
conda activate crossfit
# For building the NLP Few-shot Gym
pip install datasets==1.4.0
# For reproducing the baseline methods
pip install torch==1.1.0 higher==0.2.1 scikit-learn==0.24.1 scipy==1.4.1 rouge==1.0.0
pip install git+https://github.com/huggingface/transformers.git@7b75aa9fa55bee577e2c7403301ed31103125a35
```

### Building the NLP Few-shot Gym

```bash
# Build the NLP Few-shot Gym (estimated time of completion: 3 hours)
# --n_proc=10 means the tasks will be prosessed in parallel with 10 subprocesses. 
cd tasks
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
We use publicly-available datasets from huggingface datasets to construct the few-shot gym. 
We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. 
If you are the owner of the resources we use and wish to remove/update them, please contact us!

### Baseline Methods

:smiley: Please check `./example_scripts` for more examples!

#### Fine-tune a single few-shot task
Here we take BoolQ as an example. There are five different samples of train/dev for BoolQ in the directory `data/boolq/`. For _each_ sample, we do a grid search over learning rate (1e-5, 2e-5, 5e-5) and batch size (2, 4, 8). The results are saved in a csv file in the `--output_dir`.

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

### Tools for Analyzing Results

### Acknowledgment

### Contact Us
If you find bugs in our code, encounter problems when running the code, or have suggestions for the CrossFit project, please reach out to Qinyuan (qinyuany@usc.edu) and Bill (yuchen.lin@usc.edu)!

If you used our code in your study, or find our paper useful, please cite us:
```
@article{ye2021crossfit,
  title={CrossFit: A Few-shot Learning Challenge for Cross-task Generalization in NLP},
  author={Ye, Qinyuan and Lin, Bill Yuchen and Ren, Xiang},
  journal={arXiv preprint arXiv:2104.08835},
  year={2021}
}
```

### To-do
We will update the code as soon as possible!
- [ ] Fine-tune on a downstream task
- [ ] Multi-task learning baseline
- [ ] Meta-learning baseline
