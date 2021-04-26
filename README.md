### CrossFit :weight_lifting:: A Few-shot Learning Challenge for Cross-task Generalization in NLP

This repository contains code to construct the _NLP Few-shot Gym_ :sweat_drops: using huggingface datasets, and the baseline methods used in _the CrossFit Challenge_ :weight_lifting:.

### Configure Environment

```bash
# Create a new conda environment (optional)
conda create -n crossfit python=3.6.9
conda activate crossfit
# For building the NLP Few-shot Gym
pip install datasets==1.4.0
# For reproducing the baseline methods
pip install torch==1.1.0 higher==0.2.1
pip install git+https://github.com/huggingface/transformers.git@7b75aa9fa55bee577e2c7403301ed31103125a35
```

### Building the NLP Few-shot Gym

```bash
# Build the NLP Few-shot Gym, specify --n_proc=10 means the tasks will be prosessed in parallel with 10 subprocesses. 
cd tasks
python _build_gym.py --build --n_proc=10
# Verify with MD5Sum
python _build_gym.py --verify
```

If the processing for any individual task goes wrong (e.g, some datasets are hosted on google drive and there is quota issue), you can re-try later by running the individual script.

```bash
# For example, if you want to construct glue_sst2
cd tasks
python glue_sst2.py
```

__Disclaimer:__ We use publicly-available datasets from huggingface datasets to construct the few-shot gym. We do not own, host or distribute the datasets. If you are the author of the resources we use and wish to remove the data from the repository, please contact us!

### Contact Us
If you find bugs in our code, encounter problems when running the code, or have suggestions for the CrossFit project, please reach out to Qinyuan (qinyuany@usc.edu) or Bill (yuchen.lin@usc.edu)!

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
- [ ] MD5 Check
- [ ] Fine-tune on a downstream task
- [ ] Multi-task learning baseline
- [ ] Meta-learning baseline
