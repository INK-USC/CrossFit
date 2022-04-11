import os
import json
import re
import string
import random

import numpy as np

from collections import Counter
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from .utils import MyQADataset, MyDataLoader
from .metrics import METRICS, evaluate

class NLPFewshotGymSingleTaskData(object):

    def __init__(self, logger, args, data_path, data_type, is_training):
        # should give the tasks used in this split in the var "tasks"
        self.data_path = data_path
        self.data_type = data_type
        
        self.data = []

        self.task_name = "_".join(self.data_path.split("/")[-1].split("_")[:-3])

        with open(data_path) as fin:
            lines = fin.readlines()

        # train_examples = []
        for line in lines:
            d = line.strip().split("\t")
            if len(d[1:]) == 0:
                print("empty line? {}: {}".format(self.task_name, d))
                continue
            self.data.append((d[0], d[1:]))    

        if self.data_type == "test":
            self.data = self.data[:10000] 

        self.is_training = is_training
        self.load = not args.debug
        self.logger = logger
        self.args = args

        self.metric = METRICS[self.task_name]
        # self.max_input_length = self.args.max_input_length
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.cache = None

        self.gen_early_stop = False

    def __len__(self):
        return len(self.data)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def decode_batch(self, tokens):
        return [self.decode(_tokens) for _tokens in tokens]

    def flatten(self, answers):
        new_answers, metadata = [], []
        for answer in answers:
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
        return new_answers, metadata

    def load_dataset(self, tokenizer, do_return=False):
        self.tokenizer = tokenizer
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        
        preprocessed_path = os.path.join(
            "/".join(self.data_path.split("/")[:-1]),
            self.data_path.split("/")[-1].replace(".tsv", "-{}.json".format(postfix)))
        
        
        if self.load and os.path.exists(preprocessed_path):
            # load preprocessed input
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, \
                    metadata = json.load(f)

        else:
            self.logger.info("Start tokenizing ... {} instances".format(len(self.data)))

            inputs = []
            outputs = []

            for dp in self.data:
                inputs.append(" [{}] {}".format(self.task_name, dp[0]))
                outputs.append(dp[1]) # is a list

            self.logger.info("Printing 3 examples")
            for i in range(3):
                self.logger.info(inputs[i])
                self.logger.info(outputs[i])

            outputs, metadata = self.flatten(outputs) # what is metadata?

            if self.args.do_lowercase:
                inputs = [input0.lower() for input0 in inputs]
                outputs = [output0.lower() for output0 in outputs]
            if self.args.append_another_bos:
                inputs = ["<s> "+input0 for input0 in inputs]
                outputs = ["<s> " +output0 for output0 in outputs]
            
            self.logger.info("Tokenizing Input ...")
            tokenized_input = tokenizer.batch_encode_plus(inputs,
                                                         pad_to_max_length=True,
                                                         max_length=self.args.max_input_length)
            self.logger.info("Tokenizing Output ...")
            tokenized_output = tokenizer.batch_encode_plus(outputs,
                                                       pad_to_max_length=True,
                                                       max_length=self.args.max_output_length)

            input_ids, attention_mask = tokenized_input["input_ids"], tokenized_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = tokenized_output["input_ids"], tokenized_output["attention_mask"]
            if self.load:
                preprocessed_data = [input_ids, attention_mask,
                                     decoder_input_ids, decoder_attention_mask,
                                     metadata]
                with open(preprocessed_path, "w") as f:
                    json.dump([input_ids, attention_mask,
                               decoder_input_ids, decoder_attention_mask,
                               metadata], f)

        self.dataset = MyQADataset(input_ids, attention_mask,
                                        decoder_input_ids, decoder_attention_mask,
                                        in_metadata=None, out_metadata=metadata,
                                        is_training=self.is_training)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False):
        self.dataloader = MyDataLoader(self.args, self.dataset, self.is_training)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions, verbose=False):
        assert len(predictions)==len(self), (len(predictions), len(self))
        predictions = [prediction.strip() for prediction in predictions]
        return evaluate(predictions, self.data, self.metric)
        # ems = []
        # for (prediction, dp) in zip(predictions, self.data):
        #     ems.append(get_exact_match(prediction.strip(), [dp[1]]))
        # return np.mean(ems)

    def save_predictions(self, predictions):
        assert len(predictions)==len(self), (len(predictions), len(self))

        predictions = ['n/a' if len(prediction.strip())==0 else prediction for prediction in predictions]
        prediction_text = [prediction.strip()+'\n' for prediction in predictions]
        save_path = os.path.join(self.args.output_dir, "{}_predictions.txt".format(self.args.prefix))
        with open(save_path, "w") as f:
            f.writelines(prediction_text)
        
        self.logger.info("Saved prediction in {}".format(save_path))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_exact_match(prediction, groundtruth):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([get_exact_match(prediction, gt) for gt in groundtruth])
    return (normalize_answer(prediction) == normalize_answer(groundtruth))

def get_f1_over_list(prediction, groundtruth):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([f1_score(prediction, gt) for gt in groundtruth])
    return f1_score(prediction, groundtruth)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

