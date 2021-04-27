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

from .utils import MyMetaLearningDataset, MyMetaLearningDataLoader

class NLPFewshotGymMetaLearningData(object):

    def __init__(self, logger, args, data_path, tasks, data_type, is_training):
        # should give the tasks used in this split in the var "tasks"
        self.data_path = data_path
        self.data_type = data_type
        
        self.data = []

        # keep "sorted" so that things are consistent across machines
        for task in sorted(tasks): 
            task_dir = os.path.join(self.data_path, task)
            files = sorted(os.listdir(task_dir))
            prefixes = []
            for filename in files:
                if not filename.endswith(".tsv"):
                    continue
                prefix = "_".join(filename.split("_")[:-1])
                if prefix not in prefixes:
                    prefixes.append(prefix)

            for prefix in prefixes:
                with open(os.path.join(task_dir, prefix + "_train.tsv")) as fin:
                    lines = fin.readlines()

                train_examples = []
                for line in lines:
                    d = line.strip().split("\t")
                    if len(d) < 2:
                        print(prefix)
                        print(line)
                        print("train")
                    train_examples.append((d[0], d[1:]))

                with open(os.path.join(task_dir, prefix + "_dev.tsv")) as fin:
                    lines = fin.readlines()
                    
                dev_examples = []
                for line in lines:
                    d = line.strip().split("\t")
                    if len(d) < 2:
                        print(prefix)
                        print(line)
                        print("dev")
                    dev_examples.append((d[0], d[1:]))

                self.data.append({
                    "task_name": task,
                    "task_prefix": prefix,
                    "train_examples": train_examples,
                    "dev_examples": dev_examples
                })
            

        # if args.debug:
        #     self.data = self.data[:40]
        # print(self.data[4])

        self.is_training = is_training
        # self.load = not args.debug
        self.logger = logger
        self.args = args

        self.metric = "F1"
        # self.max_input_length = self.args.max_input_length
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.cache = None

        self.load = not args.debug

        self.gen_early_stop = False

    def __len__(self):
        return len(self.data)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def decode_batch(self, tokens):
        return [self.decode(_tokens) for _tokens in tokens]

    def flatten(self, answers):
        # not sure what this means
        new_answers, metadata = [], []
        for answer in answers:
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
        return new_answers, metadata

    def load_dataset(self, tokenizer, do_return=False):
        self.tokenizer = tokenizer
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        split_identifier = self.args.custom_tasks_splits.split("/")[-1]
        if split_identifier.endswith(".json"):
            split_identifier = split_identifier[:-5]

        preprocessed_path = os.path.join(
            self.data_path,
            self.data_type + "-meta-{}-{}.json".format(split_identifier, postfix)
        )
        
        if self.load and os.path.exists(preprocessed_path):
            # load preprocessed input
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                train_input_ids, train_attention_mask, \
                train_decoder_input_ids, train_decoder_attention_mask, \
                train_metadata_task, train_metadata_questions, \
                dev_input_ids, dev_attention_mask, \
                dev_decoder_input_ids, dev_decoder_attention_mask, \
                dev_metadata_task, dev_metadata_questions = json.load(f)

        else:
            print("Start tokenizing ... {} instances".format(len(self.data)))

            train_inputs = []
            train_outputs = []
            dev_inputs = []
            dev_outputs = []
            train_metadata_task, train_metadata_questions = [], []
            train_st, train_ed = 0, 0
            dev_metadata_task, dev_metadata_questions = [], []
            dev_st, dev_ed = 0, 0

            for task in self.data:
                task_name = task["task_name"]

                for dp in task["train_examples"]:
                    train_inputs.append(" [{}] {}".format(task_name, dp[0]))
                    train_outputs.append([" " + item for item in dp[1]])
                    
                train_st = train_ed
                train_ed = train_ed + len(task["train_examples"])
                # train_metadata_questions += [(i, i+1) for i in range(train_st, train_ed)]
                train_metadata_task.append((train_st, train_ed))


                for dp in task["dev_examples"]:
                    dev_inputs.append(" [{}] {}".format(task_name, dp[0]))
                    dev_outputs.append([" " + item for item in dp[1]])

                dev_st = dev_ed
                dev_ed = dev_ed + len(task["dev_examples"])
                # dev_metadata_questions += [(i, i+1) for i in range(dev_st, dev_ed)]
                dev_metadata_task.append((dev_st, dev_ed))

            train_outputs, train_metadata_questions = self.flatten(train_outputs)
            dev_outputs, dev_metadata_questions = self.flatten(dev_outputs)
            

            print("Printing Examples ...")
            for i in range(3):
                print(train_inputs[i])
                print(train_outputs[i])
                print()
            for i in range(3):
                print(dev_inputs[i])
                print(dev_outputs[i])
                print()

            print(train_metadata_task[:5])
            print(dev_metadata_task[:5])

            # outputs, metadata = self.flatten(outputs) # what is metadata?

            if self.args.do_lowercase:
                train_inputs = [input0.lower() for input0 in train_inputs]
                train_outputs = [output0.lower() for output0 in train_outputs]
                dev_inputs = [input0.lower() for input0 in dev_inputs]
                dev_outputs = [output0.lower() for output0 in dev_outputs]
            if self.args.append_another_bos:
                train_inputs = ["<s> "+input0 for input0 in train_inputs]
                train_outputs = ["<s> " +output0 for output0 in train_outputs]
                dev_inputs = ["<s> "+input0 for input0 in dev_inputs]
                dev_outputs = ["<s> " +output0 for output0 in dev_outputs]
            
            print("Tokenizing Train Input ...")
            train_tokenized_input = tokenizer.batch_encode_plus(train_inputs,
                                                         pad_to_max_length=True,
                                                         max_length=self.args.max_input_length)
            print("Tokenizing Train Output ...")
            train_tokenized_output = tokenizer.batch_encode_plus(train_outputs,
                                                       pad_to_max_length=True,
                                                       max_length=self.args.max_output_length)

            print("Tokenizing Dev Input ...")
            dev_tokenized_input = tokenizer.batch_encode_plus(dev_inputs,
                                                         pad_to_max_length=True,
                                                         max_length=self.args.max_input_length)
            print("Tokenizing Dev Output ...")
            dev_tokenized_output = tokenizer.batch_encode_plus(dev_outputs,
                                                       pad_to_max_length=True,
                                                       max_length=self.args.max_output_length)

            train_input_ids, train_attention_mask = train_tokenized_input["input_ids"], train_tokenized_input["attention_mask"]
            train_decoder_input_ids, train_decoder_attention_mask = train_tokenized_output["input_ids"], train_tokenized_output["attention_mask"]
            
            dev_input_ids, dev_attention_mask = dev_tokenized_input["input_ids"], dev_tokenized_input["attention_mask"]
            dev_decoder_input_ids, dev_decoder_attention_mask = dev_tokenized_output["input_ids"], dev_tokenized_output["attention_mask"]
                        
            
            if self.load:

                with open(preprocessed_path, "w") as f:
                    json.dump([train_input_ids, train_attention_mask,
                                train_decoder_input_ids, train_decoder_attention_mask,
                                train_metadata_task, train_metadata_questions,
                                dev_input_ids, dev_attention_mask,
                                dev_decoder_input_ids, dev_decoder_attention_mask,
                                dev_metadata_task, dev_metadata_questions
                                     ], f)

        self.dataset = MyMetaLearningDataset(train_input_ids, train_attention_mask,
                                        train_decoder_input_ids, train_decoder_attention_mask,
                                        train_metadata_task, train_metadata_questions,
                                        dev_input_ids, dev_attention_mask,
                                        dev_decoder_input_ids, dev_decoder_attention_mask,
                                        dev_metadata_task, dev_metadata_questions,
                                        inner_bsz=self.args.inner_bsz,
                                        is_training=self.is_training)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False):
        self.dataloader = MyMetaLearningDataLoader(self.args, self.dataset, self.is_training)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions, verbose=False):
        # not used
        return 0.0

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

