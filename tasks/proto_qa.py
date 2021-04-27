import os
import datasets
import numpy as np
import wget
import json

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class ProtoQA(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "proto_qa"
        self.task_type = "text to text"
        self.license = "unknown"

    def get_train_test_lines(self, dataset):
        # only train set, manually split 20% data as test

        lines = self.map_hf_dataset_to_list(dataset, "train")

        np.random.seed(42)
        np.random.shuffle(lines)
        
        n = len(lines)

        train_lines = lines[:int(0.8*n)]
        test_lines = lines[int(0.8*n):]

        return train_lines, test_lines

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            lines.append((datapoint["question"]["normalized"], "\t".join(datapoint["answers"]["raw"].keys())))
        return lines

    def load_dataset(self):
        if not os.path.exists("../data/proto_qa"):
            os.makedirs("../data/proto_qa/")
        if not os.path.exists("../data/proto_qa/train.jsonl"):
            wget.download("https://raw.githubusercontent.com/iesl/protoqa-data/master/data/train/train.jsonl", "../data/proto_qa/")
        with open("../data/proto_qa/train.jsonl") as fin:
            lines = fin.readlines()
            dataset = [json.loads(line) for line in lines]
        return {"train": dataset}

def main():
    dataset = ProtoQA()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()